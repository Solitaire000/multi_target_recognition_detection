"""
train.py
========
完整训练流程：两阶段训练策略
  Phase 1: 用CV伪标签预训练（粗监督）
  Phase 2: 用人工精标注微调（精监督）+ 消融实验框架
"""

from __future__ import annotations

import os
import json
import time
import argparse
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from model import *
from data_pipeline import *
from metrics import *

# ══════════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════════
# 日志输出配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ 的上一级 = 项目根目录
# config配置
DEFAULT_CONFIG = {
    # 模型
    'pretrained': True,
    'nominal_ar': 3.0,
    'px_per_um': 1.0,

    # 训练
    'image_size': 640,
    'batch_size': 16,
    'num_workers': 4,
    'phase1_epochs': 20,  # 伪标签预训练
    'phase2_epochs': 10,  # 人工标注微调
    'lr_phase1': 1e-3,
    'lr_phase2': 5e-5,  # 微调用较小学习率
    'weight_decay': 1e-4,
    'grad_clip': 0.5,
    'sigma_warmup_epochs': 5,

    # 路径（均相对于项目根目录推导）
    # train 路径
    'hrnet_root': str(PROJECT_ROOT / "data" / "model" / "model.safetensors"),
    'data_root': str(PROJECT_ROOT / "data" / "raw"),
    'output_dir': str(PROJECT_ROOT / "experiments"),
    # predict 路径
    'pth_dir': str(PROJECT_ROOT / "experiments/exp_full_1783228941/best_phase2.pth"),
    # 'predict_data_dir': str(PROJECT_ROOT / "data" / "raw"/"predict"),
    'predict_data_dir': str(PROJECT_ROOT / "data" / "raw"/"predict_image"),
    'predict_result_dir': None,

    # resume：传入某个实验目录下的 last_phase1.pth / last_phase2.pth 路径，即可从中断处
    'resume': None,

    # Dataest
    'dataset_config': dataset_config,
    # 消融实验
    'ablation_mode': 'full',  # 'full' | 'no_uncertainty' | 'no_fpn' | 单任务名
}


# ══════════════════════════════════════════════════════════════
#  训练循环
# ══════════════════════════════════════════════════════════════

class Trainer:

    def __init__(self, config: Dict):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        self.sigma_warmup_epochs = config.get('sigma_warmup_epochs', 5)
        self.scaler = torch.cuda.amp.GradScaler()
        # 模型
        self.model = GSGProbeNet(config).to(self.device)
        self.loss_fn = MultiTaskLoss(num_tasks=4).to(self.device)  # +probe_state（遮挡率/失焦度/可见比例）
        # 管理和计算评估指标
        self.metrics = GSGProbeMetrics()

        resume_path = config.get('resume')
        if resume_path:
            self.out_dir = Path(resume_path).parent
            self.out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Resume] 复用已有实验目录: {self.out_dir}")
        else:
            exp_name = f"exp_{config['ablation_mode']}_{int(time.time())}"
            self.out_dir = Path(config['output_dir']) / exp_name
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir = None

        # 保存配置（resume 时不覆盖原有 config.json，另存一份带时间戳的，保留每次启动的现场记录）
        config_filename = 'config.json' if not resume_path else f'config_resume_{int(time.time())}.json'
        with open(self.out_dir / config_filename, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        # 表示负无穷大的浮点数，即比任何有限数都小的值
        self.best_val_metric = -float('inf')
        self.history: Dict[str, list] = defaultdict(list)

    # ── 第一阶段：伪标签预训练 ─────────────────────────────────
    def phase1_pretrain(self):
        """用CV基线生成的伪标签预训练，建立基础特征"""
        logger.info("=" * 50)
        logger.info("Phase 1: 伪标签预训练")
        logger.info("=" * 50)

        # 训练集
        train_ds = GSGProbeDataset(
            self.cfg['data_root'], split='train',
            transforms=build_train_transforms(self.cfg['image_size']),
            use_pseudo_labels=True
        )
        # 验证集
        val_ds = GSGProbeDataset(
            self.cfg['data_root'], split='val',
            transforms=build_val_transforms(self.cfg['image_size']),
            use_pseudo_labels=True
        )
        # 训练集加载器
        train_loader = DataLoader(
            train_ds,  # 训练数据集（Dataset 对象）
            batch_size=self.cfg['batch_size'],  # 每个批次的样本数量
            shuffle=True,  # 是否打乱数据顺序
            num_workers=self.cfg['num_workers'],  # 多线程加载数据的线程数
            pin_memory=True,  # 是否将数据固定在 GPU 内存（加速数据传输）
            drop_last=True,  # 是否丢弃最后一个不完整的批次
            collate_fn=collate_fn  # 自定义数据整理函数
        )
        # 验证集加载器
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=self.cfg['num_workers'],
            collate_fn=collate_fn
        )
        # 构建优化器
        optimizer = AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),  # 待优化的参数
            lr=self.cfg['lr_phase1'],  # 初始学习率
            weight_decay=self.cfg['weight_decay']  # 权重衰减系数
        )
        # 构建学习率调度器
        scheduler = CosineAnnealingLR(
            optimizer,  # 绑定的优化器
            T_max=self.cfg['phase1_epochs'],  # 余弦周期的迭代次数（通常为总 epoch 数）
            eta_min=1e-6  # 最小学习率
        )


        start_epoch = self._resume_state('phase1', optimizer, scheduler, self.cfg['phase1_epochs'])

        for epoch in range(start_epoch, self.cfg['phase1_epochs']):
            # 测试代码
            # val_metrics = self._val_epoch(val_loader, epoch, 'Phase1')
            # train_metrics = {}
            # self._log_epoch(epoch, train_metrics, val_metrics, 'Phase1')
            # self._save_checkpoint(epoch, val_metrics, phase='phase1')

            train_metrics = self._train_epoch(train_loader, optimizer, epoch, 'Phase1')
            val_metrics = self._val_epoch(val_loader, epoch, 'Phase1')
            scheduler.step()
            self._log_epoch(epoch, train_metrics, val_metrics, 'Phase1')
            self._save_checkpoint(epoch, val_metrics, phase='phase1')
            self._save_last_checkpoint(epoch, 'phase1', optimizer, scheduler, val_metrics)

        logger.info("Phase 1 完成，保存预训练权重")
        self.cfg['pth_dir'] = self.out_dir / 'phase1_pretrained.pth'
        torch.save(self.model.state_dict(), self.out_dir / 'phase1_pretrained.pth')

    # ── 第二阶段：人工标注微调 ─────────────────────────────────

    def phase2_finetune(self):
        """用人工精标注微调，冻结骨干网络前几层"""
        logger.info("=" * 50)
        logger.info("Phase 2: 人工标注微调")
        logger.info("=" * 50)

        # 冻结骨干网络前两层（保留低层纹理特征，只更新任务头）
        # 注意：requires_grad 不会被 state_dict 保存/恢复，所以无论是否 resume 都必须重新执行一次。
        self._freeze_backbone_layers(num_frozen_layers=2)

        train_ds = GSGProbeDataset(
            self.cfg['data_root'], split='train',
            transforms=build_train_transforms(self.cfg['image_size']),
            use_pseudo_labels=False  # 仅用人工标注
        )
        val_ds = GSGProbeDataset(
            self.cfg['data_root'], split='val',
            transforms=build_val_transforms(self.cfg['image_size'])
        )
        # 训练集加载器
        train_loader = DataLoader(
            train_ds, batch_size=self.cfg['batch_size'],
            shuffle=True, num_workers=self.cfg['num_workers'],
            pin_memory=True, drop_last=True,
            collate_fn=collate_fn
        )
        # 验证集加载器
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=self.cfg['num_workers'],
            collate_fn=collate_fn
        )
        lr = self.cfg['lr_phase2']
        # 学习率分层
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': lr * 0.1},
            {'params': self.model.fpn.parameters(), 'lr': lr * 0.5},
            {'params': self.model.probe_det.parameters(), 'lr': lr},
            {'params': self.model.mask_head.parameters(), 'lr': lr},
            {'params': self.model.kp_head.parameters(), 'lr': lr},  # RoIKeypointHead
            {'params': self.model.calib_det.parameters(), 'lr': lr},
            {'params': self.loss_fn.parameters(), 'lr': lr},
        ]
        optimizer = AdamW(param_groups, weight_decay=self.cfg['weight_decay'])
        # 构建学习率调度器
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.cfg['phase2_epochs'], eta_min=1e-7
        )

        start_epoch = self._resume_state('phase2', optimizer, scheduler, self.cfg['phase2_epochs'])

        if start_epoch == 0:
            pretrained_path = self.best_dir
            self.best_val_metric = -float('inf')
            if pretrained_path:
                state_dict = torch.load(pretrained_path, map_location=self.device)
                self.model.load_state_dict(state_dict["model_state_dict"])
                logger.info(f"加载预训练权重: {pretrained_path}")

        for epoch in range(start_epoch, self.cfg['phase2_epochs']):
            train_metrics = self._train_epoch(train_loader, optimizer, epoch, 'Phase2')
            val_metrics = self._val_epoch(val_loader, epoch, 'Phase2')
            scheduler.step()
            self._log_epoch(epoch, train_metrics, val_metrics, 'Phase2')
            self._save_checkpoint(epoch, val_metrics, phase='phase2')
            self._save_last_checkpoint(epoch, 'phase2', optimizer, scheduler, val_metrics)

        logger.info("Phase 2 完成，保存预训练权重")
        torch.save(self.model.state_dict(), self.out_dir / 'phase2_trained.pth')

    # ── 单个epoch ─────────────────────────────────────────────
    def _train_epoch(self, loader, optimizer, epoch, phase_name):
        self.model.train()
        self.loss_fn.train()
        total_losses = {}
        total_active_counts = {}
        num_batches = 0
        sigma_warmup_done = (epoch >= self.sigma_warmup_epochs) or (phase_name == 'Phase2')
        for batch_idx, batchs in enumerate(loader):
            images = batchs['image'].to(self.device)
            optimizer.zero_grad()
            gt_rboxes_list = batchs['probe_rboxes']
            with torch.cuda.amp.autocast():
                outputs = self.model(images, rboxes_list=gt_rboxes_list)
                task_losses, task_stats = self._compute_losses(outputs, batchs, sigma_warmup_done)
                total_loss = self.loss_fn(task_losses, task_stats)

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(optimizer)
            all_params = list(self.model.parameters()) + list(self.loss_fn.parameters())
            grad_norm = nn.utils.clip_grad_norm_(all_params, self.cfg['grad_clip'])
            self.scaler.step(optimizer)
            self.scaler.update()

            for k, v in task_losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.detach().item()
                total_active_counts[k] = total_active_counts.get(k, 0) + 1
            total_losses['total'] = total_losses.get('total', 0.0) + total_loss.detach().item()
            num_batches += 1

            if batch_idx % 10 == 0:
                loss_str = " | ".join(
                    f"{name:<10}= {value.item():>5.4f}"
                    for name, value in task_losses.items()
                )
                weight_str = " | ".join(
                    f"{name:<10}= {weight:>5.4f}"
                    for name, weight in self.loss_fn.task_weights(task_stats).items()
                )

                logger.info(
                    f"[{phase_name:<6}] E{epoch:02d} [{batch_idx:03d}/{len(loader):03d}] "
                    f"grad_norm= {grad_norm:.4f} Loss= {total_loss.item():>5.4f} // "
                    f"L: {loss_str} // W: {weight_str}"
                )

        return {
            k: v / max(total_active_counts.get(k, num_batches), 1)
            for k, v in total_losses.items()
        }

    def _val_epoch(self, loader: DataLoader, epoch: int, phase_name: str) -> Dict:
        self.model.eval()
        self.metrics.reset()
        use_cuda = self.device.type == 'cuda'

        with torch.no_grad():
            for batchs in loader:
                images = batchs['image'].to(self.device)

                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = self.model(images)
                if use_cuda:
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                per_image_ms = elapsed_ms / max(len(images), 1)

                results = self.model.postprocess(outputs, x_shape=tuple(images.shape))
                # 拆开batchs，单张图片推理，无误
                for i in range(len(images)):
                    single_batch = {
                        k: (v[i] if (isinstance(v, (list, tuple))
                                     and not isinstance(v, dict)
                                     and len(v) == len(images))
                            else (v[i] if isinstance(v, torch.Tensor)
                                          and v.shape[0] == len(images)
                                  else v))
                        for k, v in batchs.items()
                    }
                    self.metrics.update(results[i], single_batch, inference_time_ms=per_image_ms)

        return self.metrics.compute()

    def _compute_losses(self, outputs: Dict, batch: Dict, sigma_warmup_done: bool) -> tuple[Dict, Dict]:

        device = self.device
        losses = {}
        task_stats = batch['task_stats']  # {'probe_det': int, 'keypoint': int, 'calib_det': int}

        # ════════════════════════════════════════════════════
        # Task A: Probe Detection + RoI Mask
        # ════════════════════════════════════════════════════
        if task_stats['probe_det'] > 0:
            det_loss = self._det_loss_from_targets(
                pred_cls_list=outputs['probe_cls'],
                pred_reg_list=outputs['probe_reg'],
                pred_ctr_list=outputs['probe_ctr'],
                cls_tgt_list=batch['probe_cls_target_list'],
                bbox_tgt_list=batch['probe_bbox_target_list'],
                ctr_tgt_list=batch['probe_centerness_target_list'],
                angle_tgt_list=batch['probe_angle_target_list'],
                tag='probe',
            )
            losses['probe_det'] = det_loss

        if task_stats['keypoint'] > 0:
            mask_roi_h, mask_roi_w = self.model.mask_roi_size
            gt_mask_roi = build_mask_targets_roi(
                rboxes_list=outputs['kp_rboxes_list'],
                masks_list=batch['probe_masks'],
                roi_size=(mask_roi_h, mask_roi_w),
                device=device,
            )
            if gt_mask_roi is not None and outputs['mask_logits'].shape[0] == gt_mask_roi.shape[0]:
                pred_mask_logits = outputs['mask_logits'].squeeze(1)  # (N, roi_h, roi_w)
                mask_bce = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_mask_roi, reduction='mean')
                mask_dice = self._dice_loss_binary(pred_mask_logits, gt_mask_roi)
                mask_loss = mask_bce + mask_dice
                losses['probe_det'] = losses.get('probe_det', torch.tensor(0., device=device)) + mask_loss

        # ════════════════════════════════════════════════════
        # Task B: RoI Keypoint Loss
        # ════════════════════════════════════════════════════
        if task_stats['keypoint'] > 0:
            rboxes_list = outputs['kp_rboxes_list']
            roi_h, roi_w = self.model.kp_head.roi_size

            gt_roi_heatmaps, gt_vis_roi, gt_xy_roi = build_keypoint_targets_roi(
                rboxes_list=rboxes_list,
                keypoints_list=batch['probe_keypoints'],
                visibility_list=batch['probe_visibility'],
                roi_size=(roi_h, roi_w),
                num_keypoints=self.model.kp_head.num_keypoints,
                device=device,
            )

            if gt_roi_heatmaps is not None:
                gt_roi_heatmaps = gt_roi_heatmaps.to(device)
                gt_vis_roi = gt_vis_roi.to(device)
                gt_xy_roi = gt_xy_roi.to(device)
                losses['keypoint'] = self._compute_keypoint_loss(
                    outputs['kp_heatmap'], outputs['kp_log_sigma2'],
                    gt_roi_heatmaps, gt_vis_roi, gt_xy_roi, rboxes_list, sigma_warmup_done,
                )
                # 逐关键点可见性分类（0缺失/1遮挡/2可见），标签直接是现成的 visibility 字段，
                # 附加在 keypoint 任务里（同一批 RoI、同一个不确定度权重），不单开任务头权重。
                vis_cls_loss = self._compute_kp_visibility_loss(
                    outputs.get('kp_vis_logits'), gt_vis_roi,
                )
                if vis_cls_loss is not None:
                    losses['keypoint'] = losses['keypoint'] + 0.3 * vis_cls_loss

        # ════════════════════════════════════════════════════
        # Task D: Probe State Loss（遮挡率 / 失焦度 / 几何可见比例）
        # ════════════════════════════════════════════════════
        if task_stats.get('probe_state', 0) > 0:
            state_loss = self._compute_probe_state_loss(
                outputs.get('state_logits'),
                batch['probe_occlusion_ratio'],
                batch['probe_defocus_level'],
                batch['probe_visible_ratio'],
            )
            if state_loss is not None:
                losses['probe_state'] = state_loss

        # ════════════════════════════════════════════════════
        # Task C: Calibration Detection Loss
        # ════════════════════════════════════════════════════
        if task_stats['calib_det'] > 0:
            losses['calib_det'] = self._det_loss_from_targets(
                pred_cls_list=outputs['calib_cls'],
                pred_reg_list=outputs['calib_reg'],
                pred_ctr_list=outputs['calib_ctr'],
                cls_tgt_list=batch['calib_cls_target_list'],
                bbox_tgt_list=batch['calib_bbox_target_list'],
                ctr_tgt_list=batch['calib_centerness_target_list'],
                angle_tgt_list=batch['calib_angle_target_list'],  # 新增
                tag='calib',
            )

        return losses, task_stats

    @staticmethod
    def _focal_loss(pred,
                    target,
                    alpha=0.25,
                    gamma=2.0,
                    reduction='mean',
                    eps=1e-6,
                    ):

        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        # logits → prob
        prob = torch.sigmoid(pred)
        # 防止 log(0)
        prob = torch.clamp(prob, eps, 1.0 - eps)
        ce_loss = -(
                target * torch.log(prob) +
                (1 - target) * torch.log(1 - prob)
        )
        # p_t
        p_t = target * prob + (1 - target) * (1 - prob)
        # focal weight
        focal_weight = (1 - p_t) ** gamma
        # alpha balance
        alpha_weight = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_weight * focal_weight * ce_loss
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    @staticmethod
    def _focal_loss_softmax(pred,
                            target,
                            alpha=0.25,
                            gamma=2.0,
                            reduction='mean',
                            eps=1e-6,
                            ):

        if target.dim() == 3:
            target = target.unsqueeze(1)  # [B,1,H,W]
        target = target.float()

        # softmax → 前景概率（通道1），与 _dice_loss 一致
        prob_fg = torch.softmax(pred, dim=1)[:, 1:2]  # [B,1,H,W]
        prob_fg = torch.clamp(prob_fg, eps, 1.0 - eps)

        ce_loss = -(
                target * torch.log(prob_fg) +
                (1 - target) * torch.log(1 - prob_fg)
        )
        p_t = target * prob_fg + (1 - target) * (1 - prob_fg)
        focal_weight = (1 - p_t) ** gamma
        alpha_weight = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_weight * focal_weight * ce_loss

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss

    def _dice_loss(
            self,
            pred,
            target,
            smooth=1.0,
            reduction='mean',
    ):
        """
        pred:   (B, 2, H, W) logits
        target: (B, H, W)
        """
        # -----------------------------
        # softmax -> foreground prob
        # -----------------------------
        prob = torch.softmax(pred, dim=1)[:, 1]
        # target -> float
        target = target.float()
        # flatten
        prob = prob.reshape(prob.size(0), -1)
        target = target.reshape(target.size(0), -1)
        intersection = (prob * target).sum(dim=1)
        union = prob.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        loss = 1.0 - dice
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss

    def _dice_loss_binary(self, pred_logits, target, smooth=1.0):
        """
        单通道 mask 头专用 dice loss（区别于 _dice_loss 的双通道 softmax 版本）。
        pred_logits: (N, H, W) logits；target: (N, H, W) ∈ {0,1}
        """
        prob = torch.sigmoid(pred_logits).reshape(pred_logits.size(0), -1)
        target = target.reshape(target.size(0), -1).float()
        intersection = (prob * target).sum(dim=1)
        union = prob.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return (1.0 - dice).mean()

    def _compute_keypoint_loss(self, pred_heatmap, pred_log_sigma2,
                               gt_heatmap, gt_visibility, gt_xy_roi, rboxes_list,
                               sigma_warmup_done: bool = True):
        pred_prob = torch.sigmoid(pred_heatmap)

        fg_weight = gt_heatmap.detach() * 4.0 + 1.0
        vis_mask = (gt_visibility > 0).float()  # (N, K)
        vis_4d = vis_mask[:, :, None, None].expand_as(pred_prob)
        valid = vis_4d.sum().clamp(min=1.0)

        # ── 分支一：热图形状 loss —— 永远非负，永远存在，永远是主项 ──
        mse = F.mse_loss(pred_prob, gt_heatmap, reduction='none')
        heatmap_loss = ((mse * fg_weight) * vis_4d).sum() / valid

        if not sigma_warmup_done:
            return heatmap_loss

        # ── 分支二：坐标级异方差 NLL —— 只用于标定不确定度，量级用像素而非[0,1]概率 ──
        #
        # [根治修复] 原实现直接在 32×32 的 ROI 网格坐标上计算 sq_err / NLL，而验证阶段
        # kp_pck_*px 等指标是在原图物理像素坐标系下计算的——两者不是同一把尺子。
        # 探针框的物理尺寸（w,h）在样本间差异很大，网络完全可以在"ROI网格尺度"上把 sigma
        # 学得很小、NLL 降得很低（训练日志显示 loss 接近 0 甚至为负），但换算回原图物理像素后
        # 误差依然很大——这正是之前"训练损失很好看、验证 PCK 却很差"的根因。
        # 这里把 ROI 网格坐标统一换算回"物理像素局部坐标"（与旋转框标注时使用的 dx,dy 同一量纲，
        # 因为局部坐标系是纯旋转 + 尺度还原，不改变欧氏距离的物理意义），
        # 使 NLL 优化目标和 PCK 评估指标处在同一个物理尺度上。
        mu_xy = soft_argmax_2d(pred_heatmap, self.model.kp_head.temperature)  # (N,K,2)，ROI网格坐标
        log_sigma2 = pred_log_sigma2.clamp(-2.0, 6.0)  # (N,K)，标量方差（物理像素²尺度）
        sigma2 = torch.exp(log_sigma2)

        roi_h, roi_w = self.model.kp_head.roi_size
        inst_w, inst_h = self._flatten_instance_wh(rboxes_list, device=mu_xy.device)
        if inst_w is None:
            # 理论上不该发生（gt_xy_roi 不为 None 时 rboxes_list 必然非空），兜底跳过 NLL 分支
            return heatmap_loss

        # local_px = (grid_coord / roi_size - 0.5) * physical_size（与 build_keypoint_targets_roi
        # 里 kp_x = (x_local/w + 0.5)*roi_w 互为逆变换）
        wh = torch.stack([inst_w, inst_h], dim=-1)[:, None, :]  # (N,1,2)
        mu_xy_px = (mu_xy / mu_xy.new_tensor([roi_w, roi_h]) - 0.5) * wh
        gt_xy_px = (gt_xy_roi / gt_xy_roi.new_tensor([roi_w, roi_h]) - 0.5) * wh

        sq_err_px = ((mu_xy_px - gt_xy_px) ** 2).sum(-1)  # (N,K)，单位：原图像素²
        # 2D 各向同性高斯 NLL（x,y 共享同一个 sigma）：log(sigma2) + err²/sigma2 (+ 常数项，不影响梯度)
        nll_coord = log_sigma2 + sq_err_px / (sigma2 + 1e-6)
        valid_k = vis_mask.sum().clamp(min=1.0)
        nll_coord = (nll_coord * vis_mask).sum() / valid_k

        # 系数给小：坐标NLL是"标定/精修"项，不应反客为主改变keypoint任务loss的总体量级和符号
        return heatmap_loss + 0.2 * nll_coord

    @staticmethod
    def _compute_kp_visibility_loss(vis_logits: Optional[torch.Tensor],
                                    gt_visibility: torch.Tensor) -> Optional[torch.Tensor]:
        """
        vis_logits:    (N, K, 3) —— 0缺失/1遮挡/2可见，来自 RoIKeypointHead.vis_cls_head
        gt_visibility: (N, K)    —— 数据集原生字段，值域 {0,1,2}，与 vis_logits 语义完全对齐，
                                     不需要任何标签映射/新标注。
        """
        if vis_logits is None or vis_logits.shape[0] == 0:
            return None
        if vis_logits.shape[0] != gt_visibility.shape[0]:
            return None  # 理论上不该发生：两者应来自同一批 RoI，防御性跳过
        logits_flat = vis_logits.reshape(-1, 3)  # (N*K, 3)
        target_flat = gt_visibility.reshape(-1).long().clamp(0, 2)  # (N*K,)
        return F.cross_entropy(logits_flat, target_flat)

    @staticmethod
    def _compute_probe_state_loss(
            state_logits: Optional[torch.Tensor],
            occ_gt_list: List[List[float]],
            defocus_gt_list: List[List[float]],
            visible_gt_list: List[List[float]],
    ) -> Optional[torch.Tensor]:
        """
        state_logits: (N_inst, 3) sigmoid前，[occlusion_ratio, defocus_level, visible_ratio]。
        三个 *_gt_list 均为 B*List[N_i] 的逐图逐实例列表，与 outputs['kp_rboxes_list']（=训练期间
        的 GT probe_rboxes）严格同序展平——extract_rotated_roi_feats 按同样的 (batch, instance)
        顺序遍历 rboxes_list，所以这里直接拼接即可对齐，不需要额外的 batch_idx 查找。
        """
        if state_logits is None or state_logits.shape[0] == 0:
            return None
        device = state_logits.device
        occ_gt = torch.tensor([v for lst in occ_gt_list for v in lst], device=device, dtype=torch.float32)
        defocus_gt = torch.tensor([v for lst in defocus_gt_list for v in lst], device=device, dtype=torch.float32)
        visible_gt = torch.tensor([v for lst in visible_gt_list for v in lst], device=device, dtype=torch.float32)
        if occ_gt.shape[0] != state_logits.shape[0]:
            return None  # 防御性跳过：实例数对不上时不构造错位的监督信号

        pred = torch.sigmoid(state_logits)  # (N,3)
        loss_occ = F.smooth_l1_loss(pred[:, 0], occ_gt)
        loss_defocus = F.smooth_l1_loss(pred[:, 1], defocus_gt)
        loss_visible = F.smooth_l1_loss(pred[:, 2], visible_gt)
        return loss_occ + loss_defocus + loss_visible

    @staticmethod
    def _flatten_instance_wh(rboxes_list, device):
        """
        按 build_keypoint_targets_roi 完全相同的遍历顺序，把每个探针实例的物理 (w,h)
        展平成 (N,) 张量，用于把 ROI 网格坐标换算回物理像素坐标。两处遍历顺序必须严格一致，
        否则会出现"坐标和框对不上"的新错位——因此这里不重新实现一遍逻辑，只做最小的字段提取。
        """
        ws, hs = [], []
        for boxes in rboxes_list:
            n_i = len(boxes)
            if n_i == 0:
                continue
            for inst_idx in range(n_i):
                box = boxes[inst_idx]
                ws.append(float(box[2]))
                hs.append(float(box[3]))
        if not ws:
            return None, None
        return (torch.tensor(ws, device=device, dtype=torch.float32),
                torch.tensor(hs, device=device, dtype=torch.float32))

    def _bbox_loss(self, pred, target, eps=1e-6):
        """
        pred/target: (N,4) = [dx, dy, w, h]，feature-map(网格)局部坐标系。
        中心偏移用 SmoothL1；宽高用对数空间 SmoothL1（对尺度更稳健，且天然保证正值梯度合理）。
        calib 框现在是 (cx,cy),(w,h),angle 的统一表示，不再需要 AABB/GIoU。
        """
        center_loss = F.smooth_l1_loss(pred[:, :2], target[:, :2], reduction='mean')
        size_loss = F.smooth_l1_loss(
            torch.log(pred[:, 2:4].clamp(min=eps)),
            torch.log(target[:, 2:4].clamp(min=eps)),
            reduction='mean',
        )
        return center_loss + size_loss

    def _centerness_loss(self, pred, target):
        """
        pred: (N,) 或 (N,1)
        target: (N,)
        """

        pred = pred.squeeze(-1)

        loss = F.binary_cross_entropy_with_logits(
            pred,
            target,
            reduction='mean'
        )

        return loss

    def _det_loss_from_targets(self, pred_cls_list, pred_reg_list, pred_ctr_list,
                               cls_tgt_list, bbox_tgt_list, ctr_tgt_list, angle_tgt_list,
                               tag: str) -> torch.Tensor:
        device = self.device
        cls_loss_total = torch.tensor(0., device=device)
        bbox_loss_total = torch.tensor(0., device=device)
        ctr_loss_total = torch.tensor(0., device=device)
        angle_loss_total = torch.tensor(0., device=device)
        total_pos = 0
        total_loc = 0

        for i in range(len(cls_tgt_list)):
            cls_target = cls_tgt_list[i].to(device)
            bbox_target = bbox_tgt_list[i].to(device)
            center_target = ctr_tgt_list[i].to(device)
            angle_target = angle_tgt_list[i].to(device)

            pred_cls = pred_cls_list[i]
            pred_bbox = pred_reg_list[i][:, :4].permute(0, 2, 3, 1)
            pred_angle = pred_reg_list[i][:, 4:5].permute(0, 2, 3, 1).squeeze(-1)
            pred_center = pred_ctr_list[i].permute(0, 2, 3, 1)

            pos_mask = center_target > 0
            num_pos = pos_mask.sum().item()
            total_pos += num_pos
            total_loc += cls_target.shape[0] * cls_target.shape[2] * cls_target.shape[3]  # B*H*W

            # 分类loss无论有没有正样本都要保留，只是归一化分母用 max(total_pos,1)（FCOS标准做法）
            layer_cls_loss = self._focal_loss(pred_cls, cls_target, reduction='sum')
            cls_loss_total += layer_cls_loss

            if num_pos > 0:
                bbox_loss_total += self._bbox_loss(pred_bbox[pos_mask], bbox_target[pos_mask]) * num_pos
                ctr_loss_total += self._centerness_loss(pred_center[pos_mask], center_target[pos_mask]) * num_pos
                angle_loss_total += self._angle_loss(pred_angle[pos_mask], angle_target[pos_mask]) * num_pos

        norm = max(total_pos, 1)
        return (cls_loss_total + bbox_loss_total + ctr_loss_total + angle_loss_total) / norm

    @staticmethod
    def _angle_loss(pred_dth: torch.Tensor, gt_angle_rad: torch.Tensor) -> torch.Tensor:

        pred_angle_rad = pred_dth * (math.pi / 4.0)
        return (1.0 - torch.cos(pred_angle_rad - gt_angle_rad)).mean()

    def _freeze_backbone_layers(self, num_frozen_layers: int = 2):

        backbone = self.model.backbone
        if not getattr(backbone, '_use_timm', False) or backbone.hrnet is None:
            logger.warning("骨干网络未使用 timm 真实权重（stub fallback），跳过冻结")
            return

        children = list(backbone.hrnet.named_children())
        frozen_names = []
        for name, module in children[:num_frozen_layers]:
            for p in module.parameters():
                p.requires_grad = False
            frozen_names.append(name)
        logger.info(f"已冻结骨干网络子模块: {frozen_names}")

    def _resume_state(self, phase: str, optimizer, scheduler, total_epochs: int) -> int:
        resume_path = self.cfg.get('resume')
        if not resume_path:
            return 0
        resume_path = Path(resume_path)
        if not resume_path.exists():
            logger.warning(f"[Resume] 指定的 resume 路径不存在，忽略并从头训练：{resume_path}")
            return 0

        ckpt = torch.load(resume_path, map_location=self.device)
        ckpt_phase = ckpt.get('phase')
        if ckpt_phase != phase:
            logger.info(f"[Resume] checkpoint 属于 phase='{ckpt_phase}'，与当前 phase='{phase}' 不符，跳过恢复")
            return 0

        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'loss_fn_state_dict' in ckpt:
            self.loss_fn.load_state_dict(ckpt['loss_fn_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.best_val_metric = ckpt.get('best_val_metric', self.best_val_metric)
        if ckpt.get('best_ckpt_path'):
            self.best_dir = Path(ckpt['best_ckpt_path'])
        if 'history' in ckpt:
            self.history = defaultdict(list, ckpt['history'])

        start_epoch = int(ckpt.get('epoch', -1)) + 1
        if start_epoch >= total_epochs:
            logger.warning(
                f"[Resume] checkpoint 记录的 epoch={ckpt.get('epoch')} 已达到/超过 "
                f"total_epochs={total_epochs}，本阶段将直接跳过训练循环。"
            )
        else:
            logger.info(
                f"[Resume] 已从 {resume_path} 恢复 {phase} 训练状态，"
                f"将从 epoch {start_epoch}/{total_epochs} 继续（best_val_metric={self.best_val_metric:.4f}）"
            )
        return start_epoch

    def _save_last_checkpoint(self, epoch: int, phase: str, optimizer, scheduler, val_metrics: Dict):

        ckpt_path = self.out_dir / f'last_{phase}.pth'
        torch.save({
            'phase': phase,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'best_ckpt_path': str(self.best_dir) if self.best_dir else None,
            'val_metrics': val_metrics,
            'history': dict(self.history),
        }, ckpt_path)
        logger.info(f"[Resume] 已更新 last 检查点（供中断续训使用）: {ckpt_path}")

    def _save_checkpoint(self, epoch: int, val_metrics: Dict, phase: str):
        """保存最优检查点"""
        current_metric = (
                val_metrics['kp_pck_2px'] * 0.5 +
                val_metrics['probe_seg_f1'] * 0.3 +
                val_metrics['calib_det_f1'] * 0.2
        )
        if current_metric > self.best_val_metric:
            self.best_val_metric = current_metric
            ckpt_path = self.out_dir / f'best_{phase}.pth'
            torch.save({
                'phase': phase,
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'loss_fn_state_dict': self.loss_fn.state_dict(),
                'val_metrics': val_metrics,
            }, ckpt_path)
            self.best_dir = ckpt_path
            logger.info(f"保存最优检查点: {ckpt_path}  PCK={current_metric:.4f}")

    def _log_epoch(self, epoch, train_m, val_m, phase):
        record = {'epoch': epoch, 'phase': phase, 'train': train_m, 'val': val_m}
        self.history[phase].append(record)
        with open(self.out_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


# ══════════════════════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════════════════════
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GSGProbeNet 训练入口")
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--phase', type=str, choices=['phase1', 'phase2', 'both'], default='both',
                        help="只跑 phase1、只跑 phase2，还是完整跑两阶段（默认 both）")
    parser.add_argument('--resume', type=str, default=None,
                        help="从指定的 checkpoint（如 <exp_dir>/last_phase1.pth 或 last_phase2.pth）"
                             "恢复训练；checkpoint 里记录的 phase 决定实际恢复的是哪个阶段，"
                             "不匹配的 phase 会被跳过、按未 resume 的原逻辑正常执行")
    parser.add_argument('--hrnet_root', type=str, default=None,
                        help="HRNet 预训练权重(.safetensors)路径；默认按项目树结构自动推导为 "
                             "<项目根>/data/model/model.safetensors，一般不需要手动指定")
    parser.add_argument('--config', type=str, default=None,
                        help="可选：从 json 文件加载配置，覆盖 DEFAULT_CONFIG 中的对应字段")
    return parser


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg.update(json.load(f))
    if args.data_root:
        cfg['data_root'] = args.data_root
    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    if args.resume:
        cfg['resume'] = args.resume
    if args.hrnet_root:
        cfg['hrnet_root'] = args.hrnet_root

    if not cfg.get('data_root') or not cfg.get('output_dir'):
        raise ValueError("必须提供 data_root 和 output_dir（通过 --data_root/--output_dir 或 --config）")

    if cfg.get('pretrained') and cfg.get('hrnet_root') and not Path(cfg['hrnet_root']).exists():
        logger.warning(
            f"[路径检查] hrnet_root 指向的文件不存在: {cfg['hrnet_root']}\n"
            f"  若本机项目目录结构与项目树不符（例如没有把 data/model/model.safetensors 放在 "
            f"<项目根>/data/model/ 下），请通过 --hrnet_root 显式指定实际路径。"
        )

    trainer = Trainer(cfg)

    if args.phase in ('phase1', 'both'):
        trainer.phase1_pretrain()
    if args.phase in ('phase2', 'both'):
        trainer.phase2_finetune()
