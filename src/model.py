from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from definitions import *

from data_pipeline import *
import torchvision.ops as tv_ops  # RoI Align
import cv2
from safetensors.torch import load_file
import timm


# 作用： 在构建batch时，将监督信号，转换为可以输入到模型中的tensor，封装在dict中
def collate_fn(batch):
    # ── 图像 ──────────────────────────────────────────────────────
    img_paths = [item["img_path"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    images = torch.stack([item["image"] for item in batch])  # [B,C,H,W]

    # ── 探针分支（统一旋转框格式：[cx,cy,w,h,theta]）───────────────────
    probe_labels = [[int(v) for v in item["probe_labels"]] for item in batch]
    probe_rboxes = [item["probe_rboxes"] for item in batch]  # B*List[N_i,5]
    probe_masks = [item["probe_masks"] for item in batch]  # B*List[N_i ndarray(H,W)]
    probe_keypoints = [item["probe_keypoints"] for item in batch]  # B*List[N_i ndarray(16,2)]
    probe_visibility = [item["probe_visibility"] for item in batch]  # B*List[N_i ndarray(16,)]

    # ── 探针状态分支（新增：遮挡率/失焦度/几何可见比例，逐实例与 probe_rboxes 对齐）──
    probe_occlusion_ratio = [item["probe_occlusion_ratio"] for item in batch]  # B*List[N_i] float
    probe_defocus_level = [item["probe_defocus_level"] for item in batch]  # B*List[N_i] float
    probe_visible_ratio = [item["probe_visible_ratio"] for item in batch]  # B*List[N_i] float

    # ── 校准片分支（同一格式：[cx,cy,w,h,theta]）─────────────────────
    calib_labels = [[int(v) for v in item["calib_labels"]] for item in batch]  # B*List[M_i] int
    calib_rboxes = [item["calib_rboxes"] for item in batch]  # B*List[M_i,5]

    # ── Calib Detection Targets
    calib_cls_target_list, calib_bbox_target_list = [], []
    calib_centerness_target_list, calib_angle_target_list = [], []

    # ── Probe Detection Targets ──────────────────────────────────
    probe_cls_target_list, probe_bbox_target_list = [], []
    probe_centerness_target_list, probe_angle_target_list = [], []

    FCOS_SIZE_RANGES = [(0, 64), (64, 128), (128, 256), (256, 9999)]
    for idx, (size_lo, size_hi) in enumerate(FCOS_SIZE_RANGES):
        stride = 2 ** (idx + 2)
        cls_t, bbox_t, cness_t, angle_t = build_detection_targets(
            images, calib_labels, calib_rboxes, stride,
            num_classes=NUM_CALIB_CLASSES, size_range=(size_lo, size_hi), center_sampling_radius=1.5,
        )
        calib_cls_target_list.append(cls_t)
        calib_bbox_target_list.append(bbox_t)
        calib_centerness_target_list.append(cness_t)
        calib_angle_target_list.append(angle_t)

        p_cls_t, p_bbox_t, p_cness_t, p_angle_t = build_detection_targets(
            images, probe_labels, probe_rboxes, stride,
            num_classes=NUM_PROBE_CLASSES, size_range=(size_lo, size_hi), center_sampling_radius=1.5,
        )
        probe_cls_target_list.append(p_cls_t)
        probe_bbox_target_list.append(p_bbox_t)
        probe_centerness_target_list.append(p_cness_t)
        probe_angle_target_list.append(p_angle_t)

    # ══════════════════════════════════════════════════════════
    # task_stats：每个任务在本 batch 内的「真实有效样本量」
    # ══════════════════════════════════════════════════════════
    n_probe_inst = sum(len(kps) for kps in probe_keypoints)
    n_probe_img = sum(1 for m in probe_masks if len(m) > 0)
    n_calib_pos = sum(len(lbl) for lbl in calib_labels)

    task_stats = {
        'probe_det': n_probe_img,
        'keypoint': n_probe_inst,
        'calib_det': n_calib_pos,
        'probe_state': n_probe_inst,  # 与 keypoint 复用同一批探针实例（同一份 RoI）
    }

    # ── 汇总输出 ───────────────────────────────────────────────────
    return {
        # 图像
        "image": images,
        "img_paths": img_paths,
        "image_ids": image_ids,

        # 探针原始标注
        "probe_labels": probe_labels,
        "probe_rboxes": probe_rboxes,
        "probe_masks": probe_masks,
        "probe_keypoints": probe_keypoints,
        "probe_visibility": probe_visibility,
        "probe_occlusion_ratio": probe_occlusion_ratio,
        "probe_defocus_level": probe_defocus_level,
        "probe_visible_ratio": probe_visible_ratio,

        # 校准片原始标注
        "calib_labels": calib_labels,
        "calib_rboxes": calib_rboxes,

        # Probe Detection targets
        "probe_cls_target_list": probe_cls_target_list,
        "probe_bbox_target_list": probe_bbox_target_list,
        "probe_centerness_target_list": probe_centerness_target_list,
        "probe_angle_target_list": probe_angle_target_list,

        # Calib Detection targets
        "calib_cls_target_list": calib_cls_target_list,
        "calib_bbox_target_list": calib_bbox_target_list,
        "calib_centerness_target_list": calib_centerness_target_list,
        "calib_angle_target_list": calib_angle_target_list,

        "task_stats": task_stats,
    }


# ══════════════════════════════════════════════════════════════
#  骨干网络：HRNet-W32
# ══════════════════════════════════════════════════════════════

class HRNetBackbone(nn.Module):

    def __init__(self, config: Dict, pretrained: bool = True):
        super().__init__()
        self.config = config
        try:
            if self.config['hrnet_root'] is not None:
                # 本地加载
                self.hrnet = timm.create_model(
                    'hrnet_w32', pretrained=False, features_only=True, out_indices=(1, 2, 3, 4),
                )
                state_dict = load_file(self.config['hrnet_root'])
                self.hrnet.load_state_dict(state_dict, strict=False)
            else:
                # 联网下载
                self.hrnet = timm.create_model(
                    'hrnet_w32', pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4)
                )

            self._use_timm = True
        except ImportError:
            self.hrnet = None
            self._use_timm = False

        self.out_channels = [32, 64, 128, 256]
        self.proj = nn.ModuleList([
            nn.Conv2d(c_in, c_out, kernel_size=1)
            for c_in, c_out in zip([128, 256, 512, 1024], self.out_channels)
        ])

    # 前馈计算
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        if self._use_timm:
            features = self.hrnet(x)  # 原始 HRNet 输出
            out = []
            for f, conv in zip(features, self.proj):
                out.append(conv(f))  # 强制变成 32/64/128/256
            return out

        # --- stub fallback（不含真实权重，仅保持尺寸正确）---
        B, C, H, W = x.shape
        return [
            torch.zeros(B, ch, H // s, W // s, device=x.device)
            for ch, s in zip(self.out_channels, [4, 8, 16, 32])
        ]


# ══════════════════════════════════════════════════════════════
#  FPN：多尺度特征融合
# ══════════════════════════════════════════════════════════════

class FPN(nn.Module):

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        # 4个 1*1 卷积层
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        laterals = [l(f) for l, f in zip(self.laterals, features)]
        # 自顶向下融合：从最高层（P5, index 3）向下逐层融合到 P2（index 0）
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest',
            )
        outs = [o(l) for o, l in zip(self.outputs, laterals)]
        return {'p2': outs[0], 'p3': outs[1], 'p4': outs[2], 'p5': outs[3]}


# ══════════════════════════════════════════════════════════════
#  任务头 ②：探针针尖关键点检测头
# ══════════════════════════════════════════════════════════════
class RoIKeypointHead(nn.Module):
    def __init__(self, in_channels=256, num_keypoints=16, roi_size=(32, 32),
                 hidden_dim=256, softmax_temperature=100.0):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.roi_size = roi_size
        self.temperature = softmax_temperature

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2), nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(hidden_dim // 2, num_keypoints, 1)

        self.sigma_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (N, hidden_dim//2, 1, 1)
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, num_keypoints, 1),  # (N, K, 1, 1)
        )
        nn.init.zeros_(self.sigma_head[-1].bias)
        nn.init.normal_(self.sigma_head[-1].weight, std=0.001)

        # ── 新增：逐关键点可见性分类（0=缺失/出界, 1=遮挡, 2=完全可见）──
        # 复用与 sigma_head 相同的 GAP 特征，训练标签直接来自数据集里已有的
        # visibility 字段（无需新标注），让模型显式学会"这个针尖点是被挡住了
        # 还是压根不在视野里，还是清晰可见"，而不只是隐式体现在热图的模糊程度上。
        self.vis_cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, num_keypoints * 3, 1),  # (N, K*3, 1, 1)
        )

    def extract_roi_feats(self, p2, rboxes_list, p2_stride: float = 4.0):
        return extract_rotated_roi_feats(p2, rboxes_list, self.roi_size, feat_stride=p2_stride)

    def forward_from_feats(self, roi_feats: torch.Tensor, batch_idx: torch.Tensor):
        if roi_feats.shape[0] == 0:
            K = self.num_keypoints
            h, w = self.roi_size
            empty_hm = roi_feats.new_zeros(0, K, h, w)
            empty_sig = roi_feats.new_zeros(0, K)
            empty_vis = roi_feats.new_zeros(0, K, 3)
            return empty_hm, empty_sig, empty_vis, batch_idx

        feat = self.conv_layers(roi_feats)
        heatmap = self.heatmap_head(feat)
        log_sigma2 = self.sigma_head(feat).squeeze(-1).squeeze(-1)  # (N, K)
        vis_logits = self.vis_cls_head(feat).squeeze(-1).squeeze(-1)  # (N, K*3)
        vis_logits = vis_logits.view(-1, self.num_keypoints, 3)  # (N, K, 3)
        return heatmap, log_sigma2, vis_logits, batch_idx

    def forward(self, p2, rboxes_list):
        roi_feats, batch_idx = self.extract_roi_feats(p2, rboxes_list)
        return self.forward_from_feats(roi_feats, batch_idx)

    # 可见性分类的类别语义：与数据集里的 visibility 字段（0/1/2）保持一致
    VIS_CLASS_NAMES = {0: 'missing', 1: 'occluded', 2: 'visible'}

    def decode(self, heatmap, log_sigma2, rboxes_list, batch_idx, p2_stride: int = 4,
               vis_logits: Optional[torch.Tensor] = None):
        if heatmap.shape[0] == 0:
            return [[[] for _ in range(len(boxes))] for boxes in rboxes_list]

        N, K, roi_h, roi_w = heatmap.shape
        device = heatmap.device

        coords_roi = soft_argmax_2d(heatmap, self.temperature)  # (N,K,2) -> (x,y)
        cx_roi, cy_roi = coords_roi[..., 0], coords_roi[..., 1]

        prob = torch.sigmoid(heatmap)
        confidence = prob.reshape(N, K, -1).max(-1).values

        log_sigma2_c = log_sigma2.clamp(-2.0, 6.0)  # 见下方"clamp 取值依据"
        sigma = torch.exp(0.5 * log_sigma2_c)  # (N, K)，单位：ROI 网格像素

        all_rbox_tensor = [torch.tensor(x, device=device) if isinstance(x, list) else x.to(device)
                           for x in rboxes_list]
        all_rboxes = torch.cat(all_rbox_tensor, dim=0)
        cx, cy, w, h, theta = all_rboxes[:, 0:1], all_rboxes[:, 1:2], all_rboxes[:, 2:3], all_rboxes[:,
                                                                                          3:4], all_rboxes[:, 4:5]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        x_local = (cx_roi / roi_w - 0.5) * w
        y_local = (cy_roi / roi_h - 0.5) * h
        x_img = cx + x_local * cos_t - y_local * sin_t
        y_img = cy + x_local * sin_t + y_local * cos_t

        cx_np, cy_np = x_img.detach().cpu().numpy(), y_img.detach().cpu().numpy()
        conf_np, sigma_np = confidence.detach().cpu().numpy(), sigma.detach().cpu().numpy()
        bidx_np = batch_idx.cpu().numpy()

        # 可见性分类：argmax 得类别，softmax 概率一并给出，供下游按置信度过滤
        if vis_logits is not None and vis_logits.shape[0] == N:
            vis_prob = torch.softmax(vis_logits, dim=-1)  # (N, K, 3)
            vis_cls = vis_prob.argmax(dim=-1)  # (N, K)
            vis_cls_np = vis_cls.detach().cpu().numpy()
            vis_prob_np = vis_prob.detach().cpu().numpy()
        else:
            vis_cls_np = None
            vis_prob_np = None

        B = len(rboxes_list)
        results = [[] for _ in range(B)]
        for n in range(N):
            b = int(bidx_np[n])
            kp_list = []
            for k in range(K):
                kp = {'keypoint_id': k, 'x': float(cx_np[n, k]), 'y': float(cy_np[n, k]),
                      'confidence': float(conf_np[n, k]), 'sigma': float(sigma_np[n, k])}
                if vis_cls_np is not None:
                    cls_id = int(vis_cls_np[n, k])
                    kp['visibility_pred'] = cls_id  # 0=缺失/出界, 1=遮挡, 2=可见
                    kp['visibility_pred_name'] = self.VIS_CLASS_NAMES[cls_id]
                    kp['visibility_prob'] = [float(v) for v in vis_prob_np[n, k]]
                kp_list.append(kp)
            results[b].append(kp_list)
        return results


# ══════════════════════════════════════════════════════════════
#  通用旋转框检测头（多尺度FCOS）
# ══════════════════════════════════════════════════════════════
#  [结构性重构] 原来 probe 分支用"语义分割+offset投票聚类"做实例分离，
#  calib 分支用 FCOS 旋转框检测——两条完全不同的技术路线共存在同一个
#  网络里。验证结果显示 calib 分支 F1=0.973，probe 分支 precision 只有
#  0.177（因为细长/旋转目标的 offset 投票极易把同一实例的像素分裂到
#  多个候选中心，产生大量假阳性实例）。
#
#  根治方案：把 probe 分支也换成同一套已被验证有效的 FCOS 旋转框检测器，
#  两个任务共用同一个类（RotatedFCOSHead），只是各自持有一份独立权重、
#  独立的类别数量和类别映射。检测出的旋转框再喂给下游 RoI 关键点头 /
#  RoI mask 头，不再依赖脆弱的"分割+聚类"后处理。
# ══════════════════════════════════════════════════════════════
class RotatedFCOSHead(nn.Module):

    def __init__(self, in_channels: int = 256, num_classes: int = 4,
                 label_to_category_id: Optional[Dict[int, int]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.label_to_category_id = label_to_category_id or {i: i for i in range(num_classes)}
        self.shared_tower = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1), nn.GroupNorm(32, 128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(32, 128), nn.ReLU(True),
        )
        # 目标检测
        self.cls_head = nn.Conv2d(128, num_classes, 1)
        self.bbox_head = nn.Conv2d(128, 4, 1)
        self.angle_head = nn.Conv2d(128, 1, 1)
        self.centerness_head = nn.Conv2d(128, 1, 1)

        # FPN 各层的 scale 参数（可学习，稳定回归）
        self.scale_p2 = nn.Parameter(torch.ones(1))
        self.scale_p3 = nn.Parameter(torch.ones(1))
        self.scale_p4 = nn.Parameter(torch.ones(1))
        self.scale_p5 = nn.Parameter(torch.ones(1))

        prior_prob = 0.01
        bias_init = -math.log((1 - prior_prob) / prior_prob)  # ≈ -4.6
        nn.init.constant_(self.cls_head.bias, bias_init)

    def _forward_single(
            self, feat: torch.Tensor, scale: nn.Parameter
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tower_feat = self.shared_tower(feat)
        cls_logits = self.cls_head(tower_feat)

        raw_reg = self.bbox_head(tower_feat)  # (B,4,H,W): dx, dy, w, h（未激活）
        safe_scale = F.softplus(scale)
        center_off = raw_reg[:, :2] * safe_scale  # dx, dy：允许正负
        size_wh = F.softplus(raw_reg[:, 2:4]) * safe_scale  # w, h：恒 > 0
        bbox_reg = torch.cat([center_off, size_wh], dim=1)  # (B,4,H,W)

        angle_reg = torch.tanh(self.angle_head(tower_feat))  # (B,1,H,W)，∈(-1,1)
        reg = torch.cat([bbox_reg, angle_reg], dim=1)  # (B,5,H,W) = dx,dy,w,h,dθ
        centerness = self.centerness_head(tower_feat)
        return cls_logits, reg, centerness

    def forward(
            self, fpn_feats: Dict[str, torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:

        keys = ['p2', 'p3', 'p4', 'p5']
        scales = [self.scale_p2, self.scale_p3, self.scale_p4, self.scale_p5]

        cls_list, reg_list, ctr_list = [], [], []

        for key, scale in zip(keys, scales):
            c, r, ct = self._forward_single(fpn_feats[key], scale)
            cls_list.append(c)
            reg_list.append(r)
            ctr_list.append(ct)

        return {
            'cls': cls_list,
            'reg': reg_list,
            'ctr': ctr_list
        }

    def decode(
            self,
            cls_list: List[torch.Tensor],  # 每项 (B, C, H, W)
            reg_list: List[torch.Tensor],  # 每项 (B, 5, H, W)
            ctr_list: List[torch.Tensor],  # 每项 (B, 1, H, W)
            strides: Tuple[int, ...] = (4, 8, 16, 32),
            score_thresh: float = 0.1,
            nms_iou_thresh: float = 0.2,
            pre_nms_top_k: int = 200,  # NMS 前最多保留的候选数，防止大量低质量框涌入
    ) -> List[List[Dict]]:

        B = cls_list[0].shape[0]
        device = cls_list[0].device

        cand_scores: List[List[torch.Tensor]] = [[] for _ in range(B)]
        cand_cls_ids: List[List[torch.Tensor]] = [[] for _ in range(B)]
        cand_rects: List[List[torch.Tensor]] = [[] for _ in range(B)]

        # 遍历FPN每一层，逐层解码、筛选候选框
        for cls, reg, ctr, stride in zip(cls_list, reg_list, ctr_list, strides):
            _, C, H, W = cls.shape

            # ====================== 1. 计算最终置信度 ======================
            # 分类sigmoid + 中心度sigmoid，开根号融合，抑制偏离网格中心的低质量框
            scores_map = (cls.sigmoid() * ctr.sigmoid()).sqrt()  # shape: (B, C, H, W)
            # 每个网格取最大类别得分、对应类别ID
            max_scores, cls_ids = scores_map.max(dim=1)  # shape: (B, H, W)

            # ====================== 2. 生成当前层所有网格中心点（映射回原图坐标） ======================
            gy, gx = torch.meshgrid(
                torch.arange(H, dtype=torch.float32, device=device),
                torch.arange(W, dtype=torch.float32, device=device),
                indexing='ij',
            )
            # 网格左上角+0.5得到网格中心，乘以步长映射原图
            anchor_cx = (gx + 0.5) * stride  # (H, W)
            anchor_cy = (gy + 0.5) * stride
            # 扩展batch维度，匹配批次数据
            anchor_cx = anchor_cx.unsqueeze(0).expand(B, -1, -1)
            anchor_cy = anchor_cy.unsqueeze(0).expand(B, -1, -1)

            # ====================== 3. 解码回归偏移，得到旋转框参数 ======================
            # 拆分5通道回归值：中心偏移dx、dy，紧贴框宽高w、h，角度偏移dθ
            dx, dy, w_, h_, dth = reg.unbind(dim=1)
            pred_cx = anchor_cx + dx * stride
            pred_cy = anchor_cy + dy * stride
            pred_w = w_ * stride
            pred_h = h_ * stride
            pred_angle = dth * (math.pi / 4)

            # ====================== 4. 置信度阈值过滤 ======================
            keep_mask = max_scores >= score_thresh  # 有效候选掩码 (B, H, W) bool
            # 遍历批次内每张图片，收集本层有效候选
            for b in range(B):
                mask = keep_mask[b]  # 当前图片掩码 (H, W)
                # 当前层无满足阈值的候选，直接跳过
                if not mask.any():
                    continue
                # 根据掩码提取筛选后的得分、类别、框参数
                s_b = max_scores[b][mask]  # (N,) N为本层有效候选数量
                cls_b = cls_ids[b][mask]  # (N,)
                rect_b = torch.stack([  # 拼接旋转框 (cx, cy, w, h, angle) -> shape (N, 5)
                    pred_cx[b][mask],
                    pred_cy[b][mask],
                    pred_w[b][mask],
                    pred_h[b][mask],
                    pred_angle[b][mask],
                ], dim=-1)
                # 候选过多时，截取置信度前pre_nms_top_k个，减少后续NMS计算量
                if s_b.shape[0] > pre_nms_top_k:
                    topk_idx = s_b.topk(pre_nms_top_k).indices
                    s_b = s_b[topk_idx]
                    cls_b = cls_b[topk_idx]
                    rect_b = rect_b[topk_idx]
                # 将本层筛选后的候选存入对应图片的缓存列表
                cand_scores[b].append(s_b)
                cand_cls_ids[b].append(cls_b)
                cand_rects[b].append(rect_b)

        temp = len(cand_scores)
        # ====================== 5. 单张图聚合所有FPN层候选 + 按类别旋转NMS去重 ======================
        batch_results: List[List[Dict]] = []
        for b in range(B):
            # 当前图片全程无任何有效候选，存入空结果
            if not cand_scores[b]:
                batch_results.append([])
                continue
            # 拼接该图片所有FPN层的候选
            scores = torch.cat(cand_scores[b])  # (Total_N,)
            cls_ids = torch.cat(cand_cls_ids[b])  # (Total_N,)
            rects = torch.cat(cand_rects[b])  # (Total_N, 5)

            # 执行**按类别旋转NMS**：同类重叠框抑制，不同类别互不干扰
            keep = self._rotated_nms_per_class(rects, scores, cls_ids, nms_iou_thresh)
            # 保留NMS筛选后的结果
            scores = scores[keep]
            cls_ids = cls_ids[keep]
            rects = rects[keep]

            # 张量转移CPU、转numpy，方便后续序列化存储/评测匹配
            rects_np = rects.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            cls_np = cls_ids.detach().cpu().numpy()

            # 封装成结构化字典，适配后处理、指标匹配、结果保存逻辑
            sample: List[Dict] = [
                {
                    'class_id': self.label_to_category_id[int(cls_np[i])],  # 原始类别ID
                    'label': int(cls_np[i]),  # 映射业务标签枚举
                    'score': float(scores_np[i]),  # 最终置信度
                    'cx': float(rects_np[i, 0]),  # 旋转框中心x
                    'cy': float(rects_np[i, 1]),  # 旋转框中心y
                    'w': float(rects_np[i, 2]),  # 外接宽度
                    'h': float(rects_np[i, 3]),  # 外接高度
                    'angle': float(rects_np[i, 4]),  # 旋转角(弧度)
                    'rect': [  # 统一旋转框格式 [cx, cy, w, h, theta_rad]，用于指标IoU匹配
                        float(rects_np[i, 0]), float(rects_np[i, 1]),
                        float(rects_np[i, 2]), float(rects_np[i, 3]),
                        float(rects_np[i, 4]),
                    ],
                    'rotated_rect': (  # 完整旋转框结构体，适配旋转IOU计算
                        (float(rects_np[i, 0]), float(rects_np[i, 1])),
                        (float(rects_np[i, 2]), float(rects_np[i, 3])),
                        float(rects_np[i, 4]),
                    ),
                }
                for i in range(len(keep))
            ]
            batch_results.append(sample)

        return batch_results

    # ──────────────────────────────────────────────────────────────────────────────
    # NMS 工具函数
    # ──────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rotated_iou_matrix(rects: torch.Tensor) -> np.ndarray:
        """rects: (M,5) cx,cy,w,h,theta(弧度) → (M,M) IoU 矩阵，CPU/numpy 计算"""
        import cv2
        r = rects.detach().cpu().numpy()
        M = r.shape[0]
        iou = np.zeros((M, M), dtype=np.float32)
        cv_rects = [((float(x[0]), float(x[1])), (float(x[2]), float(x[3])),
                     math.degrees(float(x[4]))) for x in r]
        areas = r[:, 2] * r[:, 3]
        for i in range(M):
            for j in range(i + 1, M):
                _, pts = cv2.rotatedRectangleIntersection(cv_rects[i], cv_rects[j])
                inter = 0.0
                if pts is not None and len(pts) >= 3:
                    inter = float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))
                union = areas[i] + areas[j] - inter
                v = inter / union if union > 0 else 0.0
                iou[i, j] = iou[j, i] = v
        return iou

    def _rotated_nms_per_class(self, rects, scores, cls_ids, iou_thr=0.5):
        keep_all = []
        for cls in cls_ids.unique():
            idx = (cls_ids == cls).nonzero(as_tuple=True)[0]
            sub_rects = rects[idx]
            sub_scores = scores[idx]
            order = torch.argsort(sub_scores, descending=True).cpu().numpy()
            iou_mat = self._rotated_iou_matrix(sub_rects)  # (M,M) numpy
            suppressed = np.zeros(len(order), dtype=bool)
            kept_local = []
            for oi in order:
                if suppressed[oi]:
                    continue
                kept_local.append(oi)
                suppressed |= (iou_mat[oi] > iou_thr)
                suppressed[oi] = True
            keep_all.append(idx[torch.as_tensor(kept_local, device=rects.device, dtype=torch.long)])
        return torch.cat(keep_all) if keep_all else torch.zeros(0, dtype=torch.long, device=rects.device)


CalibDetectionHead = RotatedFCOSHead


# ══════════════════════════════════════════════════════════════
#  共享 RoI 特征提取（探针检测框 → 旋转对齐特征图）
# ══════════════════════════════════════════════════════════════
#  [结构性重构] 原来 RoIKeypointHead.extract_roi_feats 只服务关键点头。
#  现在 mask head 需要完全一样的旋转 RoI 对齐特征，抽成独立函数后两个头
#  共用同一次 grid_sample，避免重复计算，也保证两个头看到的是同一份
#  "探针局部坐标系"，语义上更一致。
# ══════════════════════════════════════════════════════════════
def extract_rotated_roi_feats(feat: torch.Tensor, rboxes_list, roi_size: Tuple[int, int],
                              feat_stride: float = 4.0):
    C = feat.shape[1]
    Hf, Wf = feat.shape[-2:]
    roi_h, roi_w = roi_size
    all_rbox, batch_idx_list = [], []
    for b, rboxes in enumerate(rboxes_list):
        if len(rboxes) == 0:
            continue
        batch_idx_list.append(torch.full((len(rboxes),), b, dtype=torch.long, device=feat.device))
        all_rbox.append(rboxes)

    if not all_rbox:
        return feat.new_zeros(0, C, roi_h, roi_w), feat.new_zeros(0, dtype=torch.long)
    batch_idx = torch.cat(batch_idx_list, dim=0)
    device = batch_idx.device
    all_rbox_tensor = [
        torch.tensor(x, device=device) if isinstance(x, list) else x.to(device)
        for x in all_rbox
    ]
    rboxes = torch.cat(all_rbox_tensor, dim=0)

    theta_mat = build_rotated_affine_theta(
        rboxes[:, 0], rboxes[:, 1], rboxes[:, 2], rboxes[:, 3], rboxes[:, 4],
        feat_stride, Hf, Wf,
    )
    grid = F.affine_grid(theta_mat, size=(rboxes.shape[0], C, roi_h, roi_w), align_corners=True)
    gathered_feat = feat[batch_idx]
    roi_feats = F.grid_sample(gathered_feat, grid, mode='bilinear',
                              padding_mode='zeros', align_corners=True)
    return roi_feats, batch_idx


def paste_roi_mask_to_image(mask_prob_roi: torch.Tensor, cx: float, cy: float, w: float, h: float,
                            theta: float, img_h: int, img_w: int) -> torch.Tensor:
    """
    把 RoI 局部坐标系下的 mask 概率图，仿射逆变换贴回原图坐标系。
    mask_prob_roi: (roi_h, roi_w)，值域 [0,1]
    返回: (img_h, img_w) 的概率图（原图分辨率），供阈值化 / mask IoU 使用。

    build_rotated_affine_theta 给出的是"RoI输出坐标 → 原图归一化坐标"的仿射矩阵，
    这里通过求逆得到"原图归一化坐标 → RoI归一化坐标"，再对 RoI mask 做一次
    grid_sample，即可把 mask 无损地放回原图对应位置——不再需要 cv2 连通域/
    投票聚类等启发式操作。
    """
    device = mask_prob_roi.device
    theta_fwd = build_rotated_affine_theta(
        torch.tensor([cx], device=device), torch.tensor([cy], device=device),
        torch.tensor([w], device=device), torch.tensor([h], device=device),
        torch.tensor([theta], device=device),
        stride=1.0, feat_h=img_h, feat_w=img_w,
    )[0]  # (2,3)
    theta_homo = torch.eye(3, device=device, dtype=theta_fwd.dtype)
    theta_homo[:2, :] = theta_fwd
    theta_inv = torch.inverse(theta_homo)[:2, :]  # 原图归一化坐标 → RoI归一化坐标
    grid = F.affine_grid(theta_inv.unsqueeze(0), size=(1, 1, img_h, img_w), align_corners=True)
    sampled = F.grid_sample(mask_prob_roi[None, None], grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
    return sampled[0, 0]


# ══════════════════════════════════════════════════════════════
#  任务头 ：探针实例 mask 头（RoI-based，取代分割+offset投票聚类）
# ══════════════════════════════════════════════════════════════
class ProbeMaskHead(nn.Module):
    """
    输入已经是"每个探针实例"的旋转 RoI 特征（由检测头给出的框裁出），
    直接回归该实例的前景 mask——与 Mask R-CNN 的 mask 分支思路一致。
    不再需要对整图做语义分割 + 中心点 offset 投票聚类，因此不会再出现
    因 offset 误差把单个实例的像素投票分裂成多个候选中心、进而产生大量
    假阳性实例的问题。
    """

    def __init__(self, in_channels: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )
        self.mask_logit = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, roi_feats: torch.Tensor) -> torch.Tensor:
        """roi_feats: (N, C, roi_h, roi_w) → mask_logits: (N, 1, roi_h, roi_w)"""
        if roi_feats.shape[0] == 0:
            roi_h, roi_w = roi_feats.shape[-2:]
            return roi_feats.new_zeros(0, 1, roi_h, roi_w)
        feat = self.encoder(roi_feats)
        return self.mask_logit(feat)


def build_mask_targets_roi(rboxes_list, masks_list, roi_size: Tuple[int, int] = (28, 28),
                           device="cuda") -> Optional[torch.Tensor]:
    """
    把每个探针实例的原图分辨率 GT mask，仿射采样到与检测框对齐的 RoI 局部坐标系，
    与 build_keypoint_targets_roi 保持同样的"RoI 局部坐标系"约定，
    使 mask / keypoint 两个头在同一套几何定义下训练。
    返回: (N_total_inst, roi_h, roi_w)，值域 {0,1}；无实例时返回 None。
    """
    roi_h, roi_w = roi_size
    targets = []
    for b, boxes in enumerate(rboxes_list):
        n_i = len(boxes)
        if n_i == 0:
            continue
        masks_b = masks_list[b]
        for inst_idx in range(n_i):
            box = boxes[inst_idx]
            cx, cy, w, h, theta = [float(v) for v in box[:5]]
            mask_np = masks_b[inst_idx]
            mask_t = torch.as_tensor(np.asarray(mask_np), dtype=torch.float32, device=device)
            img_h, img_w = mask_t.shape[-2], mask_t.shape[-1]
            mask_t = mask_t[None, None]  # (1,1,H,W)

            theta_mat = build_rotated_affine_theta(
                torch.tensor([cx], device=device), torch.tensor([cy], device=device),
                torch.tensor([w], device=device), torch.tensor([h], device=device),
                torch.tensor([theta], device=device),
                stride=1.0, feat_h=img_h, feat_w=img_w,
            )
            grid = F.affine_grid(theta_mat, size=(1, 1, roi_h, roi_w), align_corners=True)
            roi_mask = F.grid_sample(mask_t, grid, mode='nearest',
                                     padding_mode='zeros', align_corners=True)
            targets.append((roi_mask.squeeze(0).squeeze(0) > 0.5).float())

    if not targets:
        return None
    return torch.stack(targets)  # (N, roi_h, roi_w)


# ══════════════════════════════════════════════════════════════
#  任务头 ：探针状态头（遮挡率 / 失焦度 / 几何可见比例）
# ══════════════════════════════════════════════════════════════
#  与 mask head 共用同一份旋转 RoI 特征（extract_rotated_roi_feats 在 p2 上
#  grid_sample 得到），只多一次轻量 conv+GAP，不引入新的 RoI 采样开销。
#  三个输出全部回归到 [0,1]（sigmoid），物理含义：
#    occlusion_ratio : 探针可见区域中被(合成)遮挡的像素占比
#    defocus_level   : 局部失焦程度，0=清晰，1=最大模拟模糊
#    visible_ratio   : 探针旋转框与画布的几何相交比例，1=完全在视野内，
#                       partial <1 = 部分结构在视野之外
#  三者的训练标签均由 data_pipeline.py 在增强阶段自动生成，见该文件顶部
#  "遮挡 / 失焦 / 出界 —— 自监督代理标签" 一节。
# ══════════════════════════════════════════════════════════════
class ProbeStateHead(nn.Module):
    STATE_NAMES = ('occlusion_ratio', 'defocus_level', 'visible_ratio')

    def __init__(self, in_channels: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 3, 1),  # [occlusion_ratio, defocus_level, visible_ratio]
        )
        # visible_ratio 的先验偏"完全可见"（大多数样本探针都在视野内），
        # 其余两项偏"未遮挡/未失焦"，bias 初始化贴近这个先验，加速收敛
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias[:] = torch.tensor([-2.0, -2.0, 2.0])  # sigmoid后 ≈[0.12,0.12,0.88]

    def forward(self, roi_feats: torch.Tensor) -> torch.Tensor:
        """roi_feats: (N, C, roi_h, roi_w) → (N, 3) logits（sigmoid 前）"""
        if roi_feats.shape[0] == 0:
            return roi_feats.new_zeros(0, 3)
        feat = self.encoder(roi_feats)
        return self.head(feat).squeeze(-1).squeeze(-1)  # (N, 3)

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)  # (N,3) ∈ [0,1]


# ══════════════════════════════════════════════════════════════
#  主网络
# ══════════════════════════════════════════════════════════════

class GSGProbeNet(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        self.backbone = HRNetBackbone(config=config, pretrained=config.get('pretrained', True))
        self.fpn = FPN(in_channels=[32, 64, 128, 256], out_channels=256)

        # Stage 1a: 探针检测
        self.probe_det = RotatedFCOSHead(
            in_channels=256, num_classes=NUM_PROBE_CLASSES,
            label_to_category_id=PROBE_LABEL_TO_CATEGORY_ID,
        )
        # Stage 1a': 探针实例 mask
        self.mask_head = ProbeMaskHead(in_channels=256, hidden_dim=128)
        self.mask_roi_size = (28, 28)

        # Stage 2: RoI-based 关键点检测
        self.kp_head = RoIKeypointHead(
            in_channels=256,
            num_keypoints=16,
            roi_size=(32, 32),
        )

        # Stage 2': 探针状态（遮挡率/失焦度/几何可见比例），复用 mask 分支的 RoI 特征
        self.state_head = ProbeStateHead(in_channels=256, hidden_dim=128)

        # 校准片检测
        self.calib_det = RotatedFCOSHead(
            in_channels=256, num_classes=NUM_CALIB_CLASSES,
            label_to_category_id=CALIB_LABEL_TO_CATEGORY_ID,
        )

        self.px_per_um = config['px_per_um']

    def forward(
            self,
            x: torch.Tensor,
            rboxes_list: Optional[List[torch.Tensor]] = None,
            probe_score_thresh: float = 0.3,
    ) -> Dict[str, object]:
        # ── 1. Backbone + FPN ──────────────────────────────────
        multi_scale_feats = self.backbone(x)
        fpn_feats = self.fpn(multi_scale_feats)  # dict: p2/p3/p4/p5
        p2 = fpn_feats['p2']  # 最高分辨率，供 RoI 使用

        # ── 2. Stage 1a: 探针检测 + 校准片检测（同构，两次前向） ──
        probe_det_outputs = self.probe_det(fpn_feats)
        calib_outputs = self.calib_det(fpn_feats)

        # ── 3. 获取 probe 实例 boxes（Stage 2 输入）─────────────
        # 训练：全部 RoI 任务（mask/keypoint）都用 GT 框，保证监督信号干净、
        #       且不再需要在训练期间跑一遍推理端 decode（见 train.py 的整改）。
        # 推理：先解码探针检测框，再据此裁 RoI 做 mask / keypoint。
        if rboxes_list is None:
            probe_det_results = self.probe_det.decode(
                probe_det_outputs['cls'], probe_det_outputs['reg'], probe_det_outputs['ctr'],
                score_thresh=probe_score_thresh,
            )
            rboxes_list_for_roi = [
                (torch.tensor([d['rect'] for d in dets], dtype=torch.float32, device=x.device)
                 if dets else torch.zeros((0, 5), dtype=torch.float32, device=x.device))
                for dets in probe_det_results
            ]
        else:
            probe_det_results = None  # 训练期间不需要，避免无谓解码开销
            rboxes_list_for_roi = rboxes_list

        # ── 4. 共享 RoI 特征：mask head / keypoint head / state head 三个头共用同一份
        #      旋转 RoI 采样结果，避免重复 grid_sample ──
        kp_roi_feats, kp_batch_idx = self.kp_head.extract_roi_feats(p2, rboxes_list_for_roi)
        kp_heatmap, kp_log_sigma2, kp_vis_logits, kp_batch_idx = self.kp_head.forward_from_feats(
            kp_roi_feats, kp_batch_idx,
        )

        mask_roi_feats, mask_batch_idx = extract_rotated_roi_feats(
            p2, rboxes_list_for_roi, self.mask_roi_size,
        )
        mask_logits = self.mask_head(mask_roi_feats)
        state_logits = self.state_head(mask_roi_feats)  # (N_inst, 3)：occlusion/defocus/visible

        return {
            'probe_cls': probe_det_outputs['cls'],
            'probe_reg': probe_det_outputs['reg'],
            'probe_ctr': probe_det_outputs['ctr'],
            'probe_det_results': probe_det_results,  # 推理时: List[B][dict]; 训练时: None

            'calib_cls': calib_outputs['cls'],
            'calib_reg': calib_outputs['reg'],
            'calib_ctr': calib_outputs['ctr'],

            'mask_logits': mask_logits,  # (N_inst, 1, roi_h, roi_w)
            'mask_batch_idx': mask_batch_idx,

            'state_logits': state_logits,  # (N_inst, 3)，sigmoid前
            'state_batch_idx': mask_batch_idx,  # 与 mask_batch_idx 同序（同一份 rboxes_list_for_roi 展平）

            'kp_heatmap': kp_heatmap,
            'kp_log_sigma2': kp_log_sigma2,
            'kp_vis_logits': kp_vis_logits,  # (N_inst, K, 3)
            'kp_batch_idx': kp_batch_idx,
            'kp_rboxes_list': rboxes_list_for_roi,
        }

    def postprocess(
            self,
            outputs: Dict,
            x_shape: Tuple[int, int, int, int],
            score_thresh_calib: float = 0.3,
            mask_thresh: float = 0.5,
    ):
        _, _, img_h, img_w = x_shape
        B = len(outputs['kp_rboxes_list'])
        probe_det_results = outputs['probe_det_results']
        if probe_det_results is None:
            raise RuntimeError("postprocess 只能用于推理模式（forward 时 rboxes_list=None）")

        kp_results = self.kp_head.decode(
            outputs['kp_heatmap'], outputs['kp_log_sigma2'],
            outputs['kp_rboxes_list'], outputs['kp_batch_idx'],
            vis_logits=outputs.get('kp_vis_logits'),
        )
        calib_results = self.calib_det.decode(
            outputs['calib_cls'], outputs['calib_reg'], outputs['calib_ctr'],
            score_thresh=score_thresh_calib,
        )

        mask_probs = torch.sigmoid(outputs['mask_logits'])  # (N_inst,1,roi_h,roi_w)
        mask_batch_idx = outputs['mask_batch_idx'].cpu().numpy()
        state_probs = self.state_head.decode(outputs['state_logits']).detach().cpu().numpy()  # (N_inst,3)

        results_batch = []
        inst_cursor = 0
        for i in range(B):
            dets_i = probe_det_results[i]  # List[dict]：来自 probe_det.decode
            kps_i = kp_results[i]  # 与 dets_i 同序同长（同一份 rboxes_list 派生）
            probes_i = []
            for j, det in enumerate(dets_i):
                cx, cy, w, h, theta = det['rect']
                mask_prob_roi = mask_probs[inst_cursor + j, 0]
                mask_full = paste_roi_mask_to_image(
                    mask_prob_roi, cx, cy, w, h, theta, img_h, img_w,
                ) >= mask_thresh
                inst = dict(det)
                inst['mask'] = mask_full.detach().cpu().numpy().astype(bool)
                inst['keypoints'] = kps_i[j] if j < len(kps_i) else []
                # 探针状态：遮挡率 / 失焦度 / 几何可见比例（部分结构是否出视野）
                occ, defocus, vis_ratio = state_probs[inst_cursor + j]
                inst['occlusion_ratio'] = float(occ)
                inst['defocus_level'] = float(defocus)
                inst['visible_ratio'] = float(vis_ratio)
                inst['is_partially_out_of_view'] = bool(vis_ratio < 0.98)
                probes_i.append(inst)
            inst_cursor += len(dets_i)

            results_batch.append({
                'probe': probes_i,
                'calibs': calib_results[i],
                'num_probe': len(probes_i),
                'num_kps': len(kps_i),
                'num_calibs': len(calib_results[i]),
            })

        return results_batch

    # ── 预留计算接口 ──────────────────────────────────────────

    @staticmethod
    def compute_contact_resistance(
            probe_keypoints: List[Dict],
            calib_pads: List[Dict],
            px_per_um: float = 1.0,
    ) -> Optional[float]:
        """
        TODO: 计算接触电阻（需结合外部测量数据）。
        预留接口：根据关键点与校准片相对位置计算接触区域参数。
        """
        raise NotImplementedError("预留：接触电阻计算")

    @staticmethod
    def compute_alignment_error(
            probe_keypoints: List[Dict],
            calib_pads: List[Dict],
            nominal_offset_um: Tuple[float, float] = (0.0, 0.0),
            px_per_um: float = 1.0,
    ) -> Optional[Dict]:
        """
        TODO: 计算探针对准误差（dx, dy）单位μm。
        预留接口：针尖坐标与校准片中心坐标之差，减去标称偏移。
        返回：{'dx_um': float, 'dy_um': float, 'total_um': float}
        """
        raise NotImplementedError("预留：探针对准误差计算")


# ══════════════════════════════════════════════════════════════
#  多任务损失函数
# ══════════════════════════════════════════════════════════════

class MultiTaskLoss(nn.Module):
    TASK_NAMES = ['probe_det', 'keypoint', 'calib_det', 'probe_state']

    def __init__(self, num_tasks: int = 4, momentum: float = 0.98):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(num_tasks))
        # 指数滑动平均的"该任务最近是否活跃"统计，仅用于日志展示，不参与梯度
        self.momentum = momentum
        self.register_buffer('active_rate', torch.ones(num_tasks))

    def forward(
            self,
            losses: Dict[str, torch.Tensor],
            task_stats: Dict[str, int],
    ) -> torch.Tensor:
        log_sigmas_clamped = self.log_sigmas.clamp(-1.5, 1.5)
        weighted = []

        for i, name in enumerate(self.TASK_NAMES):
            n_valid = task_stats.get(name, 0)
            self.active_rate[i] = (
                    self.momentum * self.active_rate[i] + (1 - self.momentum) * float(n_valid > 0)
            )

            if n_valid == 0:
                # 本batch该任务无真实监督：不构造任何依赖 log_sigma[i] 的计算图节点，
                # 该参数在本次 backward 中梯度天然为 None / 0，不会被无意义驱动。
                continue

            l = losses.get(name)
            if l is None:
                continue
            log_s = log_sigmas_clamped[i]
            weighted.append(0.5 * torch.exp(-2 * log_s) * l + log_s)

        if not weighted:
            # 整个batch三个任务都没有监督，这种情况理论上不该发生
            # （除非batch_size=0或数据异常），但保留兜底，避免训练崩溃。
            return self.log_sigmas.sum() * 0.0

        return sum(weighted)

    def task_weights(self, task_stats: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        log_sigmas_clamped = self.log_sigmas.clamp(-3.0, 3.0)
        weights = {}
        for i, name in enumerate(self.TASK_NAMES):
            if task_stats is not None and task_stats.get(name, 0) == 0:
                weights[name] = 0.0  # 本batch未激活，权重展示为0，避免误导
            else:
                weights[name] = float((0.5 * torch.exp(-2 * log_sigmas_clamped[i])).detach())
        return weights


# ══════════════════════════════════════════════════════════════
#  Keypoint GT：直接在 RoI 局部坐标系生成
# ══════════════════════════════════════════════════════════════

def soft_argmax_2d(heatmap_logits: torch.Tensor, temperature: float = 100.0) -> torch.Tensor:
    """
    heatmap_logits: (N, K, H, W) 未激活 logits
    返回: (N, K, 2) —— (x, y)，单位为该 heatmap 自身的网格坐标（如 roi_w=roi_h=32 时是 [0,32) 局部坐标）
    训练（loss）和推理（decode）共用同一实现，避免两处坐标定义"各写一遍、悄悄漂移"。
    """
    N, K, H, W = heatmap_logits.shape
    prob = torch.sigmoid(heatmap_logits)
    flat = prob.reshape(N, K, -1)
    weights = torch.softmax(flat * temperature, dim=-1)
    device = heatmap_logits.device
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    gx_flat = gx.reshape(1, 1, -1)
    gy_flat = gy.reshape(1, 1, -1)
    x = (weights * gx_flat).sum(-1)
    y = (weights * gy_flat).sum(-1)
    return torch.stack([x, y], dim=-1)  # (N, K, 2)


def build_rotated_affine_theta(rois_cx, rois_cy, rois_w, rois_h, rois_theta,
                               stride, feat_h, feat_w):
    """
    rois_*: (N,) tensor，图像坐标系下的 cx,cy,w,h,theta(弧度)
    返回: (N,2,3) theta矩阵，供 F.affine_grid 使用
    """
    cos_t = torch.cos(rois_theta)
    sin_t = torch.sin(rois_theta)

    sx = (rois_w / 2.0) / stride * (2.0 / max(feat_w - 1, 1))
    sy = (rois_h / 2.0) / stride * (2.0 / max(feat_h - 1, 1))

    tx = 2.0 * (rois_cx / stride) / max(feat_w - 1, 1) - 1.0
    ty = 2.0 * (rois_cy / stride) / max(feat_h - 1, 1) - 1.0

    theta = torch.zeros(rois_cx.shape[0], 2, 3, device=rois_cx.device, dtype=rois_cx.dtype)
    theta[:, 0, 0] = sx * cos_t
    theta[:, 0, 1] = -sy * sin_t
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sx * sin_t
    theta[:, 1, 1] = sy * cos_t
    theta[:, 1, 2] = ty
    return theta


def build_keypoint_targets_roi(rboxes_list, keypoints_list, visibility_list,
                               roi_size=(32, 32), sigma=2.0, num_keypoints=16, device="cuda"):
    roi_h, roi_w = roi_size
    two_sigma2 = 2.0 * sigma ** 2
    heatmaps, vis_out, xy_out = [], [], []  # ← 新增 xy_out

    for b, boxes in enumerate(rboxes_list):
        n_i = len(boxes)
        if n_i == 0:
            continue
        gy, gx = torch.meshgrid(
            torch.arange(roi_h, dtype=torch.float32, device=device),
            torch.arange(roi_w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        gx, gy = gx.unsqueeze(0), gy.unsqueeze(0)

        kps_b, vis_b = keypoints_list[b], visibility_list[b]
        for inst_idx in range(n_i):
            box = boxes[inst_idx]
            cx, cy, w, h, theta = box[0], box[1], box[2], box[3], box[4]

            kp = torch.as_tensor(kps_b[inst_idx], dtype=torch.float32, device=device).reshape(-1, 2)
            vis = torch.as_tensor(vis_b[inst_idx], dtype=torch.float32, device=device).reshape(-1)
            if kp.shape[0] < num_keypoints:
                kp = torch.cat([kp, torch.zeros(num_keypoints - kp.shape[0], 2, device=device)], dim=0)
            if vis.shape[0] < num_keypoints:
                vis = torch.cat([vis, torch.zeros(num_keypoints - vis.shape[0], device=device)], dim=0)
            kp, vis = kp[:num_keypoints], vis[:num_keypoints]

            theta_tensor = torch.tensor(theta, device=device)
            cos_t, sin_t = torch.cos(theta_tensor), torch.sin(theta_tensor)
            dx, dy = kp[:, 0] - cx, kp[:, 1] - cy
            x_local = dx * cos_t + dy * sin_t
            y_local = -dx * sin_t + dy * cos_t
            kp_x = (x_local / w + 0.5) * roi_w
            kp_y = (y_local / h + 0.5) * roi_h

            # ---- 新增：把 ROI 局部坐标也存下来，供坐标级 NLL 使用 ----
            # clip 到合法网格范围内：这是对"标注目标"的合理裁剪（防止个别越界标注/框回归误差
            # 产生的巨大残差把 sigma 学习带偏），不是对 loss 结果的事后裁剪。
            kp_xy_clipped = torch.stack([
                kp_x.clamp(0, roi_w - 1), kp_y.clamp(0, roi_h - 1)
            ], dim=-1)  # (K, 2)
            xy_out.append(kp_xy_clipped)

            gaussian = torch.exp(-((gx - kp_x[:, None, None]) ** 2 + (gy - kp_y[:, None, None]) ** 2) / two_sigma2)
            vis_mask = (vis > 0).float()[:, None, None]
            heatmaps.append(gaussian * vis_mask)
            vis_out.append(vis)

    if len(heatmaps) == 0:
        return None, None, None

    return torch.stack(heatmaps), torch.stack(vis_out), torch.stack(xy_out)  # 新增第三项


def build_detection_targets(
        images, labels, rboxes, stride, num_classes,
        size_range=(0, 9999), center_sampling_radius=1.5,):
    B, _, H, W = images.shape
    feat_h, feat_w = H // stride, W // stride
    device = images.device
    size_lo, size_hi = size_range

    cls_target = torch.zeros(B, num_classes, feat_h, feat_w, device=device)
    bbox_target = torch.zeros(B, feat_h, feat_w, 4, device=device)  # ← 6改4：dx,dy,w,h
    centerness_target = torch.zeros(B, feat_h, feat_w, device=device)
    angle_target = torch.zeros(B, feat_h, feat_w, device=device)
    area_map = torch.full((B, feat_h, feat_w), float('inf'), device=device)

    xs = torch.arange(feat_w, device=device, dtype=torch.float32)
    ys = torch.arange(feat_h, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    cx_grid = xx + 0.5
    cy_grid = yy + 0.5

    for b in range(B):
        for label, rbox in zip(labels[b], rboxes[b]):
            label = int(label)
            if label < 0 or label >= num_classes:
                continue
            cx0, cy0, bw, bh, theta0 = [float(v) for v in rbox[:5]]
            if bw <= 0 or bh <= 0:
                continue
            # 正样本候选区域仍用 AABB（保证候选区域覆盖整个旋转框），
            # AABB 由统一的 cx,cy,w,h,theta 旋转框角点推出，不再依赖单独的轴对齐标注
            corners = GSGProbeDataset.rect_to_corners(cx0, cy0, bw, bh, theta0)
            x1, y1 = float(corners[:, 0].min()), float(corners[:, 1].min())
            x2, y2 = float(corners[:, 0].max()), float(corners[:, 1].max())
            max_side = max(bw, bh)
            if not (size_lo <= max_side < size_hi):
                continue

            # ── 正样本区域仍用 AABB（保证候选区域覆盖整个旋转框）──
            x1_f, y1_f, x2_f, y2_f = x1 / stride, y1 / stride, x2 / stride, y2 / stride
            cx_obj, cy_obj = (x1_f + x2_f) / 2.0, (y1_f + y2_f) / 2.0

            in_box = (cx_grid >= x1_f) & (cx_grid <= x2_f) & (cy_grid >= y1_f) & (cy_grid <= y2_f)
            if center_sampling_radius and center_sampling_radius > 0:
                in_center = (((cx_grid - cx_obj).abs() <= center_sampling_radius) &
                             ((cy_grid - cy_obj).abs() <= center_sampling_radius))
                positive_mask = in_box & in_center
            else:
                positive_mask = in_box
            if not positive_mask.any():
                continue

            # centerness 仍按距 AABB 四边的距离算，衡量"离物体中心有多近"
            l = cx_grid - x1_f;
            t = cy_grid - y1_f;
            r = x2_f - cx_grid;
            b_ = y2_f - cy_grid
            area = (x2_f - x1_f) * (y2_f - y1_f)
            update_mask = positive_mask & (area < area_map[b])
            if not update_mask.any():
                continue
            area_map[b][update_mask] = area

            cls_target[b, :, update_mask] = 0.0
            cls_target[b, label][update_mask] = 1.0

            # ── 新：回归目标改为紧贴框的 dx,dy,w,h（feature-map单位）──
            cx_tight = float(rbox[0]) / stride
            cy_tight = float(rbox[1]) / stride
            w_tight = max(float(rbox[2]) / stride, 1e-3)
            h_tight = max(float(rbox[3]) / stride, 1e-3)
            dx_map = torch.full_like(cx_grid, cx_tight) - cx_grid
            dy_map = torch.full_like(cy_grid, cy_tight) - cy_grid
            reg = torch.stack([dx_map, dy_map,
                               torch.full_like(cx_grid, w_tight),
                               torch.full_like(cy_grid, h_tight)], dim=-1)
            bbox_target[b][update_mask] = reg[update_mask]

            centerness = torch.sqrt(
                (torch.min(l, r) / (torch.max(l, r) + 1e-6)) *
                (torch.min(t, b_) / (torch.max(t, b_) + 1e-6))
            )
            centerness_target[b][update_mask] = centerness[update_mask]

            theta_rad = float(rbox[4])
            angle_target[b][update_mask] = theta_rad

    return cls_target, bbox_target, centerness_target, angle_target
