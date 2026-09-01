"""
predict.py
──────────
加载 GSGProbeNet 的 .pth 权重，对文件夹（或单张图片）批量推理，
并将结果可视化保存到 output_dir；也支持基于 GSGProbeDataset 的
推理 + GT 指标评估。

用法示例：
    # 纯图片推理（无标注）
    python predict.py \
        --weights checkpoints/best.pth \
        --input   data/test_images \
        --output  results \
        --imgsz   640 \
        --device  cuda

    # 数据集推理 + metrics 评估
    python predict.py \
        --weights checkpoints/best.pth \
        --input   data/raw \
        --mode    dataset --split test \
        --device  cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline import build_val_transforms, GSGProbeDataset, DATASET_CATEGORY_NAME
from model import GSGProbeNet, collate_fn
from metrics import GSGProbeMetrics
from train import logger

# ════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ════════════════════════════════════════════════════════════
#  Letterbox 坐标反变换
# ════════════════════════════════════════════════════════════
#  data_pipeline.build_val_transforms / build_test_transforms 对图像做的是：
#      1) A.LongestMaxSize(max_size=image_size)          等比例缩放，scale = image_size / max(H0, W0)
#      2) A.PadIfNeeded(min_height=image_size,
#                        min_width=image_size, ...)       居中补零到 image_size × image_size
#                                                          （position 参数未显式设置，Albumentations
#                                                           默认值为 'center'）
#      3) A.Resize(image_size, image_size)                此时已经是 image_size×image_size，空操作
#  val/test 阶段不含随机几何增强（train 阶段的 A.Affine 才有随机性），所以
#  只根据"原图 (H0, W0)"就能精确复现 scale 和 pad 偏移，不需要修改
#  data_pipeline.py、也不需要侵入 Albumentations 内部状态。
# ════════════════════════════════════════════════════════════

@dataclass
class LetterboxInfo:
    orig_h: int
    orig_w: int
    scale: float  # 原图 → letterbox 画布的等比例缩放系数（x/y 相同）
    pad_top: int
    pad_left: int
    resized_h: int  # 缩放后、补边前的高度
    resized_w: int  # 缩放后、补边前的宽度

    @classmethod
    def compute(cls, orig_h: int, orig_w: int, image_size: int) -> "LetterboxInfo":
        scale = image_size / max(orig_h, orig_w)
        resized_h = round(orig_h * scale)
        resized_w = round(orig_w * scale)
        pad_h = image_size - resized_h
        pad_w = image_size - resized_w
        # PadIfNeeded(position='center') 的实际切分方式：pad_before = pad // 2（向下取整）
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        return cls(orig_h, orig_w, scale, pad_top, pad_left, resized_h, resized_w)

    def xy_to_original(self, x: float, y: float) -> Tuple[float, float]:
        return (x - self.pad_left) / self.scale, (y - self.pad_top) / self.scale

    def wh_to_original(self, w: float, h: float) -> Tuple[float, float]:
        # 只有平移+等比例缩放，角度 theta 不受影响，不需要转换
        return w / self.scale, h / self.scale

    def mask_to_original(self, mask_letterboxed: np.ndarray) -> np.ndarray:
        """
        mask_letterboxed: (image_size, image_size) bool。
        必须先裁掉 letterbox 补的黑边，再等比例缩放回原图尺寸——
        直接对整张 640×640 mask 做 resize 到 (orig_h, orig_w) 会把黑边一起
        拉伸进画面，导致内容区域被压缩、mask 形状失真。
        """
        top, left = self.pad_top, self.pad_left
        cropped = mask_letterboxed[top: top + self.resized_h, left: left + self.resized_w]
        resized = cv2.resize(
            cropped.astype(np.uint8), (self.orig_w, self.orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized.astype(bool)


# ════════════════════════════════════════════════════════════
#  Predictor
# ════════════════════════════════════════════════════════════
class Predictor:
    """
    GSGProbeNet 推理封装。

    两种推理入口：
        predict_images()   —— 纯图片文件夹推理（无标注）
        predict_dataset()  —— 带 Dataset/DataLoader 推理（含 metrics 评估）

    result 结构约定（来自 GSGProbeNet.postprocess，见 model.py）：
        {
            'probe': [
                {
                    'class_id', 'label', 'score',
                    'cx', 'cy', 'w', 'h', 'angle',
                    'rect': [cx, cy, w, h, theta_rad],           # 扁平，供 IoU 匹配
                    'rotated_rect': ((cx, cy), (w, h), theta_rad),  # 嵌套，供 cv2 画图（弧度！）
                    'mask': ndarray(H, W) bool，已贴回原图分辨率,
                    'keypoints': [
                        {'keypoint_id', 'x', 'y', 'confidence', 'sigma'}, ...
                    ],  # 与该 probe 实例强绑定，不在 result 顶层
                }, ...
            ],
            'calibs': [ 同 probe 的检测字段，无 mask / keypoints ],
            'num_probe', 'num_kps', 'num_calibs',
        }
    """

    # 16 个关键点的可视化颜色（BGR）
    KP_COLORS: List[Tuple[int, int, int]] = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
        (0, 128, 255), (128, 255, 0), (0, 255, 128), (255, 0, 128),
        (128, 128, 0), (0, 128, 128), (128, 0, 255), (255, 128, 128),
    ]

    # ──────────────────────────────────────────────────────────
    #  初始化
    # ──────────────────────────────────────────────────────────
    def __init__(self, config: Dict):
        self.cfg = config
        self.model: Optional[GSGProbeNet] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = GSGProbeMetrics()
        logger.info(f"Predictor 初始化，设备: {self.device}")

    # ──────────────────────────────────────────────────────────
    #  模型加载
    # ──────────────────────────────────────────────────────────

    def load_model(
            self,
            weights_path: str,
            config: Optional[Dict] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        """
        从 .pth 文件加载 GSGProbeNet。支持两种 checkpoint 格式：
          1. 仅 state_dict：torch.save(model.state_dict(), path)
          2. 含元信息字典：torch.save({'model_state_dict': ..., 'config': ...}, path)

        config 优先级（从低到高）：
          安全默认值  <  checkpoint 自带的 'config'  <  显式传入的 config 参数
        这样 checkpoint 训练时用的 hrnet_root/px_per_um 等结构参数会被优先
        采用，调用方只需在需要覆盖时显式传参。
        """
        device = device or self.device
        checkpoint = torch.load(weights_path, map_location='cpu')

        ckpt_config: Optional[Dict] = None
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            ckpt_config = checkpoint.get('config')
        else:
            state_dict = checkpoint

        merged_config: Dict = {}
        if ckpt_config:
            merged_config.update(ckpt_config)
        if config:
            merged_config.update(config)

        self.model = GSGProbeNet(merged_config)

        # 兼容 DataParallel 保存的 "module." 前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"缺失参数 ({len(missing)} 个)：{missing[:5]} ...")
        if unexpected:
            logger.warning(f"多余参数 ({len(unexpected)} 个)：{unexpected[:5]} ...")

        self.model.to(device)
        self.device = device
        logger.info(f"模型加载完成：{weights_path}")

    def _ensure_model(self) -> None:
        """推理前确保模型已加载并处于 eval 模式。"""
        if self.model is None:
            raise RuntimeError("请先调用 load_model() 加载权重。")
        self.model.eval()

    # ──────────────────────────────────────────────────────────
    #  图像收集工具
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def collect_images(input_path: str) -> List[Path]:
        """递归收集目录下（或单个）合法图像路径，按文件名排序。"""
        p = Path(input_path)
        if p.is_file():
            return [p] if p.suffix.lower() in IMG_EXTENSIONS else []
        if p.is_dir():
            return sorted(f for f in p.rglob('*') if f.suffix.lower() in IMG_EXTENSIONS)
        raise FileNotFoundError(f"路径不存在: {input_path}")

    # ──────────────────────────────────────────────────────────
    #  坐标系：把 letterbox（640×640）空间的预测/GT 映射回原图坐标
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _remap_result_to_original(result: Dict, info: "LetterboxInfo") -> Dict:
        """把 model.postprocess 输出的预测结果，从 640×640 letterbox 坐标映射回原图坐标。"""

        def _remap_flat_rect(rect_flat):
            cx, cy, w, h, theta = [float(v) for v in rect_flat[:5]]
            cx, cy = info.xy_to_original(cx, cy)
            w, h = info.wh_to_original(w, h)
            return [cx, cy, w, h, theta]

        def _remap_nested_rect(rotated_rect):
            (cx, cy), (w, h), theta = rotated_rect
            cx, cy = info.xy_to_original(float(cx), float(cy))
            w, h = info.wh_to_original(float(w), float(h))
            return (cx, cy), (w, h), theta

        def _remap_instance(inst: Dict) -> Dict:
            new_inst = dict(inst)
            if new_inst.get('rect') is not None:
                new_inst['rect'] = _remap_flat_rect(new_inst['rect'])
            if new_inst.get('rotated_rect') is not None:
                new_inst['rotated_rect'] = _remap_nested_rect(new_inst['rotated_rect'])
            # cx/cy/w/h 是和 rect 冗余的展开字段，一并同步，避免下游误用旧值
            if 'cx' in new_inst and 'cy' in new_inst:
                new_inst['cx'], new_inst['cy'] = info.xy_to_original(
                    float(new_inst['cx']), float(new_inst['cy']))
            if 'w' in new_inst and 'h' in new_inst:
                new_inst['w'], new_inst['h'] = info.wh_to_original(
                    float(new_inst['w']), float(new_inst['h']))
            if new_inst.get('mask') is not None:
                new_inst['mask'] = info.mask_to_original(
                    np.asarray(new_inst['mask'], dtype=bool))
            new_kps = []
            for kp in new_inst.get('keypoints', []):
                kp = dict(kp)
                kp['x'], kp['y'] = info.xy_to_original(float(kp['x']), float(kp['y']))
                new_kps.append(kp)
            new_inst['keypoints'] = new_kps
            return new_inst

        remapped = dict(result)
        remapped['probe'] = [_remap_instance(inst) for inst in result.get('probe', [])]
        remapped['calibs'] = [_remap_instance(det) for det in result.get('calibs', [])]
        return remapped

    @staticmethod
    def _remap_gt_to_original(gt_item: Dict, info: "LetterboxInfo") -> Dict:
        """
        把 GSGProbeDataset 输出的 GT 标注从 640×640 letterbox 坐标映射回原图坐标。
        val/test 阶段的数据增强是确定性的（无随机几何变换），所以 GT 和预测
        共享同一个 LetterboxInfo，映射方式完全一致。

        这一步同样重要：如果只把预测映射回原图、GT 仍留在 640 空间，
        metrics 里的 IoU/像素误差匹配会直接错位；如果 pred 和 GT 都不映射
        （即都留在 640 空间），IoU 匹配虽然自洽，但 kp_mean_error_px 等
        “绝对像素误差”会随每张原图分辨率/长宽比不同而含义不一致，
        不适合跨图片汇总统计或写进论文里的“像素误差”指标。
        """
        remapped = dict(gt_item)

        def _remap_rbox_list(rboxes):
            out = []
            for rbox in rboxes:
                cx, cy, w, h, theta = [float(v) for v in rbox[:5]]
                cx, cy = info.xy_to_original(cx, cy)
                w, h = info.wh_to_original(w, h)
                out.append([cx, cy, w, h, theta])
            return out

        remapped['probe_rboxes'] = _remap_rbox_list(gt_item.get('probe_rboxes', []))
        remapped['calib_rboxes'] = _remap_rbox_list(gt_item.get('calib_rboxes', []))

        remapped_kp = []
        for kps in gt_item.get('probe_keypoints', []):
            kps = np.asarray(kps, dtype=np.float32)
            if kps.size == 0:
                remapped_kp.append(kps)
                continue
            xs = (kps[:, 0] - info.pad_left) / info.scale
            ys = (kps[:, 1] - info.pad_top) / info.scale
            remapped_kp.append(np.stack([xs, ys], axis=-1))
        remapped['probe_keypoints'] = remapped_kp

        remapped_masks = []
        for mask in gt_item.get('probe_masks', []):
            mask_bool = np.asarray(mask, dtype=bool)
            remapped_masks.append(info.mask_to_original(mask_bool))
        remapped['probe_masks'] = remapped_masks

        return remapped

    # ──────────────────────────────────────────────────────────
    #  可视化
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _draw_rotated_box(
            vis: np.ndarray,
            rotated_rect: Tuple[Tuple[float, float], Tuple[float, float], float],
            color: Tuple[int, int, int],
            label_text: str,
    ) -> None:
        """rotated_rect: ((cx,cy),(w,h),theta_rad) —— 角度是弧度，cv2.boxPoints 要角度。"""
        (cx, cy), (rw, rh), theta_rad = rotated_rect
        angle_deg = math.degrees(float(theta_rad))
        box_pts = cv2.boxPoints(((float(cx), float(cy)), (float(rw), float(rh)), angle_deg))
        box_pts = np.intp(box_pts)
        cv2.drawContours(vis, [box_pts], 0, color, 2)
        top_pt = box_pts[box_pts[:, 1].argmin()]
        cv2.putText(
            vis, label_text, (int(top_pt[0]), int(top_pt[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )

    def visualize(
            self,
            img_bgr: np.ndarray,
            result: Dict,
            kp_thresh: float = 0.3,
            show_mask: bool = True,
    ) -> np.ndarray:
        """在原图上叠加推理结果（探针 mask/旋转框/关键点 + 校准片旋转框），返回标注后的 BGR 图像。"""
        vis = img_bgr.copy()
        img_h, img_w = vis.shape[:2]

        probe_list = result.get('probe', [])
        calib_list = result.get('calibs', [])

        # ── 探针：mask + 旋转框 + 关键点（三者都挂在同一个实例字典下）──────
        for inst in probe_list:
            if show_mask:
                mask = inst.get('mask')
                if mask is not None:
                    mask_bool = np.asarray(mask, dtype=bool)
                    # postprocess 里 paste_roi_mask_to_image 已把 mask 贴回原图
                    # 分辨率；这里的 resize 只是兜底，防止上游分辨率意外不一致。
                    if mask_bool.shape != (img_h, img_w):
                        mask_bool = cv2.resize(
                            mask_bool.astype(np.uint8), (img_w, img_h),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    overlay = vis.copy()
                    overlay[mask_bool] = (0, 120, 255)  # 橙色
                    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

            rotated_rect = inst.get('rotated_rect')
            if rotated_rect is not None:
                cls_name = DATASET_CATEGORY_NAME.get(inst.get('class_id'), 'probe')
                self._draw_rotated_box(
                    vis, rotated_rect, (0, 200, 255),
                    f"{cls_name}:{inst.get('score', 0.0):.2f}",
                )
                if 'occlusion_ratio' in inst:
                    (cx, cy), (_, _), _ = rotated_rect
                    state_text = (
                        f"occ:{inst['occlusion_ratio']:.2f} "
                        f"defocus:{inst['defocus_level']:.2f} "
                        f"vis:{inst['visible_ratio']:.2f}"
                    )
                    state_color = (0, 0, 255) if inst.get('is_partially_out_of_view') else (0, 200, 255)
                    cv2.putText(
                        vis, state_text, (int(cx) - 40, int(cy) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, state_color, 1,
                    )

            # 关键点：与该 probe 实例强绑定，不在 result 顶层
            inst_kps = inst.get('keypoints', [])
            if not inst_kps:
                continue

            pts = [(kp['x'], kp['y'], kp['confidence']) for kp in inst_kps]

            # 骨架连线（相邻关键点序号，非解剖学意义上的骨架，仅辅助观察）
            for k in range(len(pts) - 1):
                if pts[k][2] < kp_thresh or pts[k + 1][2] < kp_thresh:
                    continue
                xa, ya = int(round(pts[k][0])), int(round(pts[k][1]))
                xb, yb = int(round(pts[k + 1][0])), int(round(pts[k + 1][1]))
                cv2.line(vis, (xa, ya), (xb, yb), (200, 200, 200), 1)

            # 关键点圆点 + 标注：预测为“遮挡”的点画空心圈，“缺失/出界”的点直接跳过绘制
            for kp in inst_kps:
                if kp['confidence'] < kp_thresh:
                    continue
                vis_pred = kp.get('visibility_pred', 2)  # 无该字段时按可见处理，兼容旧模型
                if vis_pred == 0:
                    continue  # 预测为缺失/出界，不画点，避免误导
                x = int(round(kp['x']))
                y = int(round(kp['y']))
                kid = kp['keypoint_id']
                color = self.KP_COLORS[kid % len(self.KP_COLORS)]
                if vis_pred == 1:
                    cv2.circle(vis, (x, y), 5, color, 1)  # 遮挡：空心圈
                    tag = f"K{kid}(occ,{kp['confidence']:.2f})"
                else:
                    cv2.circle(vis, (x, y), 5, color, -1)  # 可见：实心圆
                    tag = f"K{kid}({kp['confidence']:.2f})"
                cv2.putText(
                    vis, tag, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )

        # ── 校准片：旋转框 ─────────────────────────────────────────────
        for det in calib_list:
            rotated_rect = det.get('rotated_rect')
            if rotated_rect is None:
                continue
            cls_name = DATASET_CATEGORY_NAME.get(det.get('class_id'), 'calib')
            self._draw_rotated_box(
                vis, rotated_rect, (0, 255, 80),
                f"{cls_name}:{det.get('score', 0.0):.2f}",
            )

        return vis

    # ──────────────────────────────────────────────────────────
    #  结果序列化
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def result_to_json(result: Dict) -> Dict:

        out: Dict = {}

        # ── probe（含内嵌 keypoints）──────────────────────────────────
        probe_list = []
        for inst in result.get('probe', []):
            entry: Dict = {
                'score': float(inst.get('score', 0.0)),
                'class_id': int(inst.get('class_id', 0)),
                'label': int(inst.get('label', 0)),
            }
            rect = inst.get('rect')  # 扁平 [cx, cy, w, h, theta_rad]
            if rect is not None:
                cx, cy, rw, rh, theta_rad = rect
                entry['rect'] = {
                    'cx': float(cx), 'cy': float(cy),
                    'w': float(rw), 'h': float(rh),
                    'theta_rad': float(theta_rad),
                }
            # 探针状态：遮挡率 / 失焦度 / 几何可见比例（部分结构是否出视野）
            if 'occlusion_ratio' in inst:
                entry['state'] = {
                    'occlusion_ratio': float(inst.get('occlusion_ratio', 0.0)),
                    'defocus_level': float(inst.get('defocus_level', 0.0)),
                    'visible_ratio': float(inst.get('visible_ratio', 1.0)),
                    'is_partially_out_of_view': bool(inst.get('is_partially_out_of_view', False)),
                }
            entry['keypoints'] = [
                {
                    'kid': int(kp['keypoint_id']),
                    'x': float(kp['x']),
                    'y': float(kp['y']),
                    'confidence': float(kp['confidence']),
                    'sigma': float(kp['sigma']),
                    **({
                        'visibility_pred': int(kp['visibility_pred']),
                        'visibility_pred_name': kp['visibility_pred_name'],
                    } if 'visibility_pred' in kp else {}),
                }
                for kp in inst.get('keypoints', [])
            ]
            probe_list.append(entry)
        out['probe'] = probe_list

        # ── calibs ───────────────────────────────────────────────────
        calib_list = []
        for det in result.get('calibs', []):
            entry = {
                'class_id': int(det.get('class_id', 0)),
                'label': int(det.get('label', 0)),
                'score': float(det.get('score', 0.0)),
            }
            rect = det.get('rect')  # 扁平 [cx, cy, w, h, theta_rad]
            if rect is not None:
                cx, cy, rw, rh, theta_rad = rect
                entry['rect'] = {
                    'cx': float(cx), 'cy': float(cy),
                    'w': float(rw), 'h': float(rh),
                    'theta_rad': float(theta_rad),
                }
            calib_list.append(entry)
        out['calibs'] = calib_list

        return out

    # ──────────────────────────────────────────────────────────
    #  输出目录初始化（复用逻辑）
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_output_dirs(
            out_root: Path, save_vis: bool, save_json: bool
    ) -> Tuple[Path, Path]:
        vis_dir = out_root / 'visualizations'
        json_dir = out_root / 'json'
        if save_vis:
            vis_dir.mkdir(parents=True, exist_ok=True)
        if save_json:
            json_dir.mkdir(parents=True, exist_ok=True)
        return vis_dir, json_dir

    @staticmethod
    def _save_result(
            stem: str,
            img_bgr: Optional[np.ndarray],
            result: Dict,
            vis_fn,  # callable(img_bgr, result) -> np.ndarray
            vis_dir: Path,
            json_dir: Path,
            save_vis: bool,
            save_json: bool,
            img_path: Optional[Path] = None,
    ) -> None:
        """保存单张图片的可视化和 JSON 结果。"""
        if save_vis and img_bgr is not None:
            cv2.imwrite(str(vis_dir / f'{stem}_pred.jpg'), vis_fn(img_bgr, result))
        if save_json:
            data = Predictor.result_to_json(result)
            if img_path is not None:
                data['image_path'] = str(img_path)
            with open(json_dir / f'{stem}.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    # ──────────────────────────────────────────────────────────
    #  推理入口 ①：纯图片文件夹推理（无标注）
    # ──────────────────────────────────────────────────────────

    def predict_images(
            self,
            device_str: str = 'cuda',
            save_vis: bool = True,
            save_json: bool = True,
    ) -> List[Dict]:
        """
        读取 cfg['data_root'] 下所有图像，逐张推理并保存结果。

        返回：List[{'image_path': str, 'result': Dict}]
        """
        # ── device ──────────────────────────────────────────────────
        if device_str == 'cuda' and not torch.cuda.is_available():
            logger.warning('CUDA 不可用，已切换到 CPU。')
            device_str = 'cpu'
        self.device = torch.device(device_str)

        # ── 加载模型 ─────────────────────────────────────────────────
        logger.info(f'加载模型：{self.cfg["pth_dir"]}')
        self.load_model(self.cfg['pth_dir'], config=self.cfg, device=self.device)
        self._ensure_model()

        # ── 收集图像 ─────────────────────────────────────────────────
        image_paths = self.collect_images(self.cfg['predict_data_dir'])
        if not image_paths:
            logger.warning(f'未找到图像：{self.cfg["data_root"]}')
            return []
        logger.info(f'共找到 {len(image_paths)} 张图像')

        # ── 输出目录 ─────────────────────────────────────────────────
        out_root = Path(self.cfg['predict_data_dir'] + "/results")
        vis_dir, json_dir = self._prepare_output_dirs(out_root, save_vis, save_json)

        # ── transform（与训练 val 阶段一致）──────────────────────────
        transform = build_val_transforms(self.cfg['image_size'])

        probe_score_thresh = self.cfg.get('probe_score_thresh', 0.3)
        calib_score_thresh = self.cfg.get('calib_score_thresh', 0.3)
        mask_thresh = self.cfg.get('mask_thresh', 0.5)

        all_results: List[Dict] = []

        # ── 推理循环 ─────────────────────────────────────────────────
        with torch.no_grad():
            for img_path in tqdm(image_paths):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    logger.warning(f'读取失败，跳过：{img_path}')
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                transformed = transform(
                    image=img_rgb,
                    keypoints=[], kp_visibility=[],
                    bboxes=[], category_ids=[], bbox_track_id=[],
                )
                image_tensor = transformed['image'].unsqueeze(0).to(self.device)

                outputs = self.model(image_tensor, probe_score_thresh=probe_score_thresh)
                result = self.model.postprocess(outputs, x_shape=tuple(image_tensor.shape), )[0]

                letterbox_info = LetterboxInfo.compute(
                    img_bgr.shape[0], img_bgr.shape[1], self.cfg['image_size'])
                result = self._remap_result_to_original(result, letterbox_info)

                self._save_result(
                    stem=Path(img_path).stem,
                    img_bgr=img_bgr,
                    result=result,
                    vis_fn=self.visualize,
                    vis_dir=vis_dir,
                    json_dir=json_dir,
                    save_vis=save_vis,
                    save_json=save_json,
                    img_path=img_path,
                )
                all_results.append({'image_path': str(img_path), 'result': result})

        logger.info(f'推理完成：共处理 {len(all_results)} 张图像')
        return all_results

    # ──────────────────────────────────────────────────────────
    #  推理入口 ②：Dataset + DataLoader 推理（含 metrics 评估）
    # ──────────────────────────────────────────────────────────

    def predict_dataset(
            self,
            split: str = 'val',
            device_str: str = 'cuda',
            save_vis: bool = True,
            save_json: bool = True,
    ) -> List[Dict]:
        """
        通过 GSGProbeDataset + DataLoader 推理，支持 metrics 评估。

        返回：List[{'image_path': str, 'result': Dict}]
        """
        # ── device ──────────────────────────────────────────────────
        if device_str == 'cuda' and not torch.cuda.is_available():
            logger.warning('CUDA 不可用，已切换到 CPU。')
            device_str = 'cpu'
        self.device = torch.device(device_str)

        # ── 加载模型 ─────────────────────────────────────────────────
        logger.info(f'加载模型：{self.cfg["pth_dir"]}')
        self.load_model(self.cfg['pth_dir'], config=self.cfg, device=self.device)
        self._ensure_model()

        # ── 数据集 ───────────────────────────────────────────────────
        dataset = GSGProbeDataset(
            self.cfg['predict_data_dir'],
            split=split,
            transforms=build_val_transforms(self.cfg['image_size']),
            use_pseudo_labels=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 0),
            collate_fn=collate_fn,
        )

        # ── 输出目录 ─────────────────────────────────────────────────
        out_root = Path(self.cfg['predict_data_dir'] + '/results')
        vis_dir, json_dir = self._prepare_output_dirs(out_root, save_vis, save_json)

        probe_score_thresh = self.cfg.get('probe_score_thresh', 0.3)
        calib_score_thresh = self.cfg.get('calib_score_thresh', 0.3)
        mask_thresh = self.cfg.get('mask_thresh', 0.5)

        # ── 推理循环 ─────────────────────────────────────────────────
        self.metrics.reset()
        all_results: List[Dict] = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch['image'].to(self.device)

                t0 = time.perf_counter()
                outputs = self.model(images, probe_score_thresh=probe_score_thresh)
                results = self.model.postprocess(outputs, x_shape=tuple(images.shape))
                inference_time_ms = (time.perf_counter() - t0) * 1000.0 / max(len(results), 1)

                for i, result in enumerate(results):
                    gt_item = {
                        'probe_labels': batch['probe_labels'][i],
                        'probe_rboxes': batch['probe_rboxes'][i],
                        'probe_masks': batch['probe_masks'][i],
                        'probe_keypoints': batch['probe_keypoints'][i],
                        'probe_visibility': batch['probe_visibility'][i],
                        'calib_labels': batch['calib_labels'][i],
                        'calib_rboxes': batch['calib_rboxes'][i],
                    }
                    img_path = Path(batch['img_paths'][i])
                    # 需要原图 (H0, W0) 来重建 LetterboxInfo，所以这里必须读原图，
                    # 不能像旧代码那样只在 save_vis=True 时才读——否则 dataset 模式下
                    # pred/GT 会一直停留在 640×640 letterbox 坐标系，metrics 里的
                    # 像素误差（kp_mean_error_px 等）就不是原图物理像素。
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        logger.warning(f'读取失败，跳过：{img_path}')
                        continue
                    letterbox_info = LetterboxInfo.compute(
                        img_bgr.shape[0], img_bgr.shape[1], self.cfg['image_size'])
                    result_orig = self._remap_result_to_original(result, letterbox_info)
                    gt_item_orig = self._remap_gt_to_original(gt_item, letterbox_info)

                    self.metrics.update(result_orig, gt_item_orig, inference_time_ms=inference_time_ms)
                    self._save_result(
                        stem=img_path.stem,
                        img_bgr=img_bgr,
                        result=result_orig,
                        vis_fn=self.visualize,
                        vis_dir=vis_dir,
                        json_dir=json_dir,
                        save_vis=save_vis,
                        save_json=save_json,
                        img_path=img_path,
                    )
                    all_results.append({'image_path': str(img_path), 'result': result_orig})

        # ── metrics ──────────────────────────────────────────────────
        metrics_result = self.metrics.compute()
        with open(out_root / 'history.json', 'w') as f:
            json.dump(metrics_result, f, indent=2)
        logger.info(f'推理完成：共处理 {len(all_results)} 张图像')
        return all_results
