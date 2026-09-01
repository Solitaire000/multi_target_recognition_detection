"""
data_pipeline.py
================
数据集定义、采集规范、标注格式、增强策略。
"""
from __future__ import annotations
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import torch
from pycocotools import mask as maskUtils
import math

# ── 可选依赖，缺失时降级 ──────────────────────────────────────
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# ══════════════════════════════════════════════════════════════
# 数据增强策略（训练用）
# ══════════════════════════════════════════════════════════════

def build_train_transforms(image_size: int = 640) -> A.Compose:
    return A.Compose([
        # 几何变换
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.8, 1.2),
            rotate=(-30, 30),
            p=0.5
        ),
        # A.RandomCrop(height=image_size, width=image_size, p=0.5),
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
        A.Resize(image_size, image_size),
        # A.ToTensorV2(),
        # 遮挡模拟（模拟探针/校准片被遮挡）
        # A.CoarseDropout(
        #     num_holes_range=(1, 4),  # [min, max] holes
        #     hole_height_range=(20, 80),  # [min, max] height（也支持比例：(0.03, 0.12)）
        #     hole_width_range=(20, 80),  # [min, max] width（也支持比例）
        #     fill=0,  # fill_value 改名为 fill
        #     p=0.3
        # ),

        # 光照变化
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),

        # 颜色扰动（针对不同光源颜色温度）
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5
        ),

        # 噪声/模糊（模拟相机噪声，不模拟散焦——已知图像聚焦正常）
        A.OneOf([
            A.GaussNoise(),
            A.ISONoise(p=1.0),
            A.MultiplicativeNoise(p=1.0),
        ], p=0.4),

        # 归一化
        A.Normalize(),
        ToTensorV2(),
    ],
        keypoint_params=A.KeypointParams(
            format='xy', remove_invisible=False,  # 训练/验证/测试统一不丢弃
            angle_in_degrees=True,
            label_fields=['kp_visibility'],  # 关键：声明标签字段
        ),
        bbox_params=A.BboxParams(
            format='coco', label_fields=['category_ids', 'bbox_track_id'],
            min_visibility=0.0, min_area=0,
        ))


def build_val_transforms(image_size: int = 640) -> A.Compose:
    """验证/测试阶段：仅做尺寸归一化和归一化，不做随机增强"""
    return A.Compose([
        # 注意尺寸
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
        A.Resize(image_size, image_size),

        A.Normalize(),
        ToTensorV2(),
    ],
        keypoint_params=A.KeypointParams(
            format='xy', remove_invisible=False,  # 训练/验证/测试统一不丢弃
            angle_in_degrees=True,
            label_fields=['kp_visibility'],  # 关键：声明标签字段
        ),
        bbox_params=A.BboxParams(
            format='coco', label_fields=['category_ids', 'bbox_track_id'],
            min_visibility=0.0, min_area=0,
        ))


def build_test_transforms(image_size: int = 640) -> A.Compose:
    return A.Compose([
        # 注意尺寸
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
        A.Resize(image_size, image_size),  # 兜底

        A.Normalize(),
        ToTensorV2(),
    ],
        keypoint_params=A.KeypointParams(
            format='xy', remove_invisible=False,  # 训练/验证/测试统一不丢弃
            angle_in_degrees=True,
            label_fields=['kp_visibility'],  # 关键：声明标签字段
        ),
        bbox_params=A.BboxParams(
            format='coco', label_fields=['category_ids'], min_visibility=0.0, min_area=0
        ))


# ══════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════
# =========================
# Dataset Configuration
# =========================
IMAGE_SIZE = 640
NUM_PROBE_KEYPOINTS = 16

# ============================================================================
# Dataset Categories (COCO category_id)
# ============================================================================
DATASET_CATEGORY_ID = {
    # Probe
    "PROBE_G1": 0,
    "PROBE_S": 1,
    "PROBE_G2": 2,
    "PROBE_GSG": 3,
    "PROBE_KEYPOINTS": 4,

    # Calibration
    "LOAD": 10,
    "OPEN": 11,
    "SHORT": 12,
    "THRU": 13,
}
DATASET_CATEGORY_NAME = {
    cid: name
    for name, cid in DATASET_CATEGORY_ID.items()
}
# ============================================================================
# Categories used by each task
# ============================================================================
PROBE_CATEGORY_NAMES = (
    "PROBE_GSG",
)
CALIB_CATEGORY_NAMES = (
    "LOAD",
    "OPEN",
    "SHORT",
    "THRU",
)


# ============================================================================
# 派生变量
# ============================================================================
def build_label_mapping(category_names):
    """根据类别名称生成模型 label 映射"""
    category_ids = tuple(DATASET_CATEGORY_ID[n] for n in category_names)
    category_id_to_label = {
        cid: label
        for label, cid in enumerate(category_ids)
    }
    label_to_category_id = {
        label: cid
        for cid, label in category_id_to_label.items()
    }
    label_names = tuple(
        name.lower()
        for name in category_names
    )
    num_class = len(category_names)

    return (
        category_ids,
        category_id_to_label,
        label_to_category_id,
        label_names,
        num_class,
    )


# probe 派生
(
    PROBE_CATEGORY_IDS,
    PROBE_CATEGORY_ID_TO_LABEL,
    PROBE_LABEL_TO_CATEGORY_ID,
    PROBE_LABEL_NAMES,
    NUM_PROBE_CLASSES,
) = build_label_mapping(PROBE_CATEGORY_NAMES)
# calib 派生
(
    CALIB_CATEGORY_IDS,
    CALIB_CATEGORY_ID_TO_LABEL,
    CALIB_LABEL_TO_CATEGORY_ID,
    CALIB_LABEL_NAMES,
    NUM_CALIB_CLASSES,
) = build_label_mapping(CALIB_CATEGORY_NAMES)

dataset_config = {
    "IMAGE_SIZE": IMAGE_SIZE,
    "NUM_PROBE_KEYPOINTS": NUM_PROBE_KEYPOINTS,

    "dataset": {
        "DATASET_CATEGORY_ID": DATASET_CATEGORY_ID,
        "DATASET_CATEGORY_NAME": DATASET_CATEGORY_NAME,
    },

    "probe": {
        "PROBE_CATEGORY_NAMES": list(PROBE_CATEGORY_NAMES),
        "PROBE_CATEGORY_IDS": list(PROBE_CATEGORY_IDS),
        "PROBE_CATEGORY_ID_TO_LABEL": PROBE_CATEGORY_ID_TO_LABEL,
        "PROBE_LABEL_TO_CATEGORY_ID": PROBE_LABEL_TO_CATEGORY_ID,
        "PROBE_LABEL_NAMES": list(PROBE_LABEL_NAMES),
        "NUM_PROBE_CLASSES": NUM_PROBE_CLASSES,
    },

    "calib": {
        "CALIB_CATEGORY_NAMES": list(CALIB_CATEGORY_NAMES),
        "CALIB_CATEGORY_IDS": list(CALIB_CATEGORY_IDS),
        "CALIB_CATEGORY_ID_TO_LABEL": CALIB_CATEGORY_ID_TO_LABEL,
        "CALIB_LABEL_TO_CATEGORY_ID": CALIB_LABEL_TO_CATEGORY_ID,
        "CALIB_LABEL_NAMES": list(CALIB_LABEL_NAMES),
        "NUM_CALIB_CLASSES": NUM_CALIB_CLASSES,
    }
}


# with open("dataset_config.json", "w", encoding="utf-8") as f:
#     json.dump(dataset_config, f, indent=4, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# 遮挡 / 失焦 / 出界 —— 自监督代理标签（不新增任何人工标注）
# ══════════════════════════════════════════════════════════════
#  设计原则：在不改动现有 COCO 标注的前提下，让"数据增强管线本身"成为
#  遮挡率 / 失焦度 / 几何可见比例这三个新任务的标签来源——因为增强参数
#  （挖了多大的洞、抹了多大的sigma模糊、旋转框和画布相交了多少）都是我们
#  自己设定的，天然精确，不需要再标一遍数据。
#
#   1) visible_ratio（几何可见比例）：探针旋转框与画布矩形的精确相交面积占比。
#      train/val/test 均可计算，不依赖任何随机增强，反映"部分结构出视野"。
#   2) occlusion_ratio（合成遮挡比例）：仅在 train 阶段，在该实例真实 mask
#      区域内随机抠洞，遮挡比例 = 被抠除面积 / 实例面积。
#   3) defocus_level（合成失焦度）：仅在 train 阶段，在该实例 mask 邻域内
#      施加已知 sigma 的局部高斯模糊，defocus_level = sigma / sigma_max。
#  val/test 阶段该两项恒为 0（保持评估集"干净"，可解释为"未注入合成退化"），
#  只有 visible_ratio 在三个 split 上都有意义。
# ══════════════════════════════════════════════════════════════

def rotated_rect_frame_visible_ratio(corners: np.ndarray, img_w: int, img_h: int) -> float:
    """
    corners: (4,2) 旋转框角点（增强后，可能部分/全部在画布外，未做越界裁剪）。
    返回: 旋转框与画布矩形 [0,W]x[0,H] 的相交面积 / 旋转框自身面积 ∈ [0,1]。
    1.0 = 完全在画布内；0.0 = 完全出界（部分结构在视野之外的精确几何度量）。
    """
    corners = corners.astype(np.float32)
    box_area = float(cv2.contourArea(corners))
    if box_area <= 1e-6:
        return 0.0
    frame = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32)
    try:
        _, inter_pts = cv2.intersectConvexConvex(corners, frame)
    except cv2.error:
        inter_pts = None
    if inter_pts is None or len(inter_pts) < 3:
        return 0.0
    inter_area = float(cv2.contourArea(inter_pts.astype(np.float32)))
    return float(np.clip(inter_area / box_area, 0.0, 1.0))


def apply_synthetic_occlusion(
        image: "torch.Tensor", inst_mask: np.ndarray,
        p: float = 0.4, max_holes: int = 3,
        hole_frac_range: Tuple[float, float] = (0.05, 0.35),
) -> Tuple["torch.Tensor", float]:
    """
    仅在单个探针实例的真实分割 mask 区域内挖 1~max_holes 个矩形洞，
    只影响该实例自身像素，不触碰画面其余部分（不像 CoarseDropout 那样全图乱挖，
    否则遮挡面积和"哪个实例被挡"的对应关系就丢了，没法当监督用）。

    image:     (C,H,W) 已归一化 tensor
    inst_mask: (H,W) bool/{0,1}，该实例增强后的真实 mask（与 image 同分辨率）
    返回: (occluded_image, occlusion_ratio)；未触发遮挡时 occlusion_ratio=0.0
    """
    if not HAS_TORCH:
        return image, 0.0
    mask_bool = np.asarray(inst_mask).astype(bool)
    if random.random() > p or mask_bool.sum() < 20:
        return image, 0.0

    ys, xs = np.where(mask_bool)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    inst_area = float(mask_bool.sum())

    occ_canvas = np.zeros_like(mask_bool)
    n_holes = random.randint(1, max_holes)
    for _ in range(n_holes):
        frac = random.uniform(*hole_frac_range)
        side_h = max(int((y1 - y0 + 1) * math.sqrt(frac)), 2)
        side_w = max(int((x1 - x0 + 1) * math.sqrt(frac)), 2)
        cy = random.randint(y0, y1)
        cx = random.randint(x0, x1)
        yy0, yy1 = max(cy - side_h // 2, 0), min(cy + side_h // 2, mask_bool.shape[0])
        xx0, xx1 = max(cx - side_w // 2, 0), min(cx + side_w // 2, mask_bool.shape[1])
        occ_canvas[yy0:yy1, xx0:xx1] = True

    occ_in_mask = occ_canvas & mask_bool
    occluded_area = float(occ_in_mask.sum())
    if occluded_area < 1e-6:
        return image, 0.0

    ratio = float(np.clip(occluded_area / inst_area, 0.0, 1.0))
    occ_t = torch.from_numpy(occ_in_mask).to(image.device)
    image = image.clone()
    image[:, occ_t] = 0.0  # 归一化空间下的"中性遮挡色"（≈原图逐通道均值灰）
    return image, ratio


def apply_synthetic_defocus(
        image: "torch.Tensor", inst_mask: np.ndarray,
        p: float = 0.3, sigma_range: Tuple[float, float] = (0.8, 4.0),
        sigma_max: float = 4.0,
) -> Tuple["torch.Tensor", float]:
    """
    仅在实例 mask 的膨胀邻域内施加已知 sigma 的高斯模糊，模拟"探针针尖因景深
    不足而局部失焦、画面其余部分仍清晰"的真实场景（而不是像现有
    build_train_transforms 里 GaussNoise/ISONoise 那样，全图统一加噪声，
    那是模拟相机噪声，明确不模拟散焦——本函数专门补上这一块）。

    返回: (blurred_image, defocus_level)；defocus_level = sigma / sigma_max ∈[0,1]
    """
    if not HAS_TORCH:
        return image, 0.0
    mask_bool = np.asarray(inst_mask).astype(np.uint8)
    if random.random() > p or mask_bool.sum() < 20:
        return image, 0.0

    sigma = random.uniform(*sigma_range)
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    if ksize % 2 == 0:
        ksize += 1

    dilated = cv2.dilate(mask_bool, np.ones((15, 15), np.uint8))
    region = torch.from_numpy(dilated.astype(bool)).to(image.device)

    img_np = image.permute(1, 2, 0).cpu().numpy()
    blurred_np = cv2.GaussianBlur(img_np, (ksize, ksize), sigma)
    blurred = torch.from_numpy(blurred_np).permute(2, 0, 1).to(image.device, dtype=image.dtype)

    image = image.clone()
    region_3 = region.unsqueeze(0).expand_as(image)
    image[region_3] = blurred[region_3]

    defocus_level = float(np.clip(sigma / sigma_max, 0.0, 1.0))
    return image, defocus_level


class GSGProbeDataset(Dataset):

    def __init__(
            self,
            data_root: str,
            split: str = "train",  # 'train' | 'val' | 'test'
            transforms=None,
            use_pseudo_labels: bool = False,  # 是否使用 CV 基线生成的伪标签
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.use_pseudo = use_pseudo_labels

        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = build_train_transforms()
        elif split == "val":
            self.transforms = build_val_transforms()

        # 加载 COCO 格式标注
        ann_file = self.data_root / "splits" / f"{split}_annotations.json"
        with open(ann_file, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        # image_id → image_info
        self.images: Dict[int, dict] = {
            img["id"]: img for img in self.coco_data["images"]
        }
        # image_id → List[annotation]
        self.annotations: Dict[int, List[dict]] = self._group_annotations()
        # 有序的 image_id 列表（保证 __getitem__ 索引稳定）
        self.image_ids: List[int] = list(self.images.keys())

    # ---------------------------------------------------------------------- #
    #  标准 Dataset 接口
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        anns = self.annotations.get(img_id, [])

        # 1. 读取图像
        img_path = self.data_root / img_info["file_name"]
        image = self._load_image(img_path)
        h, w = image.shape[:2]

        # 2. 解析标注，按任务分流
        probe_items = []  # 探针实例列表
        calib_items = []  # 校准片实例列表

        for ann in anns:
            cat_id = int(ann["category_id"])
            if cat_id in PROBE_CATEGORY_IDS:
                probe_items.append(self._parse_probe_ann(ann, (h, w)))
            elif cat_id in CALIB_CATEGORY_IDS:
                calib_items.append(self._parse_calib_ann(ann, (h, w)))

        # 3. 拆分探针字段
        probe_labels = [p["label"] for p in probe_items]
        probe_bboxes = [p["bbox"] for p in probe_items]  # AABB，仅供增强管线裁剪/存活判定使用
        probe_masks = [p["mask"] for p in probe_items]
        probe_keypoints = [p["keypoints"] for p in probe_items]
        probe_visibility = [p["visibility"] for p in probe_items]
        probe_corners = [p["corners"] for p in probe_items]  # List[N_i] ndarray(4,2)

        calib_bboxes = [c["bbox"] for c in calib_items]
        calib_labels = [c["label"] for c in calib_items]
        calib_corners = [c["corners"] for c in calib_items]  # List[M_i] ndarray(4,2)

        # 4. 数据增强（统一作用于图像 + 探针掩模 + 所有 bbox + 关键点）
        transformed = self._apply_transforms(
            image=image,
            probe_labels=probe_labels,
            probe_bboxes=probe_bboxes,
            probe_masks=probe_masks,
            probe_keypoints=probe_keypoints,
            probe_visibility=probe_visibility,
            probe_corners=probe_corners,
            calib_bboxes=calib_bboxes,
            calib_labels=calib_labels,
            calib_corners=calib_corners,
        )

        assert len(transformed["probe_rboxes"]) == len(transformed["probe_masks"]) == \
               len(transformed["probe_keypoints"]) == len(transformed["probe_visibility"]), \
            "probe并行数组长度不一致，数据管道存在错位"

        # 5. 遮挡 / 失焦 —— 自监督代理标签（只新增字段，不改变已有任何字段）
        #    visible_ratio 已经在 _apply_transforms 里用"裁剪前"的原始角点精确算好了
        #    （transformed["probe_visible_ratio"]），这里只需要按 split 注入合成遮挡/失焦。
        image_aug = transformed["image"]
        n_probe_out = len(transformed["probe_rboxes"])
        probe_occlusion_ratio: List[float] = []
        probe_defocus_level: List[float] = []
        probe_visible_ratio: List[float] = list(transformed["probe_visible_ratio"])

        for i in range(n_probe_out):
            occ_ratio, defocus_level = 0.0, 0.0
            if self.split == "train":
                mask_i = transformed["probe_masks"][i]
                image_aug, occ_ratio = apply_synthetic_occlusion(image_aug, mask_i)
                image_aug, defocus_level = apply_synthetic_defocus(image_aug, mask_i)

            probe_occlusion_ratio.append(occ_ratio)
            probe_defocus_level.append(defocus_level)

        return {
            "image": image_aug,  # (C,H,W) tensor
            "image_id": img_id,
            "img_path": str(img_path),
            # 探针分支（统一旋转框表示：[cx, cy, w, h, theta_rad]，theta∈(-π/4, π/4]）
            "probe_labels": transformed["probe_labels"],  # List[int]
            "probe_rboxes": transformed["probe_rboxes"],  # List[[cx,cy,w,h,theta]]
            "probe_masks": transformed["probe_masks"],  # List[ndarray (H,W)]
            "probe_keypoints": transformed["probe_keypoints"],  # List[ndarray (16,2)]
            "probe_visibility": transformed["probe_visibility"],  # List[ndarray (16,)]
            # 探针状态分支（新增，与 probe_rboxes 逐实例对齐，长度相同）
            "probe_occlusion_ratio": probe_occlusion_ratio,  # List[float]∈[0,1]，合成遮挡比例（val/test恒0）
            "probe_defocus_level": probe_defocus_level,  # List[float]∈[0,1]，合成失焦程度（val/test恒0）
            "probe_visible_ratio": probe_visible_ratio,  # List[float]∈[0,1]，与画布几何相交比例（三个split均有效）
            # 校准片分支（同一格式：[cx, cy, w, h, theta_rad]）
            "calib_labels": transformed["calib_labels"],  # List[int]
            "calib_rboxes": transformed["calib_rboxes"],  # List[[cx,cy,w,h,theta]]
        }

    # ---------------------------------------------------------------------- #
    #  标注解析
    # ---------------------------------------------------------------------- #
    def _parse_probe_ann(self, ann: dict, img_shape: Tuple[int, int]) -> dict:
        h, w = img_shape
        # 与 calib 保持统一：探针也使用 (cx,cy),(bw,bh),theta_deg 的旋转标注
        (cx, cy), (bw, bh), theta_deg = ann["rotation"]
        theta_rad = math.radians(float(theta_deg))
        corners = self.rect_to_corners(cx, cy, bw, bh, theta_rad)
        # 裁剪角点到图像范围内（避免越界角点污染后续 AABB / 增强）
        corners[:, 0] = np.clip(corners[:, 0], 0.0, w)
        corners[:, 1] = np.clip(corners[:, 1], 0.0, h)
        # axis-aligned 包围盒，仅供 Albumentations 的 bbox 增强管线用于裁剪/存活判定
        x1, y1 = corners[:, 0].min(), corners[:, 1].min()
        x2, y2 = corners[:, 0].max(), corners[:, 1].max()
        aabb = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

        class_id = ann["category_id"]
        mask = self._decode_mask(ann, img_shape)
        kps_raw = ann["keypoints"]
        expected = NUM_PROBE_KEYPOINTS * 3

        if len(kps_raw) != expected:
            kps_raw = list(kps_raw) + [0.0] * (expected - len(kps_raw))

        kps_array = np.array(kps_raw, dtype=np.float32).reshape(NUM_PROBE_KEYPOINTS, 3)
        keypoints = kps_array[:, :2]  # (16, 2)  —— x, y 坐标
        visibility = kps_array[:, 2].astype(np.int32)  # (16,)  —— 0/1/2
        return {
            "class_id": class_id,
            "label": PROBE_CATEGORY_ID_TO_LABEL[class_id],
            "bbox": aabb,  # (x,y,w,h)：仅供增强管线裁剪/存活判定使用
            "corners": corners,  # (四个角点坐标)：增强后用于重建 cx,cy,w,h,theta
            "mask": mask,
            "keypoints": keypoints,
            "visibility": visibility,
        }

    def _parse_calib_ann(self, ann: dict, img_shape: Tuple[int, int]) -> dict:
        h, w = img_shape
        (cx, cy), (bw, bh), theta_deg = ann["rotation"]
        theta_rad = math.radians(float(theta_deg))
        corners = self.rect_to_corners(cx, cy, bw, bh, theta_rad)
        # 裁剪角点到图像范围内（避免越界角点污染后续 AABB / 增强）
        corners[:, 0] = np.clip(corners[:, 0], 0.0, w)
        corners[:, 1] = np.clip(corners[:, 1], 0.0, h)
        # axis-aligned 包围盒，仅供 Albumentations 的 bbox 增强管线用于裁剪/存活判定
        x1, y1 = corners[:, 0].min(), corners[:, 1].min()
        x2, y2 = corners[:, 0].max(), corners[:, 1].max()
        aabb = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        return {
            "class_id": int(ann["category_id"]),
            "label": CALIB_CATEGORY_ID_TO_LABEL[int(ann["category_id"])],
            "bbox": aabb,  # (x1,y1,x2,y2)
            "corners": corners,  # (四个角点坐标)
        }

    # ---------------------------------------------------------------------- #
    #  数据增强
    # ---------------------------------------------------------------------- #

    def _apply_transforms(
            self, image, probe_labels, probe_bboxes, probe_masks,
            probe_keypoints, probe_visibility, probe_corners,
            calib_bboxes, calib_labels, calib_corners,
    ) -> dict:
        n_probe = len(probe_bboxes)
        n_calib = len(calib_bboxes)

        # ── probe 关键点展平 ──────────────────────────────
        flat_kps: List[Tuple] = []
        for kps in probe_keypoints:
            for x, y in kps:
                flat_kps.append((float(x), float(y)))
        flat_vis_scalar = [int(v) for vis in probe_visibility for v in vis]

        # ── probe 角点展平，紧跟在 probe 关键点之后 ────────────
        n_probe_kp_total = len(flat_kps)
        for corners in probe_corners:
            for x, y in corners:
                flat_kps.append((float(x), float(y)))
        flat_vis_scalar += [2] * (n_probe * 4)

        # ── calib 角点展平，拼在 probe 关键点+角点之后 ────────────
        n_probe_kp_and_corner_total = len(flat_kps)
        for corners in calib_corners:
            for x, y in corners:
                flat_kps.append((float(x), float(y)))
        flat_vis_scalar += [2] * (n_calib * 4)

        # ── bbox 合并（仍用 AABB 做裁剪/存活判定）──────────────────
        PROBE_FLAG, CALIB_FLAG = 0, 1
        TRACK_BASE = 100_000
        bbox_track_id = (
                [PROBE_FLAG * TRACK_BASE + i for i in range(n_probe)] +
                [CALIB_FLAG * TRACK_BASE + i for i in range(n_calib)]
        )
        all_bboxes = probe_bboxes + calib_bboxes
        all_cat_ids = probe_labels + calib_labels

        result = self.transforms(
            image=image, bboxes=all_bboxes, category_ids=all_cat_ids,
            bbox_track_id=bbox_track_id,
            keypoints=flat_kps, kp_visibility=flat_vis_scalar,
            masks=probe_masks,
        )

        aug_bboxes, aug_cat_ids, aug_track_ids = result["bboxes"], result["category_ids"], result["bbox_track_id"]

        out_probe_labels, surviving_probe_idx = [], []
        out_calib_labels, surviving_calib_idx = [], []
        for bbox, cid, tid in zip(aug_bboxes, aug_cat_ids, aug_track_ids):
            if tid < TRACK_BASE:
                out_probe_labels.append(cid)
                surviving_probe_idx.append(int(tid % TRACK_BASE))
            else:
                out_calib_labels.append(cid)
                surviving_calib_idx.append(int(tid % TRACK_BASE))

        # ── probe masks / keypoints（不变逻辑）────────────────────
        all_masks = result["masks"]
        out_probe_masks = [all_masks[i] for i in surviving_probe_idx]
        H, W = result["image"].shape[1:3]
        all_probe_kp, all_probe_vis = [], []
        kp_idx = 0
        for inst_idx in range(n_probe):
            inst_vis, inst_kp = [], []
            for k in range(NUM_PROBE_KEYPOINTS):
                x, y = result["keypoints"][kp_idx]
                v = result["kp_visibility"][kp_idx]
                if not (0 <= x < W and 0 <= y < H):
                    v = 0
                inst_vis.append(v);
                inst_kp.append((x, y))
                kp_idx += 1
            all_probe_kp.append(np.array(inst_kp, dtype=np.int32))
            all_probe_vis.append(np.array(inst_vis, dtype=np.int32))
        out_probe_keypoints = [all_probe_kp[i] for i in surviving_probe_idx]
        out_probe_visibility = [all_probe_vis[i] for i in surviving_probe_idx]

        # ── probe 角点 → 重新拟合规范化旋转框（与 calib 同一套逻辑）─────
        # [新增] visible_ratio 必须用"裁剪前"的原始角点算——裁剪后的角点已经被
        # np.clip 强行摁回画布内，此时再算"旋转框与画布的相交比例"永远≈1，
        # 量不出"部分结构出视野"这件事。所以这里先用未裁剪的 raw_pts 算出
        # 精确的几何可见比例，再执行原有的裁剪逻辑（裁剪后的角点仍然是
        # out_probe_rboxes 的输入，用于回归目标/RoI，这部分行为不变）。
        all_probe_corners = []
        all_probe_visible_ratio = []
        base_p = n_probe_kp_total
        for inst_idx in range(n_probe):
            raw_pts = np.array(
                [result["keypoints"][base_p + inst_idx * 4 + k] for k in range(4)],
                dtype=np.float64,
            )
            all_probe_visible_ratio.append(
                rotated_rect_frame_visible_ratio(raw_pts, W, H)
            )
            pts = raw_pts.copy()
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            all_probe_corners.append(pts)

        out_probe_rboxes = []
        for i in surviving_probe_idx:
            cx, cy, bw, bh, theta = self.corners_to_canonical_rect(all_probe_corners[i])
            out_probe_rboxes.append([cx, cy, bw, bh, theta])  # theta 单位：弧度，∈(-π/4, π/4]
        out_probe_visible_ratio = [all_probe_visible_ratio[i] for i in surviving_probe_idx]

        # ── calib 角点 → 重新拟合规范化旋转框 ─────────────────
        all_calib_corners = []
        base_c = n_probe_kp_and_corner_total
        for inst_idx in range(n_calib):
            pts = np.array(
                [result["keypoints"][base_c + inst_idx * 4 + k] for k in range(4)],
                dtype=np.float64,
            )
            # 裁剪x坐标和y坐标
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            all_calib_corners.append(pts)

        out_calib_rboxes = []
        for i in surviving_calib_idx:
            cx, cy, bw, bh, theta = self.corners_to_canonical_rect(all_calib_corners[i])
            out_calib_rboxes.append([cx, cy, bw, bh, theta])  # theta 单位：弧度，∈(-π/4, π/4]

        return {
            "image": result["image"],
            "probe_labels": out_probe_labels,
            "probe_rboxes": out_probe_rboxes,
            "probe_masks": out_probe_masks,
            "probe_keypoints": out_probe_keypoints,
            "probe_visibility": out_probe_visibility,
            "probe_visible_ratio": out_probe_visible_ratio,  # 新增：与画布的几何相交比例，用裁剪前角点算
            "calib_labels": out_calib_labels,
            "calib_rboxes": out_calib_rboxes,
        }

    # ---------------------------------------------------------------------- #
    #  工具方法
    # ---------------------------------------------------------------------- #
    def _load_image(self, img_path: Path) -> np.ndarray:
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _clip_bbox(bbox: List[float], img_w: int = 1, img_h: int = 1) -> List[float]:
        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h
        # 先裁剪角点坐标到合法范围
        x1c = float(np.clip(x, 0.0, img_w))
        y1c = float(np.clip(y, 0.0, img_h))
        x2c = float(np.clip(x2, 0.0, img_w))
        y2c = float(np.clip(y2, 0.0, img_h))
        # 重新组装为 [x, y, w, h]，并保证 w, h >= 0（裁剪后可能出现退化框）
        new_w = max(x2c - x1c, 0.0)
        new_h = max(y2c - y1c, 0.0)

        return [x1c, y1c, new_w, new_h]

    @staticmethod
    def _decode_mask(ann: dict, shape: Tuple[int, int]) -> np.ndarray:
        """将 COCO segmentation（polygon 或 RLE）解码为二值掩模 (H, W) uint8。"""
        h, w = shape
        seg = ann.get("segmentation")
        if not seg:
            return np.zeros((h, w), dtype=np.uint8)

        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        return mask.astype(np.uint8)

    def cat_id_to_name(self, cat_id: int) -> str:
        """category_id → category_name，未知 id 返回 'unknown'。"""
        return self.ID_TO_CATEGORY.get(cat_id, "unknown")

    @staticmethod
    def rect_to_corners(cx: float, cy: float, w: float, h: float, theta_rad: float) -> np.ndarray:
        """旋转矩形 → 4个角点 (4,2)，自定义旋转矩阵实现，不依赖 cv2.boxPoints 的版本相关角度约定。"""
        c, s = math.cos(theta_rad), math.sin(theta_rad)
        dx, dy = w / 2.0, h / 2.0
        local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float64)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        return local @ R.T + np.array([cx, cy], dtype=np.float64)

    @staticmethod
    def corners_to_canonical_rect(corners: np.ndarray) -> Tuple[float, float, float, float, float]:

        pts = corners.astype(np.float64)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        e0 = pts[1] - pts[0]
        e1 = pts[2] - pts[1]
        len0, len1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
        if len0 >= len1:
            w, h = len0, len1
            angle = math.atan2(e0[1], e0[0])
        else:
            w, h = len1, len0
            angle = math.atan2(e1[1], e1[0])
        # 利用 90° 周期性把角度规范化到 (-π/4, π/4]
        half_pi = math.pi / 2.0
        quarter_pi = math.pi / 4.0
        while angle <= -quarter_pi:
            angle += half_pi
            w, h = h, w
        while angle > quarter_pi:
            angle -= half_pi
            w, h = h, w
        return cx, cy, max(w, 1e-3), max(h, 1e-3), angle

    # ---------------------------------------------------------------------- #
    #  标注聚合
    # ---------------------------------------------------------------------- #

    def _group_annotations(self) -> Dict[int, List[dict]]:
        """按 image_id 聚合全部标注。"""
        grouped: Dict[int, List[dict]] = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            # 新建key，并将ann赋值给value
            grouped.setdefault(img_id, []).append(ann)
        return grouped


# ══════════════════════════════════════════════════════════════
# 数据集划分工具
# ══════════════════════════════════════════════════════════════

def split_dataset(
        all_annotations_path: str,
        output_dir: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by: str = 'calib_type',  # 按校准片类型分层划分
        seed: int = 42,
):
    random.seed(seed)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    with open(all_annotations_path) as f:
        data = json.load(f)

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        random.shuffle(images)

        # -------------------------
        #  按比例切分
        # -------------------------
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # -------------------------
        #  根据 image_id 过滤 annotations
        # -------------------------
        def filter_annotations(images_subset):
            image_ids = set(img["id"] for img in images_subset)
            return [ann for ann in annotations if ann["image_id"] in image_ids]

        train_annotations = filter_annotations(train_images)
        val_annotations = filter_annotations(val_images)
        test_annotations = filter_annotations(test_images)

        # -------------------------
        #  构建子集
        # -------------------------
        def build_subset(images_subset, annotations_subset):
            return {
                "images": images_subset,
                "annotations": annotations_subset,
                "categories": data.get("categories", [])
            }

        train_data = build_subset(train_images, train_annotations)
        val_data = build_subset(val_images, val_annotations)
        test_data = build_subset(test_images, test_annotations)

        # -------------------------
        #  保存
        # -------------------------
        os.makedirs(output_dir, exist_ok=True)
        # 地址
        with open(os.path.join(output_dir, "train_annotations.json"), "w") as f:
            json.dump(train_data, f)

        with open(os.path.join(output_dir, "val_annotations.json"), "w") as f:
            json.dump(val_data, f)

        with open(os.path.join(output_dir, "test_annotations.json"), "w") as f:
            json.dump(test_data, f)
