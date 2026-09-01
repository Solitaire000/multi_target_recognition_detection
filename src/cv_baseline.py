"""
cv_baseline.py
==============
第一阶段：CV为主的基线系统。
作用：
  1. 快速验证可行性，建立性能基准
  2. 为ML阶段自动生成伪标签（pseudo-label）
  3. 在ML模型置信度低时作为fallback保障

四个子模块：
  A. 探针针尖检测（形态学 + 关键点几何推算）
  B. 针尖关键点精化（基于针尖矩形几何定义）
  C. 校准片识别（颜色分割 + 形状先验补全）
  D. 针痕检测（亮点聚类 + 物理约束降维）
"""

from __future__ import annotations
import colorsys
from pathlib import Path
from pprint import pprint
from typing import Dict

from sklearn.cluster import DBSCAN
from collections import defaultdict
from definitions import *

import json
from tqdm import tqdm
from calculateFun import *
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from universalFun import *
import os


# _  : 内部函数的标志

# ══════════════════════════════════════════════════════════════
# 探针检测：暗区分割 + 形态学分析
# ══════════════════════════════════════════════════════════════


class ProbeDetectorCV:
    """
    基于CV的探针针尖检测器。
    
    流程：
      1. HSV/Lab颜色空间分析→暗区掩模
      2. 连通域分析→候选探针区域
      3. 最小外接旋转矩形→ProbeTipMask
      4. 几何关键点推算→ProbeKeyPoints
      5. 磨损评分（长宽比偏差）
    """

    def __init__(
            self,
            nominal_aspect_ratio: float = 3.0,
            min_tip_area_px: int = 200,
            dark_v_threshold: int = 80,
            px_per_um: float = 1.0,
    ):
        self.nominal_ar = nominal_aspect_ratio
        self.min_area = min_tip_area_px
        self.dark_thresh = dark_v_threshold
        self.px_per_um = px_per_um

    def detect(self, image: np.ndarray):
        """
        主入口。输入BGR图像，返回针尖掩模列表和关键点。
        """
        # 高斯去噪声
        b, g, r = cv2.split(image)
        b_filtered = cv2.GaussianBlur(b, (5, 5), 0)
        g_filtered = cv2.GaussianBlur(g, (5, 5), 0)
        r_filtered = cv2.GaussianBlur(r, (5, 5), 0)
        filtered_img = cv2.merge([b_filtered, g_filtered, r_filtered])

        # 步骤1：提取暗区掩模
        dark_mask = self._extract_dark_mask(filtered_img)
        # 二值图像反转
        dark_mask = cv2.bitwise_not(dark_mask)
        # 加载模板
        current_path = Path(__file__).resolve()
        template = cv2.imread(current_path.parent.parent / "data/template/binary_tipTemplate.png")
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        ''' 制作模板 '''
        # cv2.imwrite(current_path.parent.parent/"data/template/binary_tipTemplate.png", dark_mask)
        # showImage("dark_mask", template)

        # 步骤2：形态学清理
        # 开运算，先腐蚀后膨胀，去噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=4)
        # 闭运算，先膨胀后腐蚀，填空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        # 步骤3：连通域分析，找候选针尖区域
        # 几何模板匹配
        # 通常是一组探针
        rect, keyPoints = self._geometric_match(dark_mask, template)
        '''
        num_labels   # 连通域总数（包含背景0）
        labels       # 和原图一样大的矩阵，每个像素值=它属于哪个连通域
        stats        # 形状 [num_labels, 5]
                     # stats[label, 0] = x
                     # stats[label, 1] = y
                     # stats[label, 2] = width
                     # stats[label, 3] = height
                     # stats[label, 4] = area(面积)
        centroids    # 形状 [num_labels, 2] → 每个连通域的中心 (cx, cy)
        '''

        if not rect or not keyPoints:
            return [], []
        result_keyPoints = ProbeKeyPoints(keyPoints=keyPoints)
        result_probe = ProbeTipMask(label=ProbeLabel.PROBE_GSG, tip_rect=rect)
        return [result_keyPoints], [result_probe]

    def _geometric_match(self,
                         target,
                         template,
                         scales: Tuple[float, float, float] = (0.8, 1.2, 0.1),
                         brightness_factors=(0.8, 1.0, 1.2),
                         use_invert: bool = True,
                         threshold: float = 0.5):
        import cv2
        import numpy as np
        ''' 不动 '''
        pts = np.array([
            [145, 256], [12, 247], [12, 217], [39, 217],
            [66, 192], [133, 188], [135, 153], [10, 150],
            [10, 125], [135, 121], [135, 88], [75, 85],
            [40, 55], [13, 52], [15, 27], [152, 20]
        ], dtype=np.float32)
        template_pts = pts.reshape(-1,1,2)
        # 可视化验证点集
        # x_coords = [p[0] for p in pts]
        # y_coords = [p[1] for p in pts]
        # plt.figure(figsize=(10, 8))
        # plt.scatter(x_coords, y_coords, color='red', s=100, label='Points')  # s 控制点的大小
        # plt.show()
        rect = []
        keypoints =[]
        ''' 不动 '''

        ''' 关键点映射 '''
        # ===== 1. 特征检测器 =====
        use_sift = True
        if use_sift:
            detector = cv2.SIFT_create()
            norm = cv2.NORM_L2
        else:
            detector = cv2.ORB_create(5000)
            norm = cv2.NORM_HAMMING

        bf = cv2.BFMatcher(norm)

        best_H = None
        best_inliers = 0
        best_mapped = None
        h, w = template.shape[:2]

        def to_gray(img):
            return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        def adjust_brightness(img, factor):
            out = np.clip(img.astype(np.float32) * factor, 0, 255)
            return out.astype(np.uint8)

        scene_gray = to_gray(target)
        template_gray = to_gray(template)

        # ───── 镜像 ─────
        mirror_variants = {
            "原始": template_gray,
            "水平镜像": cv2.flip(template_gray, 1),
            "垂直镜像": cv2.flip(template_gray, 0),
        }

        # ───── 亮度变化 ─────
        def gen_brightness(img):
            variants = []
            for f in brightness_factors:
                variants.append((f"亮度×{f}", adjust_brightness(img, f)))
            if use_invert:
                variants.append(("反转", cv2.bitwise_not(img)))
            return variants

        scale_min, scale_max, scale_step = scales

        best = None

        for mirror_name, base in mirror_variants.items():
            for bright_name, bright_img in gen_brightness(base):
                scale = scale_min
                while scale <= scale_max:
                    w = int(bright_img.shape[1] * scale)
                    h = int(bright_img.shape[0] * scale)

                    if w < 5 or h < 5:
                        scale += scale_step
                        continue

                    tmpl = cv2.resize(bright_img, (w, h))

                    if tmpl.shape[0] > scene_gray.shape[0] or tmpl.shape[1] > scene_gray.shape[1]:
                        scale += scale_step
                        continue

                    res = cv2.matchTemplate(scene_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)

                    if best is None or max_val > best["score"]:
                        best = {
                            "score": float(max_val),
                            "location": max_loc,
                            "size": (w, h),
                            "transform": f"{mirror_name} + {bright_name} + scale={scale:.2f}"
                        }

                    scale += scale_step

        # print(f"score {best['score']}")
        if not best or best["score"] < threshold:
            return [],[]

        x, y = best["location"]
        w, h = best["size"]
        # 原始模板尺寸
        th, tw = template_gray.shape[:2]
        if template_pts.ndim == 3:
            pts = template_pts.reshape(-1, 2)
        else:
            pts = template_pts.copy()

        # 缩放比例
        scale_x = w / tw
        scale_y = h / th

        # 是否水平镜像
        transform_str = best.get("transform", "")
        is_flip_x = "水平镜像" in transform_str

        mapped_pts = []

        for (px, py) in pts:
            # ===== 1. 镜像（先做！）=====
            if is_flip_x:
                px = tw - 1 - px

            # ===== 2. 缩放 =====
            px_scaled = px * scale_x
            py_scaled = py * scale_y

            # ===== 3. 平移 =====
            mx = int(round(px_scaled + x))
            my = int(round(py_scaled + y))

            mapped_pts.append([mx, my])

        ''' 不动 '''
        # 外接矩形
        mapped_pts = np.asarray(mapped_pts, dtype=np.float32)
        rect = cv2.minAreaRect(mapped_pts)
        for x,y in mapped_pts:
            # print(f"[{x},{y}]")
            keypoint = KeyPoint(x = x,y = y)
            keypoints.append(keypoint)
        return rect, keypoints

    def _extract_dark_mask(self, image: np.ndarray) -> np.ndarray:
        """
        提取图像中的暗色探针区域。
        策略：HSV V通道自适应阈值（Otsu）+ Lab L通道辅助验证。
        对不同背景颜色鲁棒（白/黑/绿/纹理）。
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        # Otsu自适应阈值（对光照变化鲁棒）
        threshold, mask_otsu = cv2.threshold(
            v_channel,
            0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return mask_otsu


# ══════════════════════════════════════════════════════════════
# 校准片检测：颜色分割 + 几何先验补全
# ══════════════════════════════════════════════════════════════
CFG = dict(
    thresh_binary=70,  # global binarization threshold
    min_bar_area=500,  # min pixel area to keep
    min_ar=1.2,  # min aspect ratio (h/w) for a bar
    max_bar_width=80,  # px — reject fat blobs
    group_gap_frac=0.25,  # x-gap > this * image_width → new group
    group_y_gap_frac=0.5,  # x-gap > this * image_width → new group
    seg_bright_thr=80,  # intensity threshold for internal segments
    seg_min_height=5,  # min px height for a sub-segment to count
)


class CalibDetectorCV:
    def __init__(self, px_per_um: float = 1.0):
        self.px_per_um = px_per_um
        # 标定量（需实测）
        self.nominal_pad_height_px = 80  # 焊盘标称高度（像素@标准放大倍率）
        self.nominal_pad_spacing_px = 60  # 焊盘间距（像素）

    def detect(self, image: np.ndarray):
        img_bgr, img_rgb, gray, enhanced, binary, cleaned = self._load_and_preprocess(image)
        H, W = img_bgr.shape[:2]
        vis = image.copy()
        # 检测条状物
        bars = self._detect_bars(cleaned, gray)
        if not bars:
            return []
        # 检测校准件，将每个校准件的bar，存入同一个数组中，多个校准件组成一个多维数组
        groups = self._group_bars(bars, W, H)
        # 补充bar特征
        for bar in bars:
            """计算其他特征"""
            x, y, w, h = bar['bbox']
            roi = image[y:y + h, x:x + w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # 取 V 通道
            v = hsv[:, :, 2]
            # 阈值分割（暗区域）
            _, mask = cv2.threshold(v, 80, 255, cv2.THRESH_BINARY_INV)
            profile = mask.mean(axis=0)
            n_segs, segs = self._count_sub_segments(profile)
            color = self._extract_color_features(img_bgr, bar['bbox'])
            bar.update(dict(profile=profile, n_segs=n_segs, segs=segs, **color, ))

        # 校准件分类
        result_calibs = []
        for gr in groups:
            all_pts = []
            for bar in gr:
                x, y, w, h = bar['bbox']
                # 每个小框的4个角点
                pts = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]
                all_pts.extend(pts)
            # 转 numpy
            pts = np.array(all_pts, dtype=np.float32)
            # 计算旋转最小外接矩形
            rect = cv2.minAreaRect(pts)
            xx, yy, ww, hh = cv2.boundingRect(pts)
            # 直接裁剪ROI
            roi = cleaned[yy:yy + hh, xx:xx + ww]
            roi = cv2.resize(roi, (40, 200))
            proj = np.mean(roi, axis=1)
            mid = np.mean(proj[80:120])
            global_mean = np.mean(proj)
            ratio = mid / global_mean
            label = self._classify_group(gr, ratio)
            result_calib = CalibPad(calib_type=label, pad_rects=rect)

            result_calibs.append(result_calib)
        # 返回标签 result = [[gr['label'],rect][gr['label'],rect]]
        return result_calibs

    def _load_and_preprocess(self, img_bgr):
        """Load image, denoise, CLAHE-enhance, threshold."""
        if img_bgr is None:
            return []
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 1. 转灰度
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # showImage("gray", gray)
        # 2. 提取黑色线条（自适应更稳）
        _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)
        # showImage("mask", mask)
        # 3. 形态学去细线（开运算）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # showImage("mask2",mask)
        # 再膨胀一点，让线条mask更完整
        mask = cv2.dilate(mask, kernel, iterations=1)
        # showImage("mask2", mask)
        # 4. inpaint（直接对灰度图）
        inpainted = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
        # showImage("inpainted",inpainted)

        # 高斯模糊
        denoised = cv2.GaussianBlur(inpainted, (3, 3), 0)
        # 对比度受限的自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 全局阈值分割
        _, binary = cv2.threshold(enhanced, CFG['thresh_binary'], 255, cv2.THRESH_BINARY)

        k3 = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=4)
        k5v = np.ones((5, 1), np.uint8)  # vertical kernel — preserve tall bars
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k5v, iterations=3)
        # showImage("cleaned",cleaned)

        # 距离变换
        dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
        # 找“细线区域”（距离小的地方）
        thin_mask = dist < 4
        # 全局膨胀
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # 只替换细线区域
        result = binary.copy()
        result[thin_mask] = dilated[thin_mask]

        k5v = np.ones((3, 1), np.uint8)  # vertical kernel — preserve tall bars
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k5v, iterations=4)
        # showImage("result",result)
        return img_bgr, img_rgb, gray, enhanced, binary, result

    def _detect_bars(self, cleaned, gray):

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
        bars = []
        for i in range(1, num_labels):  # 0 = background
            x, y, w, h, area = stats[i]
            # ===== 基础过滤 =====
            if area < CFG['min_bar_area']:
                continue
            ar = h / max(w, 1)
            if ar < CFG['min_ar']:
                continue
            # ===== 填充率 =====
            rect_area = w * h
            fill_ratio = area / rect_area
            if fill_ratio < 0.4:  # 去掉不规则形状
                continue
            cx, cy = centroids[i]
            roi = gray[y:y + h, x:x + w]
            roi_binary = cleaned[y:y + h, x:x + w]
            bars.append(dict(
                label=i,
                bbox=(x, y, w, h),
                area=area,
                ar=ar,
                fill=fill_ratio,
                cx=int(cx),
                cy=int(cy),
                roi=roi,
                roi_binary=roi_binary,
            ))

        #  排序（推荐还是按 cx）
        bars.sort(key=lambda b: b['cx'])
        return bars

    def _group_bars(self, bars, img_width, img_heigh):
        if not bars:
            return []

        groups = []
        current = [bars[0]]
        gap_thresh = img_width * CFG['group_gap_frac']
        # gap_y_thresh = img_heigh * CFG['group_y_gap_frac']
        for bar in bars[1:]:
            # 负索引，最后一个元素
            prev = current[-1]
            gap = bar['cx'] - prev['cx']
            if gap > gap_thresh:
                groups.append(current)
                current = []
            # current
            current.append(bar)
        # 校准件 行分组
        groups.append(current)

        return groups

    def _count_sub_segments(self, profile, thr=None, min_h=None):
        """
        统计条形图强度分布中不同亮子段的数量。
        """

        thr = thr or CFG['seg_bright_thr']
        min_h = min_h or CFG['seg_min_height']

        bright = profile > thr
        segs, in_seg, seg_start = [], False, 0
        for i, v in enumerate(bright):
            # in_seg 记录上一个子段的亮暗属性
            if v and not in_seg:
                in_seg, seg_start = True, i
            elif not v and in_seg:
                in_seg = False
                h = i - seg_start
                # 筛选长度
                if h >= min_h:
                    segs.append((seg_start, i - 1, h))

        if in_seg:
            h = len(profile) - seg_start
            if h >= min_h:
                segs.append((seg_start, len(profile) - 1, h))

        return len(segs), segs

    def _extract_color_features(self, img_bgr, bbox):
        """Mean BGR, dominant hue, saturation in ROI."""
        x, y, w, h = bbox
        roi_bgr = img_bgr[y:y + h, x:x + w]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        mean_bgr = roi_bgr.mean(axis=(0, 1))
        mean_hsv = roi_hsv.mean(axis=(0, 1))

        return dict(
            mean_b=mean_bgr[0], mean_g=mean_bgr[1], mean_r=mean_bgr[2],
            mean_h=mean_hsv[0], mean_s=mean_hsv[1], mean_v=mean_hsv[2],
        )

    def _classify_group(self, group_bars, ratio):
        n = len(group_bars)
        # --- THRU ---
        if n == 1:
            return CalibLabel.THRU

        # --- LOAD/SHORT ---
        if 1 < n < 4:
            ars = []
            for bar in group_bars:
                ars.append(bar['ar'])
            max_ar = max(ars)
            if max_ar < 5:
                return CalibLabel.THRU

            if ratio >= 0.85:
                return CalibLabel.SHORT
            else:
                return CalibLabel.LOAD
        # --- OPEN ---
        if n >= 4:
            return CalibLabel.OPEN

        return CalibLabel.UNKNOW

    def _compute_gradient(self, img):
        # Sobel梯度
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

        magnitude = cv2.magnitude(grad_x, grad_y)

        # 归一化到0-255
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return magnitude.astype(np.uint8)


# ══════════════════════════════════════════════════════════════
# 针痕检测：物理约束降维 + DBSCAN聚类
# ══════════════════════════════════════════════════════════════

class ScrubDetectorCV:
    def __init__(
            self,
            bright_threshold: int = 200,  # 亮点检测阈值
            strip_width_px: int = 40,  # 垂直搜索条带宽度
            dbscan_eps: float = 15.0,  # DBSCAN邻域半径（像素）
            dbscan_min_samples: int = 3,
    ):
        self.bright_thresh = bright_threshold
        self.strip_width = strip_width_px
        self.eps = dbscan_eps
        self.min_samples = dbscan_min_samples

    def detect(
            self,
            image: np.ndarray,
    ):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 全局亮点检测（LoG blob detection）
        binary_spots = self._detect_bright_blobs(gray)
        spots_to_needleMarks = self._spots_to_needleMark(binary_spots)
        if len(spots_to_needleMarks) == 0:
            return []

        # grouped,ungrouped [[id,id,id],[id,id,id],[id,id,id]]
        grouped, ungrouped, centroids = self._needleMark_grouped(spots_to_needleMarks)
        if grouped is []:
            return []

        # 点集可视化
        vis = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        # 唯一的随机颜色
        np.random.seed(42)  # 固定种子，每次运行颜色都一样
        unique_colors = []
        for i in range(len(spots_to_needleMarks) + 1):
            hue = i / len(spots_to_needleMarks) + 1  # 均匀分布
            lightness = 0.6
            saturation = 0.9
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            unique_colors.append((int(b * 255), int(g * 255), int(r * 255)))
        for index, pos in spots_to_needleMarks.items():
            color = unique_colors[index]
            for (x, y) in pos:
                cv2.circle(vis, (x, y), 2, color, -1)
        # 针痕组
        points = []
        result_scrubs = []
        j = 0
        for group in grouped:
            i = 0
            result_marks = []
            for key in group:
                points.extend(spots_to_needleMarks[key])
                pts_scrub = spots_to_needleMarks[key]
                # 计算scrubMark参数
                x_min, y_min = np.min(pts_scrub, axis=0)
                x_max, y_max = np.max(pts_scrub, axis=0)

                bbox = (x_min, y_min, x_max, y_max)
                # 二值掩膜
                mask = binary_spots[y_min:y_max, x_min:x_max]
                result_mark = ScrubMark(label=i, mask=mask, centroid=centroids[key], group_id=j, bbox=bbox)
                result_marks.append(result_mark)
                i = i + 1
            result_scrub = ScrubGroup(group_id=j, marks=result_marks)
            j = j + 1
            points = []
            result_scrubs.append(result_scrub)
        # 返回标签结果
        return result_scrubs

    def _detect_bright_blobs(
            self,
            gray: np.ndarray,
    ) -> np.ndarray:
        """
        使用LoG (Laplacian of Gaussian) 检测亮斑。
        排除探针暗区内的亮点（镜面反射干扰）。
        返回亮点坐标数组 (N, 2)。
        """
        # LoG实现：DoG近似（效率高）
        g1 = cv2.GaussianBlur(gray, (1, 1), 0.1)
        g2 = cv2.GaussianBlur(gray, (15, 15), 5.0)
        log_response = (g1.astype(np.float32) - g2.astype(np.float32))
        # 阈值
        bright_mask = (log_response > 0.5).astype(np.uint8) * 255
        bright_mask &= (gray > self.bright_thresh).astype(np.uint8) * 255
        # 利用圆形核运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
        filled = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        # 拼接坐标
        return filled

    def _spots_to_needleMark(self, binary_image: np.ndarray):

        num_labels, label_ids, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        # 计算联通域的中心
        centroids_arr = []
        index = []
        for label_id in range(1, num_labels):  # 从1开始，跳过背景
            cx, cy = centroids[label_id]  # 中心点坐标
            # 过滤面积非常小的连通域
            area = stats[label_id, 4]
            if area >= 100:
                index.append(label_id)
                centroids_arr.append((cx, cy))
        # 利用联通域中心聚类
        centroids_arr = [(float(x), float(y)) for x, y in centroids_arr]
        if centroids_arr == []:
            return []
        clusterer = DBSCAN(eps=2, min_samples=1).fit(centroids_arr)
        labels = clusterer.labels_

        # 输出针痕的所有点集
        # 初始化一个特殊的dictW
        needleMarks = defaultdict(list)
        for i in range(0, len(centroids_arr)):
            points = np.argwhere(label_ids == index[i])[:, [1, 0]]  # 返回坐标
            needleMarks[labels[i]].extend(points)

        return needleMarks

    def _needleMark_grouped(self, spots_to_needleMarks):
        groups, ungrouped, centroids = group_vertical_triplets(spots_to_needleMarks)
        return groups, ungrouped, centroids


# ══════════════════════════════════════════════════════════════
# CV基线主流程
# ══════════════════════════════════════════════════════════════

class CVBaseline:
    """
    CV基线系统主入口类
    功能：集成 探针检测、校准片检测、针痕检测 三个子模块
    输出：单帧图像的完整检测结果（位置偏差、调平角、掩模、关键点等）
    """

    def __init__(self, config: Dict):
        """
        构造函数：初始化三个检测器 + 读取配置
        :param config: 参数字典（长宽比、面积阈值、像素比例尺等）
        """
        # 1. 初始化探针检测器（检测探针尖端、关键点）
        self.probe_detector = ProbeDetectorCV(
            nominal_aspect_ratio=config.get('nominal_ar', 3.0),  # 标称长宽比
            min_tip_area_px=config.get('min_tip_area', 200),  # 最小针尖面积（像素）
            dark_v_threshold=config.get('dark_thresh', 80),  # 暗区阈值
            px_per_um=config.get('px_per_um', 1.0),  # 像素/微米 比例尺
        )
        # 2. 初始化器校准片检测（用于定位目标中心）
        self.calib_detector = CalibDetectorCV(px_per_um=config.get('px_per_um', 1.0))
        # 3. 初始化针痕（scrub）检测器（检测针迹、不对称度）
        self.scrub_detector = ScrubDetectorCV(
            bright_threshold=config.get('bright_thresh', 200),  # 亮区阈值
            dbscan_eps=config.get('dbscan_eps', 15.0),  # 聚类半径
        )

        # 全局比例尺
        self.px_per_um = config.get('px_per_um', 1.0)

    def run(self, image: np.ndarray, probe: bool = True, calib: bool = True, scrub: bool = True) -> FrameResult:
        """
        单帧图像完整推理 pipeline（核心入口函数）
        :param scrub:
        :param calib:
        :param probe:
        :param image: 输入图像 numpy 数组
        :return: 完整的帧结果对象 FrameResult
        """
        import time
        t0 = time.time()  # 记录推理开始时间
        # 初始化结果对象
        result = FrameResult(px_per_um=self.px_per_um)
        # ==================== A. 探针检测 ====================
        # 输出：探针尖端掩模 + 探针、关键点（中心角度等）
        if probe:
            result.probe_keypoints, result.probe_masks = self.probe_detector.detect(image)
        # ==================== B. 校准片检测 ====================
        # 输出：校准片pad列表（目标位置）
        if calib:
            result.calib_pads = self.calib_detector.detect(image)
        # ==================== C. 针痕检测 ====================
        # 检测针痕（需要探针关键点+掩模辅助）
        # if scrub:
        #     result.scrub_groups = self.scrub_detector.detect(image)

        # 可视化
        vis = image.copy()
        # 探针关键点标签可视化
        if result.probe_keypoints is not []:
            for kPs in result.probe_keypoints:
                for i,kP in enumerate(kPs.keyPoints):
                    [x,y]= kP.as_array()
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)  # 红色点
                    cv2.putText(vis, str(i+1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,
                                cv2.LINE_AA)
        # 探针针尖标签可视化
        if result.probe_masks is not []:
            for pm in result.probe_masks:
                # 转成4个角点
                box = cv2.boxPoints(pm.tip_rect)  # shape: (4, 2)
                box = np.int32(box)
                # 在原图上画出来
                cv2.polylines(vis, [box], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.putText(vis, str(pm.label.value), (box[2][0], box[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                            cv2.LINE_AA)
        # 针痕标签可视化
        if result.scrub_groups is not []:
            for sg in result.scrub_groups:
                for s in sg.marks:
                    x1, y1, x2, y2 = s.bbox
                    # 构造矩形顶点（顺时针或逆时针顺序）
                    points = np.array([
                        [x1, y1],  # 左上
                        [x2, y1],  # 右上
                        [x2, y2],  # 右下
                        [x1, y2]  # 左下
                    ], dtype=np.int32)
                    cv2.polylines(vis, [points], 2, (0, 255, 0), -1)
                    label = f"{s.group_id}:{s.label.value}"
                    cv2.putText(vis, label, (s.bbox[0], s.bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                                cv2.LINE_AA)
        # 校准片标签可视化
        if result.calib_pads is not []:
            for cp in result.calib_pads:
                # 转成4个角点
                box = cv2.boxPoints(cp.pad_rects)  # shape: (4, 2)
                box = np.int32(box)
                # 在原图上画出来
                cv2.polylines(vis, [box], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.putText(vis, str(cp.calib_type.value), (box[2][0], box[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                            2,
                            cv2.LINE_AA)
        # 记录推理耗时（毫秒）
        result.inference_time_ms = (time.time() - t0) * 1000
        # showImage("vis",vis)
        return result

    def generate_labels(self, im_dir, save_path):
        """
        生成高质量伪标签（给机器学习模型训练用）
        """
        # coco标注
        im_list = os.listdir(im_dir)
        # 按照名称排序
        im_list.sort(key=lambda x: int(x.split('.')[0]))
        idx = 0
        image_id = -1
        images = []
        annotations = []
        #  进度条
        for im_path in tqdm(im_list):
            image_id += 1
            im = cv2.imread(os.path.join(im_dir, im_path))
            h, w = im.shape[:2]
            image = {'file_name': os.path.basename(im_path), 'width': w, 'height': h, 'id': image_id}
            images.append(image)
            # 格式化标签，labels存储单张图像的所有rect
            result = self.run(im)
            temp_anns = self._frame_to_coco_annotations(result, image_id=image_id, ann_start_id=len(annotations) + 1)
            annotations.extend(temp_anns)

        self._save(images, annotations, save_path)

    def _frame_to_coco_annotations(self, frame, image_id=1, ann_start_id=1):
        Annotations = []
        ann_id = ann_start_id
        # ─────────────────────────────
        #  ProbeMask + Keypoints
        #  每个探针，包含一组关键点
        # ─────────────────────────────
        if frame.probe_masks is not []:
            for index, tip in enumerate(frame.probe_masks):
                poly = self._rotated_rect_to_polygon(tip.tip_rect)
                xs = poly[0::2]
                ys = poly[1::2]
                x_min, y_min = min(xs), min(ys)
                x_max, y_max = max(xs), max(ys)
                if not frame.probe_keypoints[index]:
                    continue
                kps = self._keypoints_to_coco(frame.probe_keypoints[index])
                Annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": tip.label.value,
                    "keypoints": kps,
                    "num_keypoints": len(kps),
                    "segmentation": [poly],
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": float((x_max - x_min) * (y_max - y_min)),
                    "iscrowd": 0,
                    # 自定义字段
                    "rotation": tip.tip_rect,
                    "attributes": {
                        "aspect_ratio": float(tip.aspect_ratio),
                        "deformation_score": float(tip.deformation_score),
                        "wear_level": frame.probe_keypoints[index].wear_level.value,
                        "wear_score": float(frame.probe_keypoints[index].wear_score),
                        "contact_state": frame.probe_keypoints[index].contact_state.value
                    }
                })
                ann_id += 1
        # ─────────────────────────────
        #  calibs（旋转框）
        #  每个校准片
        # ─────────────────────────────
        if frame.calib_pads is not []:
            for cp in frame.calib_pads:
                poly = self._rotated_rect_to_polygon(cp.pad_rects)
                xs = poly[0::2]
                ys = poly[1::2]

                x_min, y_min = min(xs), min(ys)
                x_max, y_max = max(xs), max(ys)

                Annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cp.calib_type.value,
                    "segmentation": [poly],
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": float((x_max - x_min) * (y_max - y_min)),
                    "iscrowd": 0,
                    # 自定义字段
                    "rotation": cp.pad_rects,
                    "attributes": {
                        "deformation_score": 0
                    }
                })
                ann_id += 1

        # ─────────────────────────────
        #  ScrubMarks
        #  每个针痕
        # ─────────────────────────────
        if frame.scrub_groups is not []:
            for group in frame.scrub_groups:
                for m in group.marks:
                    polys = self._mask_to_polygons(m.mask)
                    for poly in polys:
                        x, y, w, h = cv2.boundingRect(
                            np.array(poly).reshape(-1, 2).astype(np.int32)
                        )
                        Annotations.append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": m.label.value,
                            "segmentation": [poly],
                            "bbox": [x, y, w, h],
                            "area": float(m.area_px),
                            "iscrowd": 0,
                            # 自定义字段
                            "group_id": int(group.group_id),
                            "attributes": {
                                "probe_label": m.label.value,
                                "centroid": m.centroid
                            }
                        })
                        ann_id += 1

        return Annotations

    def _rotated_rect_to_polygon(self, rect):
        box = cv2.boxPoints(rect)
        return box.reshape(-1).tolist()

    def _keypoints_to_coco(self, kps):
        out = []
        for p in kps.keyPoints:
            v = 2 if p.visible else 0
            out.extend([float(p.x), float(p.y), v])
        return out

    def _mask_to_polygons(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polys = []
        for cnt in contours:
            if len(cnt) >= 3:
                polys.append(cnt.reshape(-1, 2).flatten().tolist())
        return polys

    def _save(self, images, annotations, path):
        ann = {}
        ann['type'] = 'instances'
        ann['images'] = images
        ann['annotations'] = annotations
        # 类别与具体对象不同：类别只包含大类，对象得考虑个数，对应子模块的问题
        categories = [
            {"id": 0, "name": "PROBE_G1"},
            {"id": 1, "name": "PROBE_S"},
            {"id": 2, "name": "PROBE_G2"},
            {"id": 3, "name": "PROBE_GSG"},
            {"id": 4, "name": "KEYPOINTS"},
            {"id": 10, "name": "LOAD"},
            {"id": 11, "name": "OPEN"},
            {"id": 12, "name": "SHORT"},
            {"id": 13, "name": "THRU"},
            {"id": 14, "name": "UNKNOWN"},

        ]
        ann['categories'] = categories
        json.dump(ann, open(path, 'w'))
