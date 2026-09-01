"""
metrics.py
==========
评估指标：覆盖五个任务的完整度量体系
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import cv2
import numpy as np
from data_pipeline import *


class GSGProbeMetrics:
    # PCK 阈值（像素，原图坐标系）
    PCK_THRESHOLDS: tuple[float, ...] = (2.0, 5.0, 10.0)
    # IoU 匹配阈值
    IOU_THR: float = 0.3

    def __init__(self):
        self._inference_times = None
        self._calib_fn = None
        self._calib_fp = None
        self._calib_tp = None
        self._calib_center_errors = None
        self._calib_ious = None
        self._probe_fn = None
        self._probe_fp = None
        self._probe_tp = None
        self._probe_ious = None
        self._kp_errors = None
        self.reset()

    def reset(self):
        # ── 关键点（仅统计 visibility==2 的完全可见点）──
        self._kp_errors: list[float] = []
        # ── 关键点可见性分类（0缺失/1遮挡/2可见）──
        self._kp_vis_correct: int = 0
        self._kp_vis_total: int = 0

        # ── 探针状态（遮挡率/失焦度/几何可见比例）──
        self._occlusion_errors: list[float] = []
        self._defocus_errors: list[float] = []
        self._visible_ratio_errors: list[float] = []

        # ── 探针分割 ──
        self._probe_ious: list[float] = []  # 匹配对的 mask IoU（无 mask 退回 bbox IoU）
        self._probe_tp = 0
        self._probe_fp = 0
        self._probe_fn = 0

        # ── 校准片检测 ──
        self._calib_ious: list[float] = []
        self._calib_center_errors: list[float] = []
        self._calib_tp = 0
        self._calib_fp = 0
        self._calib_fn = 0

        # ── 推理延迟 ──
        self._inference_times: list[float] = []

    # ----------------------------------------------------------------
    # public entry
    # ----------------------------------------------------------------

    def update(
            self,
            result: dict,
            batch_item: dict,
            inference_time_ms: float | None = None,):

        # ── 解析 GT ──────────────────────────────────────────────────
        gt_probe, gt_kp = self._parse_gt_probe(batch_item)
        gt_calib = self._parse_gt_calib(batch_item)
        # ── 解析 Pred ────────────────────────────────────────────────
        pred_probe = result['probe']  # List[N_pred dict]，每个含 'keypoints'
        pred_calib = result['calibs']

        # ── 子指标更新 ───────────────────────────────────────────────
        self._update_probe(pred_probe, gt_probe, gt_kp)
        self._update_calib(pred_calib, gt_calib)

        if inference_time_ms is not None:
            self._inference_times.append(float(inference_time_ms))

    # ----------------------------------------------------------------
    # GT parsers
    # ----------------------------------------------------------------

    @staticmethod
    def _parse_gt_probe(
            batch_item: dict,
    ) -> tuple[list[dict], list[list[dict]]]:
        rboxes = batch_item['probe_rboxes']  # List[[cx,cy,w,h,theta_rad]]
        masks = batch_item['probe_masks']
        keypoints = batch_item['probe_keypoints']
        visibility = batch_item['probe_visibility']
        labels = batch_item['probe_labels']
        # 探针状态 GT（遮挡率/失焦度/几何可见比例），来自 data_pipeline 的自监督代理标签，
        # 旧版本 batch_item 里若没有这三个字段，用 get 兜底为空列表，指标里对应跳过统计。
        occ_gt = batch_item.get('probe_occlusion_ratio', [])
        defocus_gt = batch_item.get('probe_defocus_level', [])
        visible_gt = batch_item.get('probe_visible_ratio', [])

        gt_probe: list[dict] = []
        gt_kp: list[list[dict]] = []

        for i, rbox in enumerate(rboxes):

            cx, cy, bw, bh, theta = [float(v) for v in rbox[:5]]
            item: dict = dict(
                rect=[cx, cy, bw, bh, theta],  # 统一旋转框格式，与 calib 对齐
                label=labels[i],  # probe 单类
            )
            if i < len(masks) and masks[i] is not None:
                item['mask'] = np.asarray(masks[i], dtype=bool)
            if i < len(occ_gt):
                item['occlusion_ratio'] = float(occ_gt[i])
            if i < len(defocus_gt):
                item['defocus_level'] = float(defocus_gt[i])
            if i < len(visible_gt):
                item['visible_ratio'] = float(visible_gt[i])
            gt_probe.append(item)

            # ── 关键点 ─────────────────────────────────────────────
            kp_list: list[dict] = []
            if i < len(keypoints) and keypoints[i] is not None:
                kps = np.asarray(keypoints[i])  # (K, 2)
                if i < len(visibility) and visibility[i] is not None:
                    vis = np.asarray(visibility[i], dtype=int).reshape(-1)
                else:
                    vis = np.full(len(kps), 2, dtype=int)  # 默认完全可见

                for kid, (xy, v) in enumerate(zip(kps, vis)):
                    kp_list.append(dict(
                        keypoint_id=kid,
                        x=float(xy[0]),
                        y=float(xy[1]),
                        visibility=int(v),
                    ))
            gt_kp.append(kp_list)

        return gt_probe, gt_kp

    @staticmethod
    def _parse_gt_calib(batch_item: dict) -> list[dict]:
        """解析校准片 GT，使用统一旋转框 (cx,cy,w,h,theta)，与预测端的统一表示对齐。"""
        rboxes = batch_item['calib_rboxes']  # List[[cx,cy,w,h,theta_rad]]
        labels = batch_item['calib_labels']
        result = []
        for rbox, lbl in zip(rboxes, labels):
            cx, cy, w, h, theta = [float(v) for v in rbox[:5]]
            # label 字段名与 _match_by_rect 中 p['label']/g['label'] 的比较保持一致
            result.append(dict(rect=[cx, cy, w, h, theta], label=int(lbl)))
        return result

    # ----------------------------------------------------------------
    # geometry helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _rect_iou(r1: list, r2: list) -> float:
        """统一旋转框 IoU：r = [cx, cy, w, h, theta_rad]（theta 缺省按 0 处理，向后兼容）。"""
        w1, h1 = float(r1[2]), float(r1[3])
        w2, h2 = float(r2[2]), float(r2[3])
        if w1 <= 0.0 or h1 <= 0.0 or w2 <= 0.0 or h2 <= 0.0:
            return 0.0
        theta1 = float(r1[4]) if len(r1) > 4 else 0.0
        theta2 = float(r2[4]) if len(r2) > 4 else 0.0
        rect1 = ((float(r1[0]), float(r1[1])), (w1, h1), math.degrees(theta1))
        rect2 = ((float(r2[0]), float(r2[1])), (w2, h2), math.degrees(theta2))
        inter_type, inter_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
        if inter_pts is None or len(inter_pts) < 3:
            inter_area = 0.0
        else:
            inter_area = float(cv2.contourArea(cv2.convexHull(inter_pts.astype(np.float32))))
        union = w1 * h1 + w2 * h2 - inter_area
        return float(inter_area / union) if union > 0 else 0.0

    @staticmethod
    def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
        inter = float(np.logical_and(m1, m2).sum())
        union = float(np.logical_or(m1, m2).sum())
        return inter / union if union > 0 else 0.0

    # ----------------------------------------------------------------
    # matching
    # ----------------------------------------------------------------

    def _match_by_rect(
            self,
            pred: list[dict],
            gt: list[dict],
            iou_thr: float | None = None,
    ) -> tuple[list[dict], list[int], list[int]]:

        thr = iou_thr if iou_thr is not None else self.IOU_THR

        order = sorted(
            range(len(pred)),
            key=lambda i: pred[i]['score'],
            reverse=True,
        )
        gt_used: set[int] = set()
        matched_pred: set[int] = set()
        matches: list[dict] = []

        for pi in order:
            p = pred[pi]
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt):
                if j in gt_used:
                    continue
                if p['label'] != g['label']:
                    continue
                iou = self._rect_iou(p['rect'], g['rect'])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= thr:
                matches.append(dict(pred_idx=pi, gt_idx=best_j, rect_iou=best_iou))
                gt_used.add(best_j)
                matched_pred.add(pi)

        # 假正例FP：所有没匹配上真值的预测框下标
        fp_idxs = [i for i in range(len(pred)) if i not in matched_pred]
        # 假负例FN：所有没被预测框匹配到的真值框下标
        fn_idxs = [j for j in range(len(gt)) if j not in gt_used]
        return matches, fp_idxs, fn_idxs

    # ----------------------------------------------------------------
    # updaters
    # ----------------------------------------------------------------

    def _update_probe(
            self,
            pred_probe: list[dict],
            gt_probe: list[dict],
            gt_kp: list[list[dict]],
    ) -> None:

        matches, fp_idxs, fn_idxs = self._match_by_rect(pred_probe, gt_probe)
        for m in matches:
            pi = m['pred_idx']
            gi = m['gt_idx']
            p = pred_probe[pi]
            g = gt_probe[gi]

            # ── 探针分割指标 ─────────────────────────────────────────
            self._probe_tp += 1
            if 'mask' in p and 'mask' in g:
                self._probe_ious.append(self._mask_iou(p['mask'], g['mask']))
            else:
                self._probe_ious.append(m['rect_iou'])

            # ── 探针状态指标：遮挡率 / 失焦度 / 几何可见比例 ──────────────
            if 'occlusion_ratio' in g and 'occlusion_ratio' in p:
                self._occlusion_errors.append(abs(float(p['occlusion_ratio']) - g['occlusion_ratio']))
            if 'defocus_level' in g and 'defocus_level' in p:
                self._defocus_errors.append(abs(float(p['defocus_level']) - g['defocus_level']))
            if 'visible_ratio' in g and 'visible_ratio' in p:
                self._visible_ratio_errors.append(abs(float(p['visible_ratio']) - g['visible_ratio']))

            # ── 关键点指标 ────
            if gi >= len(gt_kp):
                continue
            inst_gt_kp = gt_kp[gi]
            # pred 侧：直接从 probe dict 取，永远与该实例绑定
            raw_kps = p['keypoints']  # List[K dict] 或空
            if not raw_kps:
                continue
            # 统一为 ndarray(K, 3)：[x, y, confidence]
            inst_pred_kp = np.array(
                [[kp['x'], kp['y'], kp['confidence']]
                 for kp in raw_kps],
                dtype=np.float32,
            )  # (K, 3)
            if inst_pred_kp.ndim != 2 or inst_pred_kp.shape[1] < 2:
                continue
            gt_map = {kp_dict['keypoint_id']: kp_dict for kp_dict in inst_gt_kp}
            # 可见性分类预测：与 raw_kps 顺序对齐（keypoint_id 相同）
            vis_pred_map = {
                kp['keypoint_id']: kp['visibility_pred']
                for kp in raw_kps if 'visibility_pred' in kp
            }
            for kid in range(inst_pred_kp.shape[0]):
                g_kp = gt_map.get(kid)
                if g_kp is None:
                    continue

                # 可见性分类准确率：0缺失/1遮挡/2可见，三类都参与统计（不像 PCK 那样只看 v=2）
                if kid in vis_pred_map:
                    self._kp_vis_total += 1
                    if int(vis_pred_map[kid]) == int(g_kp['visibility']):
                        self._kp_vis_correct += 1

                if g_kp['visibility'] != 2:
                    # PCK 仅评估完全可见点（v=2），遮挡(v=1)和缺失(v=0)跳过
                    continue
                px = float(inst_pred_kp[kid, 0])
                py = float(inst_pred_kp[kid, 1])
                err = float(np.hypot(px - g_kp['x'], py - g_kp['y']))
                self._kp_errors.append(err)

        self._probe_fp += len(fp_idxs)
        self._probe_fn += len(fn_idxs)

    def _update_calib(
            self,
            pred_calib: list[dict],
            gt_calib: list[dict],
    ):
        """校准片检测评估：bbox 匹配 + 中心点偏差。"""
        matches, fp_idxs, fn_idxs = self._match_by_rect(pred_calib, gt_calib)

        for m in matches:
            p = pred_calib[m['pred_idx']]
            g = gt_calib[m['gt_idx']]
            self._calib_tp += 1
            self._calib_ious.append(m['rect_iou'])
            # 中心点偏差：rect 格式为 [cx, cy, w, h]
            self._calib_center_errors.append(float(np.hypot(
                p['rect'][0] - g['rect'][0],
                p['rect'][1] - g['rect'][1],
            )))

        self._calib_fp += len(fp_idxs)
        self._calib_fn += len(fn_idxs)

    # ----------------------------------------------------------------
    # compute
    # ----------------------------------------------------------------

    def compute(self) -> dict:
        """汇总所有累积统计，返回指标字典。"""
        r: dict = {}
        # ── 关键点 ──────────────────────────────────────────────────
        if self._kp_errors:
            e = np.array(self._kp_errors)
            r['kp_mean_error_px'] = float(np.mean(e))
            r['kp_median_error_px'] = float(np.median(e))
            r['kp_std_error_px'] = float(np.std(e))
            r['kp_sample_count'] = len(e)
            for thr in self.PCK_THRESHOLDS:
                r[f'kp_pck_{thr:g}px'] = float(np.mean(e < thr))
        else:
            r['kp_mean_error_px'] = 0.0
            r['kp_sample_count'] = 0
            for thr in self.PCK_THRESHOLDS:
                r[f'kp_pck_{thr:g}px'] = 0.0

        # ── 关键点可见性分类（0缺失/1遮挡/2可见）──────────────────────
        r['kp_visibility_acc'] = (
            self._kp_vis_correct / self._kp_vis_total if self._kp_vis_total > 0 else 0.0
        )
        r['kp_visibility_sample_count'] = self._kp_vis_total

        # ── 探针状态：遮挡率 / 失焦度 / 几何可见比例（部分结构出视野）──────
        r['occlusion_mae'] = float(np.mean(self._occlusion_errors)) if self._occlusion_errors else 0.0
        r['defocus_mae'] = float(np.mean(self._defocus_errors)) if self._defocus_errors else 0.0
        r['visible_ratio_mae'] = float(np.mean(self._visible_ratio_errors)) if self._visible_ratio_errors else 0.0
        r['probe_state_sample_count'] = len(self._occlusion_errors)

        # ── 探针分割 ─────────────────────────────────────────────────
        r['probe_seg_miou'] = (
            float(np.mean(self._probe_ious)) if self._probe_ious else 0.0
        )
        probe_p = self._probe_tp / max(self._probe_tp + self._probe_fp, 1)
        probe_r = self._probe_tp / max(self._probe_tp + self._probe_fn, 1)
        r['probe_seg_precision'] = probe_p
        r['probe_seg_recall'] = probe_r
        r['probe_seg_f1'] = (
            2 * probe_p * probe_r / (probe_p + probe_r)
            if (probe_p + probe_r) > 0 else 0.0
        )
        r['probe_tp'] = self._probe_tp
        r['probe_fp'] = self._probe_fp
        r['probe_fn'] = self._probe_fn

        # ── 校准片检测 ────────────────────────────────────────────────
        r['calib_det_miou'] = (
            float(np.mean(self._calib_ious)) if self._calib_ious else 0.0
        )
        calib_p = self._calib_tp / max(self._calib_tp + self._calib_fp, 1)
        calib_r = self._calib_tp / max(self._calib_tp + self._calib_fn, 1)
        r['calib_det_precision'] = calib_p
        r['calib_det_recall'] = calib_r
        r['calib_det_f1'] = (
            2 * calib_p * calib_r / (calib_p + calib_r)
            if (calib_p + calib_r) > 0 else 0.0
        )
        r['calib_tp'] = self._calib_tp
        r['calib_fp'] = self._calib_fp
        r['calib_fn'] = self._calib_fn
        if self._calib_center_errors:
            ce = np.array(self._calib_center_errors)
            r['calib_center_err_mean'] = float(np.mean(ce))
            r['calib_center_err_std'] = float(np.std(ce))
            r['calib_center_err_p95'] = float(np.percentile(ce, 95))
        else:
            r['calib_center_err_mean'] = 0.0
            r['calib_center_err_std'] = 0.0
            r['calib_center_err_p95'] = 0.0

        # ── 推理延迟 ─────────────────────────────────────────────────
        mean_ms = float(np.mean(self._inference_times)) if self._inference_times else 0.0
        p95_ms = float(np.percentile(self._inference_times, 95)) if self._inference_times else 0.0
        r['latency_mean_ms'] = mean_ms
        r['latency_p95_ms'] = p95_ms
        r['fps'] = (1000.0 / mean_ms) if mean_ms > 0 else 0.0

        return r
