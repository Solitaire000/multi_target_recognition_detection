"""
definitions.py
==============
核心几何定义与数据结构。所有模块共享此文件的类型定义。
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple
import numpy as np


# ─────────────────────────────────────────────
# 枚举：目标类别
# ─────────────────────────────────────────────

class ProbeLabel(Enum):
    """探针部件语义标签（用于实例分割）"""
    PROBE_G1 = 0
    PROBE_S = 1
    PROBE_G2 = 2
    PROBE_GSG = 3
    PROBE_KYEPOINTS = 4
class CalibLabel(Enum):
    """校准片类型标签"""
    LOAD  = 10   # Load校准片
    OPEN  = 11   # Open校准片
    SHORT = 12   # Short校准片
    THRU  = 13   # Thru校准片
    UNKNOWN = 14
class ContactState(Enum):
    """探针接触状态"""
    NON_CONTACT = auto()   # 探针悬空，无针痕
    CONTACT     = auto()   # 探针已接触，有针痕，探针微形变
class WearLevel(Enum):
    """磨损等级（用于报警和维护建议）"""
    NORMAL   = 20   # 针尖形状规则，关键点可确定
    MILD     = 21   # 轻微形变，关键点仍可估计，置信度下降
    SEVERE   = 22   # 严重形变，针尖点无法确定，需人工干预


# ─────────────────────────────────────────────
# 几何数据类
# ─────────────────────────────────────────────

@dataclass
class KeyPoint:
    """
    探针针尖关键点。
    定义：
    confidence: 模型输出置信度 [0,1]；低于阈e值时触发WarLevel升级
    """
    x: float
    y: float
    confidence: float = 1.0
    visible: bool = True          # 是否在图像FOV内且未被严重遮挡

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class ProbeKeyPoints:
    """
    GSG探针的关键点
    """
    # 序号 0-15
    # 统一对应点： 0-15
    label: ProbeLabel = 4
    keyPoints : List[KeyPoint] = field(default_factory=list)
    wear_level: WearLevel = WearLevel.NORMAL
    wear_score: float = 0.0       # 连续磨损分数 [0,1]，越大越严重
    contact_state: ContactState = ContactState.NON_CONTACT

    def tip_array(self) -> np.ndarray:
        return np.array([[kp.x, kp.y] for kp in self.keyPoints], dtype=np.float32)

    def mean_confidence(self) -> float:
        vis = [p for p in self.keyPoints if p.visible]
        return float(np.mean([p.confidence for p in vis])) if vis else 0.0


@dataclass
class ProbeTipMask:
    """
    单根探针针尖的分割掩模及几何参数。
    tip_polygon: 针尖轮廓多边形（亚像素精度）
    是否规则 = width/height 接近已知标称长宽比
    """
    tip_rect: Tuple[Tuple[float, float], Tuple[float, float], float]  # 每个probe的((cx,cy),(w,h),θ)
    label: ProbeLabel = ProbeLabel.PROBE_GSG
    aspect_ratio: float = 0.0        # 实测长宽比
    nominal_aspect_ratio: float = 3.0  # 标称长宽比（需标定）
    deformation_score: float = 0.0   # 形变评分：|实测-标称|/标称


@dataclass
class ScrubMark:
    """
    单个针痕实例。
    同一组针痕由同一次接触产生，label标识属于G1/S/G2哪根针。
    area_px: 针痕面积（像素²）
    area_um2: 针痕面积（µm²，需像素-物理尺寸标定系数）
    centroid: 针痕重心坐标
    group_id: 所属接触组编号（同一次接触的三个针痕共享同一group_id）
    """
    label: ProbeLabel
    mask: np.ndarray                # 二值掩模
    area_px: float = 0.0
    area_um2: Optional[float] = None
    centroid: Tuple[float, float] = (0.0, 0.0)
    group_id: int = -1              # -1表示未分组
    bbox: Tuple[int,int,int,int] = (0,0,0,0)   # x1y1x2y2


@dataclass
class ScrubGroup:
    """
    一组针痕（一次接触的G1+S+G2三个针痕）。
    marks: 最多3个ScrubMark，按 G1/S/G2 顺序
    asymmetry_index: (A_G1 - A_G2)/(A_G1 + A_G2)，用于调平角估计
    """
    group_id: int
    marks: List[ScrubMark] = field(default_factory=list)
    asymmetry_index: float = 0.0
    contact_force_estimate: Optional[float] = None  # 单位 mN，需标定

    def total_area_px(self) -> float:
        return sum(m.area_px for m in self.marks)

    def compute_asymmetry(self):
        areas = {m.label: m.area_px for m in self.marks}
        a_g1 = areas.get(ProbeLabel.PROBE_G1, 0.0)
        a_g2 = areas.get(ProbeLabel.PROBE_G2, 0.0)
        denom = a_g1 + a_g2
        self.asymmetry_index = (a_g1 - a_g2) / denom if denom > 1e-6 else 0.0


@dataclass
class CalibPad:
    """
    单个校准片焊盘的检测结果。
    calib_type: Load/Open/Short/Thru
    pad_rects: 每个焊盘的旋转外接矩形列表（Load有3个，其他视结构而定）
    center: 校准片整体中心（多焊盘几何中心）
    visible_ratio: 可见面积比（0~1，受遮挡影响）
    completed_center: 遮挡补全后的中心估计（可能与center不同）
    """
    calib_type: CalibLabel
    pad_rects: Tuple[Tuple[float, float], Tuple[float, float], float]  # 每个pad的((cx,cy),(w,h),θ)
    center: Tuple[float, float] = (0.0, 0.0)
    visible_ratio: float = 1.0
    completed_center: Optional[Tuple[float,float]] = None
    confidence: float = 1.0


@dataclass
class FrameResult:
    """
    单帧推理的完整输出，供控制层使用。
    """
    # 初始化类
    probe_keypoints: List[ProbeKeyPoints] = field(default_factory=list)
    probe_masks: List[ProbeTipMask] = field(default_factory=list)
    calib_pads: List[CalibPad] = field(default_factory=list)
    scrub_groups: List[ScrubGroup] = field(default_factory=list)

    # 控制量
    delta_x_px: float = 0.0          # 针尖→目标焊盘 X偏差（像素）
    delta_y_px: float = 0.0          # 针尖→目标焊盘 Y偏差（像素）
    delta_theta_y: float = 0.0       # 调平角偏差估计（度）
    px_per_um: float = 1.0           # 像素-物理尺寸比（标定值）

    # 帧质量
    frame_quality: float = 1.0       # 图像质量评分 [0,1]
    inference_time_ms: float = 0.0
