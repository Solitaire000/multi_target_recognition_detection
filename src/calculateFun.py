import numpy as np
from itertools import combinations
import math


# ──────────────────────────────────────────────
# 数据分组
# ──────────────────────────────────────────────

def compute_centroid(points):
    """计算点集的重心。points: list of (x, y)"""
    arr = np.array(points)
    return arr[:, 0].mean(), arr[:, 1].mean()


def line_is_vertical(p1, p2, p3, slope_threshold=5.0):
    """
    判断三点确定的直线是否近似垂直（平行 Y 轴）。
    用 X 方向的极差 / Y 方向的极差 来判断：
      dx_range << dy_range  →  近似垂直
    slope_threshold: |dy| / |dx| 的最低倍数，默认 dx 不超过 dy 的 1/5
    """
    xs = [p1[0], p2[0], p3[0]]
    ys = [p1[1], p2[1], p3[1]]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    if dy == 0:
        return False  # 三点共水平线，不符合
    if dx == 0:
        return True   # 完美垂直
    return (dy / dx) >= slope_threshold


def space_is_equal(p1,p2,p3):

    def distance(p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    d12 = distance(p1,p2)
    d13 = distance(p1,p3)
    d23 = distance(p2,p3)

    if abs(d12-d23)<80 and abs(d13-d12-d23)<1:
        return True
    else:
        return False


def is_collinear(p1, p2, p3, collinear_tol=1e-2):
    """
    判断三点是否（近似）共线。
    用面积法：三角形面积 / 边长归一化后 < collinear_tol
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    # 叉积绝对值 = 2 * 三角形面积
    cross = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    # 归一化：除以最长边的平方，使结果无量纲
    max_side_sq = max(
        (x2 - x1) ** 2 + (y2 - y1) ** 2,
        (x3 - x1) ** 2 + (y3 - y1) ** 2,
        (x3 - x2) ** 2 + (y3 - y2) ** 2,
        1e-9  # 防止除零
    )
    return (cross / max_side_sq) < collinear_tol


def group_vertical_triplets(
    spots_to_needleMarks: dict,
    x_tolerance: float = 50,        # X 坐标偏差容忍度，根据实际坐标尺度调整
    slope_threshold: float = 5.0,   # dy/dx 最小值，值越大要求越接近垂直
    collinear_tol: float = 0.1,    # 共线容忍度，值越小要求越严格
    ):
    """
    将点簇三个一组分类，使得三个点簇的中心：
      1. 近似共线
      2. 连线近似平行于 Y 轴（垂直于 X 轴）
    返回
    ----
    groups   : list of list   每组包含三个 key，代表一个三元组
    ungrouped: list           未能分组的 key
    """

    # 1. 计算每个点簇的中心
    centroids = {}
    for key, pts in spots_to_needleMarks.items():
        centroids[key] = compute_centroid(pts)

    keys = list(centroids.keys())

    # 2. 按 X 坐标排序，便于聚类
    # sorted原理：遍历keys中的值，传入lambda中，计算返回值，根据返回值的大小，对keys重新排序
    keys_sorted_by_x = sorted(keys, key=lambda k: centroids[k][0])

    # 对序号进行操作，不改变原来的dict数据
    # 3. 将 X 坐标相近的点簇归入同一"竖列候选组"
    columns = []   # list of list[key]
    used = set()

    for k in keys_sorted_by_x:
        if k in used:
            continue
        cx = centroids[k][0]
        col = [k]
        used.add(k)
        for k2 in keys_sorted_by_x:
            if k2 in used:
                continue
            if abs(centroids[k2][0] - cx) <= x_tolerance:
                col.append(k2)
                used.add(k2)
        columns.append(col)

    # 4. 在每个竖列内，按 Y 坐标排序，每三个取一组，并验证共线+垂直

    groups = []
    ungrouped = []
    for col in columns:
        # 按 Y 坐标排序
        col_sorted = sorted(col, key=lambda k: centroids[k][1])
        i = 0
        while i + 2 < len(col_sorted):
            trio = col_sorted[i:i + 3]
            pts_trio = [centroids[k] for k in trio]

            ok_collinear = is_collinear(*pts_trio, collinear_tol=collinear_tol)
            ok_vertical  = line_is_vertical(*pts_trio, slope_threshold=slope_threshold)
            ok_space = space_is_equal(*pts_trio)
            if ok_collinear and ok_vertical and ok_space:
                groups.append(trio)
                i += 3
            else:
                # 当前三点不满足条件，跳过第一个，尝试下一个组合
                ungrouped.append(col_sorted[i])
                i += 1

        # 剩余不足三个的
        while i < len(col_sorted):
            ungrouped.append(col_sorted[i])
            i += 1

    # 返回的是分组的编号
    return groups, ungrouped,centroids


def group_vertical_triplets_exhaustive(
    spots_to_needleMarks: dict,
    slope_threshold: float = 5.0,
    collinear_tol: float = 1e-2,
) -> tuple[list, list]:
    """
    穷举所有三元组，找出满足条件的非重叠最大匹配。
    适合点簇数量不多（<= ~30）的场景。
    """
    centroids = {k: compute_centroid(v) for k, v in spots_to_needleMarks.items()}
    keys = list(centroids.keys())

    # 找出所有合法三元组
    valid_triplets = []
    for trio in combinations(keys, 3):
        pts = [centroids[k] for k in trio]
        if is_collinear(*pts, collinear_tol=collinear_tol) and \
           line_is_vertical(*pts, slope_threshold=slope_threshold):
            valid_triplets.append(list(trio))

    # 贪心选取非重叠的三元组（按三点 X 坐标方差升序，优先选最"垂直"的）
    def vertical_score(trio):
        xs = [centroids[k][0] for k in trio]
        return np.var(xs)  # 越小越垂直

    valid_triplets.sort(key=vertical_score)

    used = set()
    groups = []
    for trio in valid_triplets:
        if any(k in used for k in trio):
            continue
        groups.append(trio)
        used.update(trio)

    ungrouped = [k for k in keys if k not in used]
    return groups, ungrouped
