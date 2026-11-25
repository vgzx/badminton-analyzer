"""球场几何：单应变换 + 坐标单位。

两种模式
--------
* **calibrated**：用户点击画面中 4 个球场角点，求解单应矩阵 H，
  像素 -> 米 -> 球场单位 u，得到真实尺度与纵深。
* **relative**：未标定。把画面当作“整块场地”线性归一化，
  只能得到相对位移，不能得到米制距离。报告里会明确标注。

单应求解用纯 numpy 实现（SVD 解 8 自由度线性系统），
这样标定逻辑本身可以在没有 OpenCV 的环境里被测试。
"""

from __future__ import annotations

import json
import os

import numpy as np

from .config import COURT_LENGTH_M, COURT_LENGTH_U, COURT_WIDTH_M


def dlt_homography(src_pts, dst_pts) -> np.ndarray:
    """四点及以上的直接线性变换求 H，使 dst ≈ H @ src。

    src_pts / dst_pts: 形如 [[x, y], ...] 的序列，长度 >= 4。
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src_pts 与 dst_pts 必须是同形状的 Nx2 数组")
    if len(src) < 4:
        raise ValueError("单应变换至少需要 4 个点")

    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(rows, dtype=np.float64)
    singular = np.linalg.svd(A, compute_uv=False)
    if singular[0] <= 0 or singular[-1] < 1e-9 * singular[0]:
        raise ValueError("点集退化（共线或重复），无法求解单应矩阵")
    _, _, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("退化配置，无法求解单应矩阵（点共线或重复）")
    H = H / H[2, 2]
    # 像素点共线时会得到一个奇异矩阵：平面被压成一条线，变换不可逆
    try:
        condition = float(np.linalg.cond(H))
    except np.linalg.LinAlgError:
        condition = float("inf")
    if not np.isfinite(condition) or condition > 1e10:
        raise ValueError("点集退化：四个点近似共线，请重新选择球场角点")
    # 兜底：几何上不可解的情况用重投影误差识别
    projected = apply_homography(H, src)
    error = float(np.max(np.linalg.norm(projected - dst, axis=1)))
    scale = float(np.max(np.ptp(dst, axis=0))) or 1.0
    if error > 1e-2 * max(scale, 1.0):
        raise ValueError("点集退化（共线或重复），重投影误差 %.3f 像素" % error)
    return H


def apply_homography(H: np.ndarray, pts) -> np.ndarray:
    """对 Nx2 像素点应用 H，返回 Nx2。"""
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homo = np.hstack([arr, np.ones((len(arr), 1))])
    out = homo @ H.T
    w = out[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return out[:, :2] / w


# 4 个角点的米制坐标，顺序必须与用户点击顺序一致：
# 近端左角 -> 近端右角 -> 远端左角 -> 远端右角（从底线后机位看）
COURT_CORNERS_M = [
    [0.0, 0.0],
    [COURT_WIDTH_M, 0.0],
    [0.0, COURT_LENGTH_M],
    [COURT_WIDTH_M, COURT_LENGTH_M],
]


class Court:
    """像素坐标 <-> 球场单位坐标（u，1 u = 6.10 m）。"""

    def __init__(self, H=None, frame_w=1920, frame_h=1080, points=None):
        self.frame_w = float(frame_w)
        self.frame_h = float(frame_h)
        self.H = None
        self.H_inv = None
        self.points = points or []
        self.mode = "relative"
        if H is not None:
            self.H = np.asarray(H, dtype=np.float64)
            self.H_inv = np.linalg.inv(self.H)
            self.mode = "calibrated"

    # ---------- 构造 ----------
    @classmethod
    def from_points(cls, image_points, frame_w, frame_h) -> "Court":
        """用 4 个图像角点（近左、近右、远左、远右）构造标定好的球场。"""
        pts = np.asarray(image_points, dtype=np.float64)
        if pts.shape != (4, 2):
            raise ValueError("需要恰好 4 个角点，形如 [[x,y], ...]")
        H = dlt_homography(COURT_CORNERS_M, pts)  # 米 -> 像素
        court = cls(H=H, frame_w=frame_w, frame_h=frame_h, points=pts.tolist())
        return court

    @classmethod
    def relative(cls, frame_w, frame_h) -> "Court":
        return cls(H=None, frame_w=frame_w, frame_h=frame_h)

    @classmethod
    def load(cls, path, frame_w, frame_h) -> "Court":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            data = {"points": data}
        points = data.get("points")
        if not points:
            raise ValueError("标定文件缺少 points 字段")
        if data.get("frame_size") and list(data["frame_size"]) != [frame_w, frame_h]:
            # 分辨率变了但角点仍是像素坐标 -> 按比例缩放
            sw = frame_w / float(data["frame_size"][0])
            sh = frame_h / float(data["frame_size"][1])
            points = [[p[0] * sw, p[1] * sh] for p in points]
        return cls.from_points(points, frame_w, frame_h)

    def save(self, path) -> None:
        payload = {
            "points": self.points,
            "frame_size": [self.frame_w, self.frame_h],
            "mode": self.mode,
            "note": "points 顺序：近端左角、近端右角、远端左角、远端右角",
        }
        folder = os.path.dirname(os.path.abspath(path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ---------- 变换 ----------
    def to_unit(self, px: float, py: float):
        """像素 -> 球场单位 (x_u, y_u)。y_u 从近端底线向远端增长。"""
        if self.mode == "calibrated":
            mx, my = apply_homography(self.H_inv, [[px, py]])[0]
            return mx / COURT_WIDTH_M, my / COURT_WIDTH_M
        x_u = px / self.frame_w
        y_u = (1.0 - py / self.frame_h) * COURT_LENGTH_U
        return x_u, y_u

    def to_px(self, x_u: float, y_u: float):
        """球场单位 -> 像素（画图用）。"""
        if self.mode == "calibrated":
            pt = apply_homography(self.H, [[x_u * COURT_WIDTH_M, y_u * COURT_WIDTH_M]])[0]
            return float(pt[0]), float(pt[1])
        px = x_u * self.frame_w
        py = (1.0 - y_u / COURT_LENGTH_U) * self.frame_h
        return float(px), float(py)

    def unit_label(self) -> str:
        return "u" if self.mode == "calibrated" else "相对单位"

    def describe(self) -> str:
        if self.mode == "calibrated":
            return "已标定（4 角点单应，1 u = 6.10 m）"
        return "未标定（画面线性归一化，距离仅为相对单位，非米制）"
