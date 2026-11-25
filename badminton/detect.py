"""球员检测与跟踪。

默认后端是**纯运动前景法**（MOG2 背景减除 + 形态学 + 轮廓），只依赖 OpenCV，
家用电脑 CPU 就能跑，固定机位下对“两个人在场上跑动”足够好用。

可选后端是 YOLO（ultralytics），装了就用；接口一致，返回同样的 Obs。

输出两路观测：near（近端，画面下方那个）/ far（远端）。
两者都是“脚点”（框底边中点）像素坐标。
"""

from __future__ import annotations

import math

from .cvreq import get_cv2


class Obs:
    """单帧单人观测。"""

    __slots__ = ("x", "y", "x1", "y1", "x2", "y2", "energy", "score")

    def __init__(self, x, y, x1, y1, x2, y2, energy, score=1.0):
        self.x = float(x)
        self.y = float(y)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.energy = float(energy)
        self.score = float(score)

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def __repr__(self):  # pragma: no cover
        return "<Obs foot=(%.0f,%.0f) h=%.0f e=%.2f>" % (self.x, self.y, self.h, self.energy)


def _order_and_pick(boxes, prev, meta, cfg):
    """从候选框里挑出近端/远端。

    规则：
      1. 有上一帧位置时，先按脚点距离关联（gate 内最近者）当作近端；
      2. 否则取脚点最低（画面最下）的框当作近端；
      3. 远端 = 剩余框里脚点最靠上、且明显高于近端、面积最大者。
    """
    frame_h = meta.height
    near = far = None
    rest = list(boxes)

    if prev is not None:
        gate = cfg.track_gate_frac * frame_h
        best, best_d = None, gate
        for b in rest:
            d = math.hypot(b.x - prev.x, b.y - prev.y)
            if d < best_d:
                best, best_d = b, d
        if best is not None:
            near = best
            rest.remove(best)

    if near is None and rest:
        near = max(rest, key=lambda b: b.y)
        rest.remove(near)

    if near is not None and rest:
        floor = near.y - 0.06 * frame_h
        cands = [b for b in rest if b.y < floor and b.y > 0.03 * frame_h]
        if cands:
            far = max(cands, key=lambda b: b.h * b.w)
    return near, far


class MotionTracker:
    """MOG2 前景 + 轮廓 + 近/远端指派。"""

    name = "motion"

    def __init__(self, meta, cfg):
        cv2 = get_cv2()
        self.cv2 = cv2
        self.meta = meta
        self.cfg = cfg
        self.dw = int(max(64, cfg.detect_width))
        self.scale = meta.width / float(self.dw)
        self.dh = int(round(meta.height / self.scale))
        self.fg = cv2.createBackgroundSubtractorMOG2(
            history=int(cfg.bg_history),
            varThreshold=float(cfg.bg_var_threshold),
            detectShadows=False,
        )
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.warmup = 12
        self.prev_near = None
        self.miss_near = 0
        self.miss_far = 0
        self.prev_far = None

    def _foreground_boxes(self, gray):
        cv2 = self.cv2
        small = cv2.resize(gray, (self.dw, self.dh), interpolation=cv2.INTER_AREA)
        mask = self.fg.apply(small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        min_h = self.cfg.min_box_h_frac * self.dh
        max_h = self.cfg.max_box_h_frac * self.dh
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.cfg.min_area_px:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bh < min_h or bh > max_h:
                continue
            if bw <= 2 or bh / float(bw) < 0.8:
                continue  # 人应该明显高于宽
            roi = mask[by:by + bh, bx:bx + bw]
            energy = float(roi.mean()) / 255.0 if roi.size else 0.0

            s = self.scale
            x1, y1 = bx * s, by * s
            x2, y2 = (bx + bw) * s, (by + bh) * s
            boxes.append(Obs((x1 + x2) / 2.0, y2, x1, y1, x2, y2, energy,
                             score=area / float(self.dw * self.dh)))
        return boxes

    def step(self, gray):
        if self.warmup > 0:
            self.warmup -= 1
            self._foreground_boxes(gray)
            return {"near": None, "far": None}

        boxes = self._foreground_boxes(gray)
        near, far = _order_and_pick(boxes, self.prev_near, self.meta, self.cfg)

        if near is None:
            self.miss_near += 1
            if self.miss_near > self.cfg.track_max_miss:
                self.prev_near = None
            near = self.prev_near
        else:
            self.miss_near = 0
            self.prev_near = near

        if far is None:
            self.miss_far += 1
            if self.miss_far > self.cfg.track_max_miss:
                self.prev_far = None
            far = self.prev_far
        else:
            self.miss_far = 0
            self.prev_far = far

        return {"near": near, "far": far}

    def close(self):
        pass


class YoloTracker:
    """可选后端：ultralytics YOLO 人体检测（需要额外安装与模型文件）。"""

    name = "yolo"

    def __init__(self, meta, cfg, weights="yolov8n.pt", conf=0.25):
        try:
            from ultralytics import YOLO  # noqa: WPS433
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("未安装 ultralytics，无法使用 yolo 检测后端") from exc
        self.model = YOLO(weights)
        self.meta = meta
        self.cfg = cfg
        self.conf = conf
        self.prev_near = None
        self.prev_far = None

    def step(self, gray):
        cv2 = get_cv2()
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        results = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)
        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf[0]) if b.conf is not None else 1.0
                boxes.append(Obs((x1 + x2) / 2.0, y2, x1, y1, x2, y2, 0.5, conf))
        near, far = _order_and_pick(boxes, self.prev_near, self.meta, self.cfg)
        if near is not None:
            self.prev_near = near
        if far is not None:
            self.prev_far = far
        return {"near": near, "far": far}

    def close(self):
        pass


def build_tracker(meta, cfg):
    backend = (cfg.detector if hasattr(cfg, "detector") else "motion") or "motion"
    if backend == "yolo":
        return YoloTracker(meta, cfg)
    return MotionTracker(meta, cfg)
