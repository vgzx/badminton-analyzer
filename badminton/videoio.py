"""视频读取：探测元数据、按目标帧率抽帧、按时间段读取（给片段导出用）。"""

from __future__ import annotations

import os

from .cvreq import get_cv2


class VideoMeta:
    __slots__ = ("path", "fps", "width", "height", "n_frames", "duration")

    def __init__(self, path, fps, width, height, n_frames):
        self.path = path
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.width = int(width)
        self.height = int(height)
        self.n_frames = int(n_frames) if n_frames and n_frames > 0 else 0
        self.duration = self.n_frames / self.fps if self.n_frames else 0.0

    def __repr__(self):  # pragma: no cover
        return "<VideoMeta %dx%d %.2ffps %.1fs>" % (
            self.width, self.height, self.fps, self.duration)


def probe(path: str) -> VideoMeta:
    cv2 = get_cv2()
    if not os.path.exists(path):
        raise FileNotFoundError("视频不存在：%s" % path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(
            "无法打开视频（编码可能不被支持）：%s\n"
            "可先用 ffmpeg 转成 H.264 mp4 再试。" % path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    return VideoMeta(path, fps, width, height, n)


class FrameSampler:
    """顺序读取视频，按目标帧率抽帧。

    用 grab() 跳过不需要的帧（不做事解码），保证时间戳与原始帧号一致。
    """

    def __init__(self, meta: VideoMeta, target_fps: float, max_seconds: float = 0.0):
        self.meta = meta
        self.target_fps = float(target_fps)
        if self.target_fps <= 0:
            raise ValueError("analysis_fps 必须 > 0")
        self.step = max(1, int(round(meta.fps / self.target_fps)))
        self.max_frames = int(max_seconds * meta.fps) if max_seconds and max_seconds > 0 else 0
        self.cv2 = get_cv2()

    @property
    def effective_fps(self) -> float:
        return self.meta.fps / self.step

    def __iter__(self):
        cap = self.cv2.VideoCapture(self.meta.path)
        if not cap.isOpened():
            raise RuntimeError("无法打开视频：%s" % self.meta.path)
        idx = 0
        try:
            while True:
                if self.max_frames and idx >= self.max_frames:
                    break
                ok = cap.grab()
                if not ok:
                    break
                if idx % self.step == 0:
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        break
                    t = idx / self.meta.fps
                    yield idx, t, frame
                idx += 1
        finally:
            cap.release()


def crop(frame, bbox, scale=1.0):
    """按 bbox 裁出正方形区域并适度外扩，越界部分自动裁剪。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * float(scale)
    half = side / 2.0
    nx1 = int(max(0, round(cx - half)))
    ny1 = int(max(0, round(cy - half)))
    nx2 = int(min(w, round(cx + half)))
    ny2 = int(min(h, round(cy + half)))
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return None, (nx1, ny1, nx2, ny2)
    return frame[ny1:ny2, nx1:nx2], (nx1, ny1, nx2, ny2)
