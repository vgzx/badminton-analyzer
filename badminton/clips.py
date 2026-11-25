"""片段导出：按回合起止裁切，可选叠加层。"""

from __future__ import annotations

import os

from .cvreq import get_cv2


def export_clip(meta, t0, t1, out_path, cfg, overlay_fn=None, max_width=0):
    """把 [t0, t1] 导出为 mp4，前后各留 pad 秒。返回 (路径, 实际时长) 或 None。"""
    cv2 = get_cv2()
    start = max(0.0, float(t0) - cfg.clip_pad_before)
    end = float(t1) + cfg.clip_pad_after
    if meta.duration:
        end = min(end, meta.duration)
    if end - start < 0.3:
        return None

    cap = cv2.VideoCapture(meta.path)
    if not cap.isOpened():
        return None
    fps = meta.fps
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    writer = None
    out_size = None
    produced = 0
    try:
        # 某些后端 seek 不精确，允许最多 0.5s 的偏差后再开始写入
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t < start - 0.5:
                continue
            if t > end:
                break
            if overlay_fn is not None:
                frame = overlay_fn(frame, t)
            if max_width and frame.shape[1] > max_width:
                ratio = max_width / float(frame.shape[1])
                frame = cv2.resize(frame, (max_width, int(round(frame.shape[0] * ratio))),
                                   interpolation=cv2.INTER_AREA)
            if writer is None:
                h, w = frame.shape[:2]
                out_size = (w, h)
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*cfg.clip_fourcc),
                                         fps, out_size)
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    raise RuntimeError(
                        "无法写入 mp4（编码 %s 不可用）。请确认 opencv 带编码支持，"
                        "或先用 ffmpeg 把源视频转成 H.264 mp4 再试。" % cfg.clip_fourcc)
            writer.write(frame)
            produced += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if produced == 0:
        try:
            os.remove(out_path)
        except OSError:
            pass
        return None
    return str(out_path), produced / fps
