"""叠加层绘制（脚点、轨迹、球场线、状态条）。

只有真正要出视频时才调用，模块本身不 import cv2。
"""

from __future__ import annotations

from .court import COURT_CORNERS_M
from .cvreq import get_cv2

NEAR_COLOR = (60, 60, 255)     # BGR 红
FAR_COLOR = (255, 140, 60)     # BGR 蓝
LINE_COLOR = (140, 220, 140)


def draw_court(frame, court, thickness=2):
    import numpy as np

    cv2 = get_cv2()
    pts = []
    for idx in [0, 1, 3, 2]:
        x_m, y_m = COURT_CORNERS_M[idx]
        pts.append(court.to_px(x_m / 6.10, y_m / 6.10))
    arr = np.asarray(pts, dtype=np.int32)
    cv2.polylines(frame, [arr], True, LINE_COLOR, thickness, cv2.LINE_AA)
    # 中线与前发球线
    for (xa, ya), (xb, yb) in (
        ((0.5, 0.0), (0.5, 2.1967)),
        ((0.0, 0.774), (1.0, 0.774)),
        ((0.0, 1.42), (1.0, 1.42)),
    ):
        p1 = tuple(int(v) for v in court.to_px(xa, ya))
        p2 = tuple(int(v) for v in court.to_px(xb, yb))
        cv2.line(frame, p1, p2, LINE_COLOR, 1, cv2.LINE_AA)
    return frame


def draw_players(frame, feat, trail=None):
    cv2 = get_cv2()
    for side, color in (("near", NEAR_COLOR), ("far", FAR_COLOR)):
        player = getattr(feat, side, None)
        if player is None or not player.present:
            continue
        try:
            px, py = int(player.px), int(player.py)
        except (TypeError, ValueError):
            continue
        if not (0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]):
            continue
        cv2.circle(frame, (px, py), 7, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 2, color, -1, cv2.LINE_AA)
        label = "near %.2f u/s" % player.speed if side == "near" else "far"
        cv2.putText(frame, label, (px + 9, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
        if trail is not None:
            trail.append((px, py))
            if len(trail) > 24:
                trail.pop(0)
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], color, 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, t, state, rally_id=None, shots=None, hits=None):
    cv2 = get_cv2()
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (20, 20, 20), -1)
    text = "t=%s  state=%s" % (_mmss(t), state)
    if rally_id is not None:
        text += "  rally#%d" % rally_id
    if shots:
        text += "  shots=%s" % shots
    cv2.putText(frame, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)
    if hits:
        cv2.circle(frame, (w - 24, 17), 8, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "HIT", (w - 70, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 220, 255), 1, cv2.LINE_AA)
    return frame


def _mmss(t):
    t = max(0.0, float(t))
    return "%02d:%05.2f" % (int(t // 60), t % 60)
