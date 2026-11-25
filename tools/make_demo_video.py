#!/usr/bin/env python
"""生成一段合成羽毛球视频，用来在没有真实比赛素材时验证整条视频链路。

    python tools/make_demo_video.py --out demo.mp4 --seconds 80

画面：浅色背景 + 透视球场线 + 两个深色“球员”矩形，节奏是
    发球准备 1.4s → 活球 Ns → 静止 1.8s → 走动捡球 1.2s
与 badminton/synthetic.py 的计划一致，方便对照。

用途：
    python tools/make_demo_video.py --out demo.mp4
    python run.py --input demo.mp4 --out out_demo
然后看 out_demo/report_rallies.md 是否能数出 6 个回合。
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from badminton.cvreq import get_cv2  # noqa: E402

PLAN = [(6.0, 1.30), (12.0, 1.05), (4.0, 1.60), (3.0, 2.40), (9.0, 1.20), (18.0, 0.95)]
W, H = 1280, 720
NEAR_BASE_Y, FAR_BASE_Y = 700.0, 120.0
NEAR_X = (150.0, 1130.0)
FAR_X = (380.0, 900.0)


def court_to_px(x_u, y_u):
    t = max(0.0, min(2.1967, y_u)) / 2.1967
    y = NEAR_BASE_Y + (FAR_BASE_Y - NEAR_BASE_Y) * t
    left = NEAR_X[0] + (FAR_X[0] - NEAR_X[0]) * t
    right = NEAR_X[1] + (FAR_X[1] - NEAR_X[1]) * t
    x = left + (right - left) * max(0.0, min(1.0, x_u))
    return int(round(x)), int(round(y))


def draw_court(cv2, frame):
    for (xa, ya), (xb, yb) in (
        ((0.0, 0.0), (1.0, 0.0)),      # 近底线
        ((0.0, 2.1967), (1.0, 2.1967)),  # 远底线
        ((0.0, 0.0), (0.0, 2.1967)),   # 左边线
        ((1.0, 0.0), (1.0, 2.1967)),   # 右边线
        ((0.5, 0.0), (0.5, 2.1967)),   # 中线
        ((0.0, 0.774), (1.0, 0.774)),  # 近前发球线
        ((0.0, 1.42), (1.0, 1.42)),    # 远前发球线
    ):
        cv2.line(frame, court_to_px(xa, ya), court_to_px(xb, yb), (90, 90, 90), 2, cv2.LINE_AA)
    # 网
    cv2.line(frame, court_to_px(0.0, 1.098), court_to_px(1.0, 1.098), (40, 40, 40), 4)
    return frame


def draw_player(cv2, frame, x_u, y_u, height_px, color):
    cx, cy = court_to_px(x_u, y_u)
    half_w = max(10, int(height_px * 0.19))
    top = cy - height_px
    cv2.rectangle(frame, (cx - half_w, top), (cx + half_w, cy), color, -1)
    # 头
    cv2.circle(frame, (cx, top - int(half_w * 0.7)), int(half_w * 0.8), color, -1)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成合成羽毛球视频")
    parser.add_argument("-o", "--out", default="demo.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help=">0 时截断到该时长")
    args = parser.parse_args(argv)

    cv2 = get_cv2()
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, H))
    if not writer.isOpened():
        print("无法创建视频文件（mp4v 编码不可用）", file=sys.stderr)
        return 2

    fps = args.fps
    t = 0.0
    expected = []

    def emit(near_xy, far_xy, near_h, far_h):
        nonlocal t
        frame = np.full((H, W, 3), 205, dtype=np.uint8)
        frame = draw_court(cv2, frame)
        # 轻微抖动，避免 MOG2 把静止的人吸收进背景
        jx = 0.0015 * math.sin(t * 7.0)
        draw_player(cv2, frame, near_xy[0] + jx, near_xy[1], near_h, (35, 40, 45))
        draw_player(cv2, frame, far_xy[0] - jx, far_xy[1], far_h, (45, 50, 60))
        cv2.putText(frame, "t=%05.1fs" % t, (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (20, 20, 20), 2, cv2.LINE_AA)
        writer.write(frame)
        t += 1.0 / fps

    for live_dur, hit_gap in PLAN:
        for _ in range(int(1.4 * fps)):
            emit((0.52, 0.45), (0.48, 1.80), 150, 96)
        start = t
        for k in range(int(live_dur * fps)):
            lt = k / fps
            phase = math.sin(2 * math.pi * lt / 2.2)
            emit((0.50 + 0.30 * phase, 0.55 + 0.28 * math.cos(2 * math.pi * lt / 2.2)),
                 (0.50 - 0.26 * phase, 1.60 + 0.26 * math.cos(2 * math.pi * lt / 2.6)),
                 150, 96)
        for _ in range(int(1.8 * fps)):
            emit((0.52, 0.62), (0.48, 1.72), 150, 96)
        for _ in range(int(1.2 * fps)):
            emit((0.52, 0.20), (0.48, 1.30), 150, 96)
        expected.append((start, start + live_dur, live_dur, hit_gap))
        if args.seconds and t >= args.seconds:
            break

    writer.release()
    print("已生成：%s（%.1fs）" % (os.path.abspath(args.out), t))
    print("预期回合（起点秒, 终点秒, 时长, 拍间隔）：")
    for item in expected:
        print("  %.1f -> %.1f  (%.1fs, 每 %.2fs 一拍)" % item)
    print("\n下一步：python run.py --input %s --out out_demo" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
