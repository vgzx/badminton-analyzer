#!/usr/bin/env python
"""球场四点标定工具。

用法：
    python tools/pick_court.py --input match.mp4 --out court.json
    python tools/pick_court.py --input match.mp4 --at 30 --out court.json

操作：
    鼠标左键  依次点击四个角点
              顺序必须是：近端左角 → 近端右角 → 远端左角 → 远端右角
              （站在底线后面向球网看）
    R         重选
    S         保存并退出（需要 4 个点）
    Q / ESC   不保存退出

为什么要标定：标定后距离才是米制，回合的纵深判断（前/中/后场）才准。
不标定也能跑，但所有距离都是“画面宽度归一化单位”。
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from badminton.court import Court  # noqa: E402
from badminton.cvreq import get_cv2  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="羽毛球球场四点标定")
    parser.add_argument("-i", "--input", required=True, help="视频文件")
    parser.add_argument("-o", "--out", default="court.json", help="输出 JSON")
    parser.add_argument("--at", type=float, default=1.0, help="取第几秒的画面（默认 1s）")
    args = parser.parse_args(argv)

    cv2 = get_cv2()
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("无法打开视频：%s" % args.input, file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, args.at) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("读取画面失败", file=sys.stderr)
        return 2

    points = []
    win = "pick court (L=click, R=reset, S=save, Q=quit)"

    def redraw():
        canvas = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(canvas, pt, 6, (0, 0, 255), -1)
            cv2.putText(canvas, str(i + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(canvas, points[i - 1], points[i], (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(canvas, points[3], points[0], (0, 255, 0), 2)
            # 用四点算出的单应把整个球场画出来，用于验收
            try:
                court = Court.from_points(points, frame.shape[1], frame.shape[0])
                for (xa, ya), (xb, yb) in (
                    ((0.0, 0.0), (1.0, 0.0)),
                    ((0.0, 2.1967), (1.0, 2.1967)),
                    ((0.0, 0.0), (0.0, 2.1967)),
                    ((1.0, 0.0), (1.0, 2.1967)),
                    ((0.5, 0.0), (0.5, 2.1967)),
                    ((0.0, 0.774), (1.0, 0.774)),
                    ((0.0, 1.42), (1.0, 1.42)),
                ):
                    p1 = tuple(int(v) for v in court.to_px(xa, ya))
                    p2 = tuple(int(v) for v in court.to_px(xb, yb))
                    cv2.line(canvas, p1, p2, (255, 200, 0), 1)
            except ValueError as exc:
                cv2.putText(canvas, "degenerate: %s" % exc, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(win, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            print("已取消")
            return 1
        if key == ord("r"):
            points.clear()
            redraw()
        if key == ord("s"):
            if len(points) != 4:
                print("需要 4 个点，当前 %d 个" % len(points))
                continue
            try:
                court = Court.from_points(points, frame.shape[1], frame.shape[0])
            except ValueError as exc:
                print("标定失败：%s" % exc, file=sys.stderr)
                continue
            court.save(args.out)
            cv2.destroyAllWindows()
            print("已保存：%s" % os.path.abspath(args.out))
            print("用法：python run.py --input %s --out out --court %s"
                  % (args.input, args.out))
            return 0


if __name__ == "__main__":
    sys.exit(main())
