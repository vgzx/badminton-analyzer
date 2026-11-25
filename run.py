#!/usr/bin/env python
"""命令行入口。

示例
----
    # 1) 先跑自检，确认环境与依赖没问题（不需要视频）
    python run.py --selftest --out out_selftest

    # 2) 完整分析
    python run.py --input match.mp4 --out out

    # 3) 有标定文件时（推荐）
    python run.py --input match.mp4 --out out --court court.json

    # 4) 只跑前 3 分钟快速摸底
    python run.py --input match.mp4 --out out_fast --max-seconds 180 --no-clips
"""

from __future__ import annotations

import argparse
import os
import sys

from badminton import pipeline
from badminton.config import Config


def build_parser():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="离线羽毛球视频分析：回合切分 + 拍数统计 + 近端姿态 + 报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-i", "--input", help="输入视频路径")
    parser.add_argument("-o", "--out", default="out", help="输出目录（默认 out）")
    parser.add_argument("--court", help="球场四点标定 JSON（tools/pick_court.py 生成）")
    parser.add_argument("--pose", default=None,
                        choices=["auto", "mediapipe", "proxy", "off"],
                        help="姿态后端，默认 auto（有 mediapipe 就用）")
    parser.add_argument("--detector", default=None, choices=["motion", "yolo"],
                        help="球员检测后端，默认 motion（纯前景，无需模型）")
    parser.add_argument("--fps", type=float, default=None, help="分析帧率，默认 10")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="只分析前 N 秒，用于快速摸底")
    parser.add_argument("--no-clips", action="store_true", help="不导出回合片段")
    parser.add_argument("--overlay", action="store_true", help="片段叠加脚点/球场线")
    parser.add_argument("--config", help="从 JSON 读取参数（命令行参数优先级更高）")
    parser.add_argument("--save-config", help="把本次参数保存为 JSON 后退出")
    parser.add_argument("--selftest", action="store_true",
                        help="用合成数据跑通整条链路，不需要视频与 OpenCV")
    parser.add_argument("--quiet", action="store_true", help="不打印进度")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    cfg = cfg.merge(
        court_json=args.court,
        pose_backend=args.pose,
        detector=args.detector,
        analysis_fps=args.fps,
        max_seconds=args.max_seconds,
        overlay_clips=True if args.overlay else None,
        export_clips=False if args.no_clips else None,
    )

    if args.save_config:
        cfg.save(args.save_config)
        print("参数已保存：%s" % os.path.abspath(args.save_config))
        return 0

    verbose = not args.quiet

    if args.selftest:
        try:
            result = pipeline.selftest(args.out, verbose=verbose)
        except Exception as exc:  # pragma: no cover
            print("[失败] 自检异常：%s" % exc, file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
        if result["recall"] < 0.8:
            print("[警告] 自检召回率偏低（%.0f%%），逻辑可能需要检查"
                  % (result["recall"] * 100), file=sys.stderr)
            return 3
        return 0

    if not args.input:
        print("[失败] 需要 --input 指定视频，或使用 --selftest 先自检", file=sys.stderr)
        return 2
    if not os.path.exists(args.input):
        print("[失败] 视频不存在：%s" % args.input, file=sys.stderr)
        return 2

    try:
        pipeline.run(cfg, args.input, args.out, verbose=verbose)
    except RuntimeError as exc:
        print("\n[失败] %s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[中断] 用户取消", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
