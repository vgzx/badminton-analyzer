"""编排：视频 -> 帧特征 -> 回合 -> 拍数 -> 指标 -> 片段 -> 报告。"""

from __future__ import annotations

import bisect
import csv
import json
import os
import time

from . import hits as hits_mod
from . import metrics as metrics_mod
from . import report as report_mod
from .config import Config
from .court import Court
from .detect import build_tracker
from .features import FeatureExtractor
from .pose import PoseService
from .rally import RallySegmenter
from .videoio import FrameSampler, crop, probe


def _mkdir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def _progress(msg):
    print(msg, flush=True)


def run(cfg: Config, source: str, out_dir: str, verbose=True) -> dict:
    from .cvreq import get_cv2

    cv2 = get_cv2()
    _mkdir(out_dir)
    _mkdir(os.path.join(out_dir, "clips"))
    _mkdir(os.path.join(out_dir, "debug"))

    t_start = time.time()
    meta = probe(source)
    warnings = []

    # ---------- 标定 ----------
    if cfg.court_json and os.path.exists(cfg.court_json):
        court = Court.load(cfg.court_json, meta.width, meta.height)
    elif cfg.court_json:
        raise FileNotFoundError("标定文件不存在：%s" % cfg.court_json)
    else:
        court = Court.relative(meta.width, meta.height)
        warnings.append("未提供球场标定（--court），距离与速度为画面相对单位，非米制。"
                        "建议用 tools/pick_court.py 标定后重跑。")

    # ---------- 组件 ----------
    tracker = build_tracker(meta, cfg)
    pose = PoseService(cfg)
    if pose.mode == "proxy":
        warnings.append("姿态后端降级为**代理模式**（只有球员框几何）。"
                        "安装 mediapipe 可获得真实关节指标：pip install mediapipe")
    extractor = FeatureExtractor(court, cfg)
    segmenter = RallySegmenter(cfg)
    sampler = FrameSampler(meta, cfg.analysis_fps, cfg.max_seconds)
    eff_fps = sampler.effective_fps

    if verbose:
        _progress("[1/6] 读取视频：%dx%d @ %.2ffps，时长 %s，分析帧率 %.2f"
                  % (meta.width, meta.height, meta.fps,
                     report_mod._t(meta.duration), eff_fps))

    # ---------- 主循环 ----------
    frames = []
    prev_t = None
    pose_counter = 0
    both_present = 0
    total = 0
    last_t = 0.0
    for idx, t, frame in sampler:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        obs = tracker.step(gray)

        pose_res = None
        if pose.available and obs.get("near") is not None:
            if pose_counter % max(1, cfg.pose_every) == 0:
                if pose.mode == "mediapipe":
                    # 只有真姿态需要像素；代理模式直接用框几何，省掉裁剪
                    crop_img, offset = crop(frame, obs["near"].bbox, cfg.pose_crop_scale)
                    if crop_img is not None:
                        pose_res = pose.infer(crop_img, obs["near"].bbox, offset)
                else:
                    pose_res = pose.infer(None, obs["near"].bbox, (0, 0))
            pose_counter += 1

        dt = (t - prev_t) if prev_t is not None else (1.0 / eff_fps)
        feat = extractor.update(t, idx, dt, obs.get("near"), obs.get("far"), pose_res)
        segmenter.update(t, feat)
        frames.append(feat)
        prev_t = t
        last_t = t

        total += 1
        if feat.near.present and feat.far.present:
            both_present += 1
        if verbose and cfg.progress_every and total % cfg.progress_every == 0:
            done = report_mod._t(t)
            _progress("      已处理 %d 帧（%s），当前状态 %s"
                      % (total, done, segmenter.state))

    segmenter.finish()
    rallies = list(segmenter.rallies)
    tracker.close()
    pose.close()

    coverage = both_present / float(total) if total else 0.0
    if verbose:
        _progress("[2/6] 初筛回合 %d 个，双人可见率 %.0f%%" % (len(rallies), coverage * 100))
    if coverage < 0.55:
        warnings.append("双人可见率只有 %.0f%%，回合切分与拍数统计会不可靠。"
                        "请检查机位、光照，或改用 --detector yolo。" % (coverage * 100))

    # ---------- 拍数 ----------
    hcfg = cfg.copy()
    hcfg.analysis_fps = eff_fps
    duration = meta.duration or last_t
    hits_mod.attach_to_rallies(rallies, frames, hcfg, duration=duration)

    # ---------- 指标 ----------
    if verbose:
        _progress("[3/6] 计算指标")
    rows = [metrics_mod.rally_row(r, frames, cfg) for r in rallies]
    agg = metrics_mod.aggregate(rows, rallies, duration, cfg,
                                pose_mode=pose.mode, court_mode=court.mode)
    cons = metrics_mod.consistency(rows)
    low_conf = sum(1 for r in rows if r.get("hit_confidence") == "low")
    if rows and low_conf / len(rows) > 0.4:
        warnings.append("%d/%d 个回合的拍数可信度为低，拍数统计建议只看趋势、不要看绝对值。"
                        % (low_conf, len(rows)))

    # ---------- 片段 ----------
    if cfg.export_clips and rallies:
        if verbose:
            _progress("[4/6] 导出回合片段（%d 个）" % len(rallies))
        times = [f.t for f in frames]
        overlay = None
        if cfg.overlay_clips:
            overlay = _make_overlay(court, frames, times)

        for rally, row in zip(rallies, rows):
            if rally["kind"] != "rally":
                continue
            name = "r%03d_%s.mp4" % (rally["id"], _stamp(rally["t0"]))
            path = os.path.join(out_dir, "clips", name)
            result = None
            try:
                from .clips import export_clip
                result = export_clip(meta, rally["t0"], rally["t1"], path, cfg,
                                     overlay_fn=overlay)
            except RuntimeError as exc:
                warnings.append("片段导出失败：%s" % exc)
                break
            if result:
                row["clip"] = os.path.relpath(result[0], out_dir).replace("\\", "/")
                print("      + %s（%.1fs）" % (name, result[1]), flush=True)
    else:
        if verbose:
            _progress("[4/6] 跳过片段导出")

    # ---------- 报告 ----------
    if verbose:
        _progress("[5/6] 生成报告")
    ctx = {
        "source": source,
        "cfg": cfg.to_dict(),
        "meta": {
            "width": meta.width, "height": meta.height, "fps": meta.fps,
            "duration": duration, "analysis_fps": eff_fps, "step": sampler.step,
        },
        "detector": getattr(tracker, "name", "motion"),
        "pose_mode": pose.mode,
        "court_desc": court.describe(),
        "summary": agg["summary"],
        "buckets": agg["shot_buckets"],
        "windows": agg["windows"],
        "rows": rows,
        "rallies": rallies,
        "consistency": cons,
        "warnings": warnings,
        "coverage": round(coverage, 3),
    }

    rally_md = os.path.join(out_dir, "report_rallies.md")
    pose_md = os.path.join(out_dir, "report_pose.md")
    report_mod.write_rally_report(rally_md, ctx)
    report_mod.write_pose_report(pose_md, ctx)

    # ---------- 结构化产物 ----------
    _write_csv(os.path.join(out_dir, "rallies.csv"), rows)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": agg["summary"], "buckets": agg["shot_buckets"],
                   "windows": agg["windows"], "consistency": cons,
                   "rallies": rows}, fh, ensure_ascii=False, indent=2)
    cfg.save(os.path.join(out_dir, "run_config.json"))
    with open(os.path.join(out_dir, "hits.json"), "w", encoding="utf-8") as fh:
        json.dump([{"id": r["id"], "t0": r["t0"], "t1": r["t1"], "shots": r["shots"],
                    "hits": r.get("hits", [])} for r in rallies], fh,
                  ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    if verbose:
        _progress("[6/6] 完成，用时 %.1fs" % elapsed)
        _progress("      回合 %d 个｜有效时间 %s（%s）｜平均 %s 拍"
                  % (agg["summary"]["n_rallies"],
                     report_mod._t(agg["summary"]["active_time"]),
                     report_mod._pct(agg["summary"]["active_ratio"]),
                     report_mod._f(agg["summary"]["avg_shots"], 1)))
        _progress("      输出目录：%s" % os.path.abspath(out_dir))
    ctx["elapsed"] = round(elapsed, 1)
    return ctx


def _stamp(t):
    t = max(0, int(t))
    return "%03dm%02ds" % (t // 60, t % 60)


def _write_csv(path, rows):
    keys = ["id", "t0", "t1", "duration", "shots", "tempo", "kind", "hit_confidence",
            "distance_u", "distance_m", "avg_speed", "peak_speed", "front_share",
            "mid_share", "back_share", "base_dist", "max_excursion", "recovery_s",
            "crouch", "crouch_min", "stance", "racket_up_rate", "wrist_peak",
            "pose_frames", "end_reason", "clip"]
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_overlay(court, frames, times):
    """返回一个 (frame, t) -> frame 的绘制回调。"""
    from . import viz

    def overlay(frame, t):
        i = bisect.bisect_left(times, t)
        i = min(max(i, 0), len(frames) - 1)
        feat = frames[i]
        try:
            viz.draw_court(frame, court)
            viz.draw_players(frame, feat)
        except Exception:
            pass
        return frame

    return overlay


# ----------------------------------------------------------------------
# 自检：不需要视频，验证切分/拍数/报告整条链路
# ----------------------------------------------------------------------
def selftest(out_dir: str, verbose=True) -> dict:
    from .synthetic import synth_frames

    _mkdir(out_dir)
    frames, truth, hit_truth = synth_frames()
    cfg = Config()
    cfg.export_clips = False
    cfg.court_json = ""
    cfg.pose_backend = "proxy"

    seg = RallySegmenter(cfg)
    for frame in frames:
        seg.update(frame.t, frame)
    seg.finish()

    rallies = list(seg.rallies)
    hits_mod.attach_to_rallies(rallies, frames, cfg,
                               duration=frames[-1].t if frames else 0.0)
    rows = [metrics_mod.rally_row(r, frames, cfg) for r in rallies]
    agg = metrics_mod.aggregate(rows, rallies, frames[-1].t, cfg,
                                pose_mode="mediapipe", court_mode="relative")
    cons = metrics_mod.consistency(rows)

    truth_pairs = [(t["t0"], t["t1"]) for t in truth]
    detected = [(r["t0"], r["t1"]) for r in rallies if r["kind"] != "serve_fault"]
    match = 0
    for t0, t1 in truth_pairs:
        if any(abs(d0 - t0) <= 0.8 and abs(d1 - t1) <= 1.2 for d0, d1 in detected):
            match += 1
    recall = match / float(len(truth_pairs)) if truth_pairs else 0.0

    shot_errors = []
    for rally, ref in zip([r for r in rallies if r["kind"] != "serve_fault"], truth):
        if rally.get("shots") and ref.get("shots"):
            shot_errors.append(abs(rally["shots"] - ref["shots"]))
    mean_err = sum(shot_errors) / len(shot_errors) if shot_errors else None

    warnings = ["这是 `--selftest` 生成的**合成数据**，只用于验证链路是否跑通，"
                "不代表真实视频精度。"]
    ctx = {
        "source": "<synthetic>",
        "cfg": cfg.to_dict(),
        "meta": {"width": 1920, "height": 1080, "fps": 30.0,
                 "duration": frames[-1].t, "analysis_fps": cfg.analysis_fps, "step": 3},
        "detector": "synthetic",
        "pose_mode": "mediapipe",
        "court_desc": "合成数据：相对单位",
        "summary": agg["summary"],
        "buckets": agg["shot_buckets"],
        "windows": agg["windows"],
        "rows": rows,
        "rallies": rallies,
        "consistency": cons,
        "warnings": warnings,
        "coverage": 1.0,
    }
    report_mod.write_rally_report(os.path.join(out_dir, "report_rallies.md"), ctx)
    report_mod.write_pose_report(os.path.join(out_dir, "report_pose.md"), ctx)
    _write_csv(os.path.join(out_dir, "rallies.csv"), rows)
    cfg.save(os.path.join(out_dir, "run_config.json"))

    result = {
        "truth_rallies": len(truth_pairs),
        "detected_rallies": len(detected),
        "match_within_tolerance": match,
        "recall": round(recall, 3),
        "mean_abs_shot_error": None if mean_err is None else round(mean_err, 2),
        "out_dir": os.path.abspath(out_dir),
    }
    if verbose:
        _progress("自检结果：真值 %d 个回合，检出 %d 个，容差内匹配 %d 个（召回 %.0f%%），"
                  "拍数平均误差 %s"
                  % (result["truth_rallies"], result["detected_rallies"],
                     result["match_within_tolerance"], recall * 100,
                     report_mod._f(mean_err, 2)))
        _progress("报告已写入：%s" % result["out_dir"])
    return result
