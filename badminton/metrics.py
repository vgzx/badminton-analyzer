"""回合指标与全局聚合。

只对**近端球员**做详细姿态/移动统计（远端像素太少，姿态不可靠）。
未标定时所有距离是“画面宽度归一化单位”，报告里必须如实标注。
"""

from __future__ import annotations

import bisect
import math

import numpy as np

from .config import COURT_LENGTH_U, u_to_m

BASE_X = 0.5                 # 中线
BASE_Y = 3.4 / 6.10          # 距底线 3.4 m
FRONT_Y = 4.72 / 6.10        # 前发球线
BACK_Y = 1.6 / 6.10          # 底线前 1.6 m
OWN_HALF_Y = COURT_LENGTH_U / 2.0


def _times(frames):
    return [f.t for f in frames]


def slice_frames(frames, t0, t1, ts=None):
    if not frames:
        return []
    ts = ts if ts is not None else _times(frames)
    lo = bisect.bisect_left(ts, t0)
    hi = bisect.bisect_right(ts, t1)
    return frames[lo:hi]


def _p95(values):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def _mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None
    return float(np.mean(values))


def rally_row(rally, frames, cfg, ts=None):
    """计算单个回合的指标。"""
    seg = slice_frames(frames, rally["t0"], rally["t1"], ts=ts)
    row = {
        "id": rally["id"],
        "t0": rally["t0"],
        "t1": round(rally.get("t1", rally["t0"]), 3),
        "duration": round(rally.get("duration", 0.0), 3),
        "shots": rally.get("shots"),
        "kind": rally.get("kind", "rally"),
        "hit_confidence": rally.get("hit_confidence", "low"),
        "end_reason": rally.get("end_reason", ""),
    }
    if not seg:
        row.update({"distance_u": 0.0, "avg_speed": 0.0, "peak_speed": 0.0,
                    "front_share": None, "mid_share": None, "back_share": None,
                    "base_dist": None, "max_excursion": None, "recovery_s": None,
                    "crouch": None, "crouch_min": None, "stance": None,
                    "racket_up_rate": None, "wrist_peak": None, "pose_frames": 0})
        return row

    dt = np.median([f.dt for f in seg]) if seg else 0.1
    speeds, dist = [], 0.0
    front = mid = back = 0
    base_d, excursions = [], []
    pose_n = 0
    crouch_ready, stance_ready, crouch_strike = [], [], []
    racket_up = [0, 0]
    wrists = []

    hit_times = [h["t"] for h in (rally.get("hits") or [])]

    for frame in seg:
        feat = frame.near
        if not feat.present:
            continue
        speeds.append(feat.speed)
        if feat.speed > 0.015:
            dist += feat.speed * dt
        if feat.y > FRONT_Y:
            front += 1
        elif feat.y < BACK_Y:
            back += 1
        else:
            mid += 1
        d = math.hypot(feat.x - BASE_X, feat.y - BASE_Y)
        base_d.append(d)
        excursions.append(d)
        if feat.pose_ok:
            pose_n += 1
        # 蹲伏：mediapipe 用真关节；代理模式用框高比值（已在 features 里算好）
        if feat.crouch is not None and not math.isnan(feat.crouch):
            if not feat.pose_ok:
                pose_n += 1
            in_strike = any(abs(frame.t - ht) <= 0.35 for ht in hit_times)
            if in_strike:
                crouch_strike.append(feat.crouch)
            elif feat.speed < cfg.still_speed:
                crouch_ready.append(feat.crouch)
        if feat.pose_ok:
            if feat.racket_up is not None:
                racket_up[1] += 1
                racket_up[0] += 1 if feat.racket_up else 0
            if feat.stance is not None and not math.isnan(feat.stance):
                stance_ready.append(feat.stance)
            if feat.wrist_v > 0:
                wrists.append(feat.wrist_v)

    total_pos = front + mid + back
    row["distance_u"] = round(dist, 3)
    row["distance_m"] = round(u_to_m(dist), 2)
    row["avg_speed"] = round(float(np.mean(speeds)), 4) if speeds else 0.0
    row["peak_speed"] = round(_p95(speeds), 4)
    row["front_share"] = round(front / total_pos, 3) if total_pos else None
    row["mid_share"] = round(mid / total_pos, 3) if total_pos else None
    row["back_share"] = round(back / total_pos, 3) if total_pos else None
    row["base_dist"] = round(_mean(base_d), 3) if base_d else None
    row["max_excursion"] = round(max(excursions), 3) if excursions else None
    row["crouch"] = _round(_mean(crouch_ready))
    row["crouch_min"] = _round(min(crouch_strike)) if crouch_strike else None
    row["stance"] = _round(_mean(stance_ready))
    row["racket_up_rate"] = round(racket_up[0] / racket_up[1], 3) if racket_up[1] else None
    row["wrist_peak"] = _round(_p95(wrists)) if wrists else None
    row["pose_frames"] = pose_n
    row["recovery_s"] = _recovery(seg, rally, cfg)
    row["tempo"] = round((row["shots"] or 0) / row["duration"], 2) if row["shots"] else None
    return row


def _round(value, nd=3):
    if value is None:
        return None
    return round(float(value), nd)


def _recovery(seg, rally, cfg):
    """击球后回到静止所需时间（秒），取不到返回 None。"""
    hit_times = [h["t"] for h in (rally.get("hits") or [])]
    if not hit_times:
        return None
    last = hit_times[-1]
    still_run = 0.0
    for frame in seg:
        if frame.t <= last:
            continue
        still_run = still_run + frame.dt if frame.near.speed < cfg.still_speed else 0.0
        if still_run >= 0.3:
            return round(frame.t - last, 2)
        if frame.t - last > 4.0:
            break
    return None


def aggregate(rows, rallies, video_duration, cfg, pose_mode="off", court_mode="relative"):
    active = sum(r["duration"] for r in rows if r["kind"] != "serve_fault")
    durations = [r["duration"] for r in rows if r["kind"] != "serve_fault"]
    shots = [r["shots"] for r in rows if r.get("shots") and r["kind"] != "serve_fault"]
    distances = [r["distance_u"] for r in rows if r.get("distance_u")]

    summary = {
        "n_rallies": len(rows),
        "n_serve_fault": sum(1 for r in rows if r["kind"] == "serve_fault"),
        "video_duration": round(video_duration, 2),
        "active_time": round(active, 2),
        "dead_time": round(max(0.0, video_duration - active), 2),
        "active_ratio": round(active / video_duration, 3) if video_duration else None,
        "avg_duration": round(float(np.mean(durations)), 2) if durations else None,
        "median_duration": round(float(np.median(durations)), 2) if durations else None,
        "max_duration": round(max(durations), 2) if durations else None,
        "avg_shots": round(float(np.mean(shots)), 1) if shots else None,
        "max_shots": max(shots) if shots else None,
        "avg_tempo": round(float(np.mean([r["tempo"] for r in rows if r.get("tempo")])), 2)
        if any(r.get("tempo") for r in rows) else None,
        "total_distance_m": round(u_to_m(sum(distances)), 1) if distances else None,
        "total_distance_u": round(sum(distances), 1) if distances else None,
        "pose_mode": pose_mode,
        "court_mode": court_mode,
    }

    buckets = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 999)]
    dist = []
    for lo, hi in buckets:
        sel = [r for r in rows if r.get("shots") and lo <= r["shots"] <= hi]
        dist.append({
            "label": "%d 拍" % lo if lo == hi else ("%d+ 拍" % lo if hi >= 999 else "%d-%d 拍" % (lo, hi)),
            "count": len(sel),
            "avg_duration": round(float(np.mean([r["duration"] for r in sel])), 2) if sel else None,
            "avg_distance_m": round(u_to_m(float(np.mean([r["distance_u"] for r in sel]))), 1) if sel else None,
            "avg_distance_u": round(float(np.mean([r["distance_u"] for r in sel])), 2) if sel else None,
            "avg_crouch": _round(_mean([r["crouch"] for r in sel if r.get("crouch") is not None])),
            "racket_up_rate": _round(_mean([r["racket_up_rate"] for r in sel
                                            if r.get("racket_up_rate") is not None])),
        })

    windows = _windows(rows, video_duration)
    return {"summary": summary, "shot_buckets": dist, "windows": windows}


def _windows(rows, video_duration, window=300.0):
    """每 5 分钟一个窗口，用来看体能衰减趋势。"""
    if not video_duration or video_duration <= 0:
        return []
    out = []
    start = 0.0
    while start < video_duration:
        end = min(start + window, video_duration)
        sel = [r for r in rows if start <= r["t0"] < end and r["kind"] != "serve_fault"]
        out.append({
            "label": "%d-%d min" % (int(start // 60), int(round(end / 60.0))),
            "rallies": len(sel),
            "avg_duration": round(float(np.mean([r["duration"] for r in sel])), 2) if sel else None,
            "avg_shots": round(float(np.mean([r["shots"] for r in sel if r.get("shots")])), 1)
            if sel and any(r.get("shots") for r in sel) else None,
            "avg_distance_m": round(u_to_m(float(np.mean([r["distance_u"] for r in sel]))), 1)
            if sel else None,
            "avg_distance_u": round(float(np.mean([r["distance_u"] for r in sel])), 2)
            if sel else None,
            "avg_speed": round(float(np.mean([r["avg_speed"] for r in sel])), 3) if sel else None,
            "recovery_s": _round(_mean([r["recovery_s"] for r in sel
                                        if r.get("recovery_s") is not None]), 2),
        })
        start = end
    return out


def consistency(rows):
    """拍数 vs 时长的一致性检查，用来识别算法可疑的回合。"""
    pairs = [(r["shots"], r["duration"]) for r in rows
             if r.get("shots") and r["duration"] > 0]
    if len(pairs) < 5:
        return {"corr": None, "suspicious": []}
    shots = np.asarray([p[0] for p in pairs], dtype=np.float64)
    durs = np.asarray([p[1] for p in pairs], dtype=np.float64)
    corr = float(np.corrcoef(shots, durs)[0, 1]) if shots.std() > 0 and durs.std() > 0 else None
    suspicious = []
    for r in rows:
        if not r.get("shots"):
            continue
        per_shot = r["duration"] / r["shots"]
        if per_shot < 0.45:
            suspicious.append((r["id"], "拍数偏多（每拍 < 0.45s）"))
        elif per_shot > 6.0:
            suspicious.append((r["id"], "拍数偏少（每拍 > 6s）"))
    return {"corr": round(corr, 3) if corr is not None else None, "suspicious": suspicious}
