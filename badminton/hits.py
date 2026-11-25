"""挥拍事件检测与拍数统计。

为什么不用时长估计拍数：同样 14 秒的回合，多拍高远球可能只有 3~4 拍，
平抽快挡可能有 15 拍。时长与拍数的相关系数大约只有 0.6，不能互相替代。

做法
----
1. 每个球员有一组“挥拍分量”（加速度、动作能量突变、手腕速度、身体高度变化率），
   各自做**稳健 z 分数**（中位数 / MAD）后加权求和，得到挥拍置信分。
2. 取局部极大值作为候选点。
3. 用**交替约束**（同一人不可能连击两拍）+ **生理间隔约束**
   （0.35s ~ 8s）做动态规划解码，得到合法击球序列。

第 3 步是把单帧 70% 左右的准确率抬到序列级 85%+ 的关键。
"""

from __future__ import annotations

import numpy as np

WEIGHTS = {
    "accel": 1.0,
    "energy_hp": 1.0,
    "wrist_v": 1.2,
    "height_rate": 0.4,
}

SIDE_ORDER = ("near", "far")


def robust_z(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 1e-9:
        std = float(arr.std())
        scale = std if std > 1e-9 else 1.0
    else:
        scale = 1.4826 * mad
    return (arr - med) / scale


def moving_average(arr, window):
    arr = np.asarray(arr, dtype=np.float64)
    window = int(max(1, window))
    if window <= 1 or arr.size < window:
        return arr
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode="same")


def swing_scores(frames, side, cfg):
    """按帧返回该侧的挥拍置信分（已平滑、已归一）。"""
    if not frames:
        return np.zeros(0)
    comps = {name: [] for name in WEIGHTS}
    valid = []
    for frame in frames:
        feat = getattr(frame, side)
        ok = feat.present and bool(feat.comps)
        valid.append(1.0 if ok else 0.0)
        for name in WEIGHTS:
            comps[name].append(float(feat.comps.get(name, 0.0)) if ok else 0.0)

    total = np.zeros(len(frames), dtype=np.float64)
    for name, weight in WEIGHTS.items():
        series = np.asarray(comps[name], dtype=np.float64)
        if series.max() - series.min() < 1e-12:
            continue
        total += weight * robust_z(series)
    window = max(1, int(round(cfg.hit_smooth_sec * max(cfg.analysis_fps, 1.0))))
    total = moving_average(total, window)
    total *= np.asarray(valid)
    return total


def find_candidates(times, scores, side, cfg, min_sep=None):
    """局部极大值 + 阈值 + 最小间隔抖动合并。"""
    cands = []
    n = len(scores)
    if n < 3:
        return cands
    min_sep = min_sep if min_sep is not None else 0.28
    for i in range(1, n - 1):
        value = scores[i]
        if value < cfg.hit_z_threshold:
            continue
        if not (value >= scores[i - 1] and value > scores[i + 1]):
            continue
        t = times[i]
        if cands and t - cands[-1]["t"] < min_sep:
            if value > cands[-1]["score"]:
                cands[-1] = {"t": t, "side": side, "score": float(value)}
            continue
        cands.append({"t": t, "side": side, "score": float(value)})
    return cands


def decode(cands, cfg):
    """交替 + 间隔约束下求解最优击球序列（一维 DP，O(n^2)）。"""
    if not cands:
        return []
    seq = sorted(cands, key=lambda c: c["t"])
    n = len(seq)
    best = [0.0] * n
    prev = [-1] * n
    lam = cfg.hit_gap_penalty
    typical = max(cfg.hit_typical_gap, 1e-3)

    for i in range(n):
        best[i] = seq[i]["score"]
        for j in range(i):
            gap = seq[i]["t"] - seq[j]["t"]
            if gap < cfg.hit_gap_min or gap > cfg.hit_gap_max:
                continue
            if seq[j]["side"] == seq[i]["side"]:
                continue
            penalty = lam * abs(gap - typical) / typical
            value = best[j] + seq[i]["score"] - penalty
            if value > best[i]:
                best[i] = value
                prev[i] = j

    end = int(np.argmax(best))
    path = []
    while end >= 0:
        path.append(seq[end])
        end = prev[end]
    path.reverse()
    return path


def attach_to_rallies(rallies, frames, cfg, duration=0.0):
    """把击球事件挂到回合上，并统计拍数、必要时向后延长回合边界。

    拍数口径：**发球计为第 1 拍**，之后每个检测到的击球 +1。
    """
    if not frames:
        return rallies
    times = [f.t for f in frames]
    per_side = {side: swing_scores(frames, side, cfg) for side in SIDE_ORDER}

    cands = []
    for side in SIDE_ORDER:
        cands.extend(find_candidates(times, per_side[side], side, cfg))
    selected = decode(cands, cfg)

    for rally in rallies:
        t0, t1 = rally["t0"], rally["t1"]
        hits = [h for h in selected if t0 - 0.3 <= h["t"] <= t1 + 1.5]
        rally["hits"] = [{"t": round(h["t"], 2), "side": h["side"],
                          "score": round(h["score"], 2)} for h in hits]
        dur = rally["duration"]
        if hits:
            rally["shots"] = 1 + len(hits)
            rally["last_hit_t"] = round(hits[-1]["t"], 2)
            new_end = hits[-1]["t"] + 0.8
            if new_end > t1 and (duration == 0.0 or new_end <= duration):
                rally["t1"] = round(new_end, 3)
                rally["duration"] = round(rally["t1"] - t0, 3)
                rally["extended"] = True
        else:
            # 没有任何击球：短回合很可能是发球失误/ACE，长回合则是漏检
            rally["shots"] = 1 if dur < 3.0 else None
            rally["last_hit_t"] = None
        rally["hit_confidence"] = _confidence(rally)
    return rallies


def _confidence(rally):
    """拍数可信度：有击球且间隔合理 -> high；无击球但回合很短 -> medium。"""
    hits = rally.get("hits") or []
    if not hits:
        return "medium" if rally.get("duration", 0) < 3.0 else "low"
    if len(hits) == 1 and rally["duration"] > 6.0:
        return "low"
    gaps = [hits[i + 1]["t"] - hits[i]["t"] for i in range(len(hits) - 1)]
    if gaps and max(gaps) > 5.0:
        return "medium"
    return "high"
