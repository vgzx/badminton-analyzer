"""Markdown 报告生成：回合描述文档 + 姿态分析文档。

写报告的三条自我约束：
1. 每个数字都能在 CSV/片段里核对，所以全部带时间戳；
2. 数据缺失时写“—”，不猜、不四舍五入成好看的整数；
3. 明确写出“这份报告的结论建立在什么假设上”，尤其是未标定/无球轨迹时。
"""

from __future__ import annotations

import datetime
import os

DASH = "—"


# ----------------------------------------------------------------------
# 格式化
# ----------------------------------------------------------------------
def _t(sec):
    if sec is None:
        return DASH
    sec = max(0.0, float(sec))
    return "%d:%05.2f" % (int(sec // 60), sec % 60)


def _f(value, nd=2, suffix=""):
    if value is None:
        return DASH
    try:
        return ("%." + str(nd) + "f%s") % (float(value), suffix)
    except (TypeError, ValueError):
        return str(value)


def _pct(value, nd=1):
    if value is None:
        return DASH
    return "%.*f%%" % (nd, float(value) * 100.0)


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _kv_table(pairs):
    return _table(["项目", "值"], [[k, v] for k, v in pairs])


def _is_calibrated(summary):
    return (summary or {}).get("court_mode") == "calibrated"


def _dist(row, calibrated):
    """按标定与否选择米制还是球场单位，绝不把相对单位标成米。"""
    return row.get("distance_m") if calibrated else row.get("distance_u")


def _bucket_dist(bucket, calibrated):
    return bucket.get("avg_distance_m") if calibrated else bucket.get("avg_distance_u")


def _window_dist(window, calibrated):
    return window.get("avg_distance_m") if calibrated else window.get("avg_distance_u")


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _warn_block(warnings):
    if not warnings:
        return ""
    lines = ["", "> **本次运行的注意事项**", ">"]
    for w in warnings:
        lines.append("> - %s" % w)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 报告一：回合描述
# ----------------------------------------------------------------------
def write_rally_report(path, ctx):
    cfg = ctx["cfg"]
    summary = ctx["summary"]
    rows = ctx["rows"]
    warnings = ctx.get("warnings") or []
    meta = ctx["meta"]

    lines = []
    lines.append("# 羽毛球回合分析报告")
    lines.append("")
    lines.append("> 生成时间：%s　|　源视频：`%s`" % (_now(), ctx.get("source", "")))
    lines.append("")

    lines.append("## 1. 运行概况")
    lines.append("")
    lines.append(_kv_table([
        ("视频分辨率 / 帧率", "%dx%d @ %.2f fps" % (meta["width"], meta["height"], meta["fps"])),
        ("视频时长", _t(meta["duration"])),
        ("分析帧率", "%.2f fps（抽帧步长 %d）" % (meta["analysis_fps"], meta["step"])),
        ("球员检测后端", ctx["detector"]),
        ("姿态后端", ctx["pose_mode"]),
        ("球场标定", ctx["court_desc"]),
        ("回合判定阈值（静止/走动）",
         "%.2f / %.2f u/s" % (cfg["still_speed"], cfg["walk_on_speed"])),
    ]))
    lines.append("")

    if summary["n_rallies"] == 0:
        lines.append("## 2. 结论")
        lines.append("")
        lines.append("本次没有识别出任何有效回合。请按以下顺序排查：")
        lines.append("")
        lines.append("1. 视频是否是**固定机位**、整场都在画面内；")
        lines.append("2. 用 `--max-seconds 120` 只跑前两分钟，看 `out/debug/` 下的检测预览；")
        lines.append("3. 如果画面里有大量走动的人，提高 `--detector-min-height` 或改用 yolo 后端；")
        lines.append("4. 若球员几乎不动（多球训练、单练），回合状态机不适用，"
                     "应改用固定节奏切分。")
        lines.append("")
        lines += _warn_lines(warnings)
        _write(path, lines)
        return path

    lines.append("## 2. 总览")
    lines.append("")
    calibrated = _is_calibrated(summary)
    unit = "m" if calibrated else "u"
    total_dist = (summary["total_distance_m"] if calibrated
                  else summary.get("total_distance_u"))
    lines.append(_kv_table([
        ("识别回合数", summary["n_rallies"]),
        ("其中发球失误/极短回合", summary["n_serve_fault"]),
        ("有效比赛时间", _t(summary["active_time"])),
        ("无效时间（捡球、走位、停顿）", _t(summary["dead_time"])),
        ("**有效时间占比**", _pct(summary["active_ratio"])),
        ("平均回合时长 / 中位数", "%s / %s" % (_f(summary["avg_duration"], 2, " s"),
                                          _f(summary["median_duration"], 2, " s"))),
        ("最长回合", _f(summary["max_duration"], 2, " s")),
        ("平均拍数 / 最多拍数", "%s / %s" % (_f(summary["avg_shots"], 1), summary["max_shots"])),
        ("平均节奏（拍/秒）", _f(summary["avg_tempo"], 2)),
        ("近端球员累计跑动", "%s %s" % (_f(total_dist, 1), unit)),
    ]))
    lines.append("")
    lines.append("> 有效时间占比是这套流程最直接的价值指标：一场 60 分钟的视频，"
                 "真正在打球的时间往往只有 15~20 分钟。")
    lines.append("")

    # ---- 回合列表 ----
    lines.append("## 3. 回合列表")
    lines.append("")
    header = ["#", "开始", "结束", "时长(s)", "拍数", "节奏", "跑动(%s)" % unit,
              "峰值速度(%s/s)" % unit, "前/中/后场", "可信度", "结束原因"]
    body = []
    for row in rows:
        body.append([
            row["id"],
            _t(row["t0"]),
            _t(row["t1"]),
            _f(row["duration"], 2),
            row["shots"] if row["shots"] is not None else DASH,
            _f(row.get("tempo"), 2),
            _f(_dist(row, calibrated), 1),
            _f(row.get("peak_speed"), 2),
            "%s/%s/%s" % (_pct(row.get("front_share"), 0),
                          _pct(row.get("mid_share"), 0),
                          _pct(row.get("back_share"), 0)),
            {"high": "高", "medium": "中", "low": "低"}.get(row.get("hit_confidence"), DASH),
            row.get("end_reason", ""),
        ])
    lines.append(_table(header, body))
    lines.append("")
    lines.append("拍数口径：**发球计为第 1 拍**。可信度低表示该回合击球事件与时长不自洽"
                 "（可能漏检或误检），做统计时建议剔除。")
    lines.append("")

    # ---- 分布 ----
    lines.append("## 4. 拍数分布")
    lines.append("")
    lines.append(_table(
        ["拍数区间", "回合数", "平均时长(s)", "平均跑动(%s)" % unit, "平均蹲伏", "持拍手抬起率"],
        [[b["label"], b["count"], _f(b["avg_duration"], 2),
          _f(_bucket_dist(b, calibrated), 1),
          _f(b["avg_crouch"], 2), _pct(b["racket_up_rate"], 0)]
         for b in ctx["buckets"]]))
    lines.append("")
    longest = max(ctx["buckets"], key=lambda b: (b["avg_duration"] or 0)) if ctx["buckets"] else None
    if longest and longest["count"]:
        lines.append("读法：`%s` 这一档平均最长，说明你的长回合主要靠**多拍相持**堆出来的；"
                     "如果它的平均跑动明显高于短回合，则长回合更耗体能。" % longest["label"])
        lines.append("")

    # ---- 时间趋势 ----
    if ctx["windows"]:
        lines.append("## 5. 时间趋势（每 5 分钟）")
        lines.append("")
        lines.append(_table(
            ["时间段", "回合数", "平均时长(s)", "平均拍数", "平均跑动(%s)" % unit,
             "平均速度(%s/s)" % unit, "回位时间(s)"],
            [[w["label"], w["rallies"], _f(w["avg_duration"], 2), _f(w["avg_shots"], 1),
              _f(_window_dist(w, calibrated), 1), _f(w["avg_speed"], 3),
              _f(w["recovery_s"], 2)]
             for w in ctx["windows"]]))
        lines.append("")
        trend = _trend_text(ctx["windows"], calibrated)
        if trend:
            lines.append(trend)
            lines.append("")

    # ---- 重点回合 ----
    lines.append("## 6. 重点回合")
    lines.append("")
    chosen = _pick_highlights(rows, limit=4)
    if not chosen:
        lines.append("（无）")
        lines.append("")
    for row in chosen:
        lines.append("### 回合 #%d　%s – %s（%s，%s 拍）" % (
            row["id"], _t(row["t0"]), _t(row["t1"]),
            _f(row["duration"], 2, " s"),
            row["shots"] if row["shots"] is not None else "?"))
        lines.append("")
        lines.append("- 近端跑动 %s %s，平均速度 %s %s/s，峰值 %s %s/s" % (
            _f(_dist(row, calibrated), 1), unit,
            _f(row.get("avg_speed"), 2), unit, _f(row.get("peak_speed"), 2), unit))
        lines.append("- 位置分布：前场 %s，中场 %s，后场 %s" % (
            _pct(row.get("front_share"), 0), _pct(row.get("mid_share"), 0),
            _pct(row.get("back_share"), 0)))
        lines.append("- 离基点平均 %s u，最远 %s u；击球后回位 %s s" % (
            _f(row.get("base_dist"), 2), _f(row.get("max_excursion"), 2),
            _f(row.get("recovery_s"), 2)))
        if row.get("crouch") is not None:
            lines.append("- 蹲伏度 %s（击球瞬间最深 %s），站距比 %s，持拍手抬起率 %s" % (
                _f(row.get("crouch"), 2), _f(row.get("crouch_min"), 2),
                _f(row.get("stance"), 2), _pct(row.get("racket_up_rate"), 0)))
        hits = next((r.get("hits") for r in ctx["rallies"] if r["id"] == row["id"]), None) or []
        if hits:
            timeline = "、".join("%s %s" % (_t(h["t"]), "近" if h["side"] == "near" else "远")
                                for h in hits[:12])
            lines.append("- 击球时间线：%s%s" % (timeline, " …" if len(hits) > 12 else ""))
        lines.append("- 结束原因：%s" % row.get("end_reason", DASH))
        lines.append("")
        if row.get("clip"):
            lines.append("片段：`%s`" % row["clip"])
            lines.append("")

    # ---- 一致性 ----
    lines.append("## 7. 一致性检查")
    lines.append("")
    cons = ctx.get("consistency") or {}
    lines.append("- 拍数与时长相关系数：%s（正常应在 0.5~0.85；过低说明拍数检测不稳）"
                 % _f(cons.get("corr"), 3))
    suspicious = cons.get("suspicious") or []
    if suspicious:
        lines.append("- 可疑回合：")
        for rid, reason in suspicious[:10]:
            lines.append("  - #%d：%s" % (rid, reason))
    else:
        lines.append("- 未发现明显不自洽的回合")
    lines.append("")

    # ---- 方法与局限 ----
    lines.append("## 8. 方法与局限")
    lines.append("")
    lines.append("**回合判定**：双方在发球区静止 ≥ %.1fs 视为发球准备；任一方速度超过 "
                 "%.2f u/s 进入活球；双方同时静止 ≥ %.1fs 只进入“待定”，"
                 "必须再出现持续 %.2fs 的走动（捡球/回位）才确认死球。"
                 % (cfg["serve_still_duration"], cfg["move_on_speed"],
                    cfg["quiet_duration"], cfg["walk_duration"]))
    lines.append("")
    lines.append("**拍数判定**：不是按时长估算。对每位球员计算挥拍置信分"
                 "（加速度 + 动作能量突变 + 手腕速度 + 身体高度变化率，各自做稳健 z 分数后加权），"
                 "取局部极大值作为候选，再用“同一个人不能连击两拍”的交替约束和 "
                 "%.2f~%.1f s 的生理间隔约束做动态规划解码。"
                 % (cfg["hit_gap_min"], cfg["hit_gap_max"]))
    lines.append("")
    lines.append("**已知局限**：")
    lines.append("")
    lines.append("1. 不使用球轨迹，因此**没有落点、没有球路**，也无法判断谁得分；")
    lines.append("2. 单打、固定机位（底线后）效果最好；双打需改成按半场聚合；")
    lines.append("3. 多球训练没有“发球准备”段，状态机不会启动；")
    lines.append("4. 长高远球对拉时双方都会静止，靠“待定态 + 走动判据”避免误切，"
                 "但极端情况下仍可能少切一刀；")
    lines.append("5. 未标定时距离是相对单位，不能当米制结论使用。")
    lines.append("")

    lines.append("## 9. 如何核对")
    lines.append("")
    lines.append("1. 看 `out/clips/` 里每个片段的首尾：片段开始应该是发球动作，"
                 "结束应该是球落地后 0.5~1s；")
    lines.append("2. 抽 5 个“可信度=低”的回合，数一下实际拍数，和表格对比；")
    lines.append("3. 第 7 节的可疑回合逐个人工确认；")
    lines.append("4. 用 `--max-seconds 180` 做一次 3 分钟回归，改动参数后误差不应变大。")
    lines.append("")

    lines += _warn_lines(warnings)
    _write(path, lines)
    return path


# ----------------------------------------------------------------------
# 报告二：姿态分析
# ----------------------------------------------------------------------
def write_pose_report(path, ctx):
    cfg = ctx["cfg"]
    rows = ctx["rows"]
    mode = ctx["pose_mode"]
    summary = ctx["summary"]

    lines = ["# 近端球员姿态分析报告", ""]
    lines.append("> 生成时间：%s　|　源视频：`%s`" % (_now(), ctx.get("source", "")))
    lines.append("")

    lines.append("## 1. 数据来源")
    lines.append("")
    if mode == "off":
        lines += [
            "本次**未启用姿态分析**（`--pose off`）。", "",
            "启用方式：`pip install mediapipe` 后使用 `--pose auto`。", "",
        ]
        _write(path, lines)
        return path

    pose_rows = [r for r in rows if r.get("pose_frames")]
    total_pose_frames = sum(r.get("pose_frames") or 0 for r in pose_rows)
    lines.append(_kv_table([
        ("姿态后端", "MediaPipe Pose（33 关键点）" if mode == "mediapipe"
         else "几何代理（**非真实姿态**，仅框体几何）"),
        ("姿态抽样", "每 %d 个分析帧一次" % cfg["pose_every"]),
        ("有姿态数据的回合", "%d / %d" % (len(pose_rows), len(rows))),
        ("累计姿态帧数", total_pose_frames),
    ]))
    lines.append("")
    if mode != "mediapipe":
        lines += [
            "> **重要**：当前用的是**代理模式**——只有球员框的高度与宽度变化，"
            "没有关节坐标。因此本报告只包含“蹲伏代理”和“身高变化”，"
            "**没有**站距、持拍手、手腕速度、肘角等指标。",
            "> 想要真正的姿态指标，请安装 mediapipe：`pip install mediapipe`。", "",
        ]

    lines.append("## 2. 相位定义")
    lines.append("")
    lines.append(_table(
        ["相位", "判定", "看什么"],
        [["击球（STRIKE）", "以每个检测到的击球时刻为中心 ±0.35s", "蹲伏是否到位、手腕速度峰值"],
         ["准备（READY）", "回合内速度 < 静止阈值且不在击球窗", "持拍手抬不抬、站距、重心"],
         ["移动（MOVE）", "其余活球帧", "位移与速度（见回合报告）"]],
    ))
    lines.append("")

    lines.append("## 3. 全局姿态指标")
    lines.append("")
    valid = [r for r in rows if r.get("crouch") is not None or r.get("racket_up_rate") is not None]
    if not valid:
        lines.append("本次没有可用的姿态指标（可能是球员框过小或遮挡过多）。")
        lines.append("")
    else:
        def mean_of(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        pairs = [
            ("平均蹲伏度", _f(mean_of("crouch"), 2)),
            ("击球瞬间最深蹲伏", _f(mean_of("crouch_min"), 2)),
            ("平均站距比（踝距/肩宽）", _f(mean_of("stance"), 2)),
            ("持拍手抬起率", _pct(mean_of("racket_up_rate"), 1)),
            ("手腕速度峰值（中位）", _f(_median([r["wrist_peak"] for r in rows
                                              if r.get("wrist_peak") is not None]), 2)),
            ("击球后回位时间（中位）", _f(_median([r["recovery_s"] for r in rows
                                              if r.get("recovery_s") is not None]), 2, " s")),
        ]
        lines.append(_kv_table(pairs))
        lines.append("")
        lines.append("指标含义：")
        lines.append("")
        if mode == "mediapipe":
            lines.append("- **蹲伏度** = (踝中点 - 髋中点) / 躯干长。直立约 1.7~2.1，下蹲变小。"
                         "它只在你自己的视频内部可比，不要跨机位比较。")
            lines.append("- **站距比** = 双踝距离 / 双肩距离。单打接发与防守时通常 > 1.2。")
            lines.append("- **持拍手抬起率** = 手腕高于肩的帧占比。这是业余选手最容易被识别、"
                         "也最容易改进的一项。")
            lines.append("- **回位时间** = 最后一拍之后速度降到静止阈值以下所需时间。")
        else:
            lines.append("- **蹲伏（代理）** = 球员框高度 / 站立高度基线。**1.00 表示与站立时等高**，"
                         "越小说明身体越压缩（下蹲或屈膝）。它不是关节角度，不能评价技术动作。")
            lines.append("- **回位时间** = 最后一拍之后速度降到静止阈值以下所需时间。")
            lines.append("- 站距、持拍手、手腕速度需要 mediapipe，本模式下为空。")
        lines.append("")

    # 按拍数分组
    if ctx["buckets"] and mode == "mediapipe":
        lines.append("## 4. 不同拍数区间的姿态对比")
        lines.append("")
        lines.append(_table(
            ["拍数区间", "回合数", "平均蹲伏", "持拍手抬起率"],
            [[b["label"], b["count"], _f(b["avg_crouch"], 2), _pct(b["racket_up_rate"], 0)]
             for b in ctx["buckets"]]))
        lines.append("")
        lines.append("读法：如果长回合的蹲伏度明显下降、持拍手抬起率下降，"
                     "说明**多拍过程中姿态在垮**——这通常比“体能不够”更具体、更好练。")
        lines.append("")

    # 时间趋势
    if ctx["windows"]:
        calibrated = _is_calibrated(summary)
        unit = "m" if calibrated else "u"
        lines.append("## 5. 时间趋势（体能/注意力）")
        lines.append("")
        lines.append(_table(
            ["时间段", "回合数", "平均跑动(%s)" % unit, "回位时间(s)", "平均速度(%s/s)" % unit],
            [[w["label"], w["rallies"], _f(_window_dist(w, calibrated), 1),
              _f(w["recovery_s"], 2), _f(w["avg_speed"], 3)] for w in ctx["windows"]]))
        lines.append("")
        trend = _trend_text(ctx["windows"], calibrated)
        if trend:
            lines.append(trend)
            lines.append("")

    # 重点回合相位明细
    lines.append("## 6. 重点回合的相位明细")
    lines.append("")
    for row in _pick_highlights(rows, limit=3):
        if row.get("crouch") is None and row.get("racket_up_rate") is None:
            continue
        lines.append("### 回合 #%d　%s – %s（%s 拍）" % (
            row["id"], _t(row["t0"]), _t(row["t1"]),
            row["shots"] if row["shots"] is not None else "?"))
        lines.append("")
        lines.append(_table(
            ["相位", "指标", "值"],
            [["击球", "最深蹲伏", _f(row.get("crouch_min"), 2)],
             ["击球", "手腕速度峰值", _f(row.get("wrist_peak"), 2)],
             ["准备", "平均蹲伏", _f(row.get("crouch"), 2)],
             ["准备", "平均站距比", _f(row.get("stance"), 2)],
             ["准备", "持拍手抬起率", _pct(row.get("racket_up_rate"), 0)],
             ["回位", "回位时间", _f(row.get("recovery_s"), 2, " s")]]))
        lines.append("")

    lines.append("## 7. 局限（必须读）")
    lines.append("")
    lines.append("1. 单目 2D 姿态**无法**给出真实转体角度、拍面朝向、击球点高度；")
    lines.append("2. 底线后机位存在透视，2D 膝角/肘角会“看起来更直”，"
                 "所以本报告全部使用**尺度无关的比值**，不做绝对角度判断；")
    lines.append("3. 远端球员像素太小，不做姿态分析；")
    lines.append("4. 代理模式下只有框体几何，不能用它评价技术动作；")
    lines.append("5. 蹲伏度、站距比的绝对值取决于机位与身高，"
                 "**只能用于同一机位下的自身对比**（不同场次、不同时间）。")
    lines.append("")
    lines.append("## 8. 建议的用法")
    lines.append("")
    lines.append("1. 先看“持拍手抬起率”：低于 40% 基本可以确定你等球时拍子是放下的；")
    lines.append("2. 再看长回合（≥7 拍）的蹲伏度是否比短回合低 15% 以上；")
    lines.append("3. 最后看回位时间的时间趋势：后半场比前半场慢 30% 以上，就是体能问题；")
    lines.append("4. 每次只改一个指标，下一场再看同一张表。")
    lines.append("")

    lines += _warn_lines(ctx.get("warnings") or [])
    _write(path, lines)
    return path


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------
def _median(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0


def _pick_highlights(rows, limit=4):
    seen = set()
    out = []
    for key in ("shots", "duration", "distance_u"):
        ordered = sorted([r for r in rows if r.get(key) is not None],
                         key=lambda r: r[key], reverse=True)
        for row in ordered[:limit]:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            out.append(row)
            if len(out) >= limit:
                return out
    return out


def _trend_text(windows, calibrated=True):
    valid = [w for w in windows if _window_dist(w, calibrated) is not None]
    if len(valid) < 2:
        return ""
    first, last = valid[0], valid[-1]
    msg = []
    first_d = _window_dist(first, calibrated)
    last_d = _window_dist(last, calibrated)
    if first_d and last_d:
        drop = 1.0 - last_d / first_d
        if drop > 0.15:
            msg.append("最后一段时间每回合跑动比开始下降 **%s**，"
                       "在回合数没有明显减少的前提下，这更像体能下降而非打法变保守。"
                       % _pct(drop, 0))
        elif drop < -0.15:
            msg.append("最后一段时间每回合跑动比开始上升 **%s**，"
                       "可能是对手把你调动得更开，或你在主动加速。" % _pct(-drop, 0))
    if first.get("recovery_s") and last.get("recovery_s"):
        slower = last["recovery_s"] / first["recovery_s"] - 1.0
        if slower > 0.3:
            msg.append("回位时间变慢 **%s**，是典型的疲劳信号。" % _pct(slower, 0))
    if first.get("avg_duration") and last.get("avg_duration"):
        shorter = 1.0 - last["avg_duration"] / first["avg_duration"]
        if shorter > 0.25:
            msg.append("平均回合时长缩短 **%s**，说明多拍相持变少（失分变快或发接发变多）。"
                       % _pct(shorter, 0))
    return "\n".join("- " + m for m in msg) if msg else ""


def _warn_lines(warnings):
    if not warnings:
        return []
    out = ["", "---", "", "### 运行告警", ""]
    out += ["- %s" % w for w in warnings]
    out.append("")
    return out


def _write(path, lines):
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")
