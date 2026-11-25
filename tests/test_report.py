"""报告与端到端自检测试。"""

import os

from _util import tmpdir
from badminton import pipeline, report


def test_selftest_end_to_end():
    out = tmpdir("selftest")
    result = pipeline.selftest(out, verbose=False)
    assert result["truth_rallies"] >= 5
    assert result["detected_rallies"] >= 5
    assert result["recall"] >= 0.8, result

    for name in ("report_rallies.md", "report_pose.md", "rallies.csv", "run_config.json"):
        path = os.path.join(out, name)
        assert os.path.exists(path), path
        assert os.path.getsize(path) > 0, path

    text = open(os.path.join(out, "report_rallies.md"), encoding="utf-8").read()
    for token in ("回合分析报告", "总览", "回合列表", "拍数分布", "方法与局限"):
        assert token in text, token
    assert text.count("|") > 50

    pose = open(os.path.join(out, "report_pose.md"), encoding="utf-8").read()
    for token in ("姿态分析报告", "相位定义", "全局姿态指标", "局限"):
        assert token in pose, token


def test_empty_report_is_graceful():
    out = os.path.join(tmpdir("empty"), "r.md")
    ctx = {
        "source": "none.mp4",
        "cfg": {"still_speed": 0.11, "walk_on_speed": 0.1,
                "serve_still_duration": 0.8, "move_on_speed": 0.16,
                "quiet_duration": 1.5, "walk_duration": 0.55,
                "hit_gap_min": 0.35, "hit_gap_max": 8.0, "pose_every": 2},
        "meta": {"width": 1920, "height": 1080, "fps": 30.0,
                 "duration": 60.0, "analysis_fps": 10.0, "step": 3},
        "detector": "motion",
        "pose_mode": "off",
        "court_desc": "未标定",
        "summary": {"n_rallies": 0, "n_serve_fault": 0, "active_time": 0.0,
                    "dead_time": 60.0, "active_ratio": 0.0, "avg_duration": None,
                    "median_duration": None, "max_duration": None, "avg_shots": None,
                    "max_shots": None, "avg_tempo": None, "total_distance_m": None},
        "buckets": [],
        "windows": [],
        "rows": [],
        "rallies": [],
        "consistency": {"corr": None, "suspicious": []},
        "warnings": [],
    }
    report.write_rally_report(out, ctx)
    text = open(out, encoding="utf-8").read()
    assert "没有识别出任何有效回合" in text


def test_pose_report_off_mode():
    out = os.path.join(tmpdir("poseoff"), "p.md")
    ctx = {"source": "x.mp4", "cfg": {"pose_every": 2},
           "meta": {"width": 640, "height": 360, "fps": 30.0, "duration": 10.0,
                    "analysis_fps": 10.0, "step": 3},
           "rows": [], "summary": {}, "windows": [], "buckets": [],
           "rallies": [], "warnings": [], "pose_mode": "off", "detector": "motion",
           "court_desc": "未标定"}
    report.write_pose_report(out, ctx)
    text = open(out, encoding="utf-8").read()
    assert "未启用姿态分析" in text
