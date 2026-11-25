"""指标聚合测试。"""

from badminton import hits as H
from badminton import metrics as M
from badminton.config import Config
from badminton.synthetic import synth_frames


def _rows():
    frames, truth, _ = synth_frames()
    cfg = Config()
    from badminton.rally import RallySegmenter

    seg = RallySegmenter(cfg)
    for frame in frames:
        seg.update(frame.t, frame)
    seg.finish()
    rallies = H.attach_to_rallies(seg.rallies, frames, cfg, duration=frames[-1].t)
    rows = [M.rally_row(r, frames, cfg) for r in rallies]
    return frames, rallies, rows, cfg


def test_rally_row_fields():
    frames, rallies, rows, cfg = _rows()
    assert rows, "应当至少有一个回合"
    row = rows[0]
    for key in ("id", "t0", "t1", "duration", "shots", "distance_u", "peak_speed"):
        assert key in row, key
    assert row["duration"] > 0
    assert row["distance_u"] >= 0


def test_pose_metrics_present():
    _, _, rows, _ = _rows()
    with_crouch = [r for r in rows if r.get("crouch") is not None]
    assert with_crouch, "合成数据带姿态，应当产出蹲伏度"
    assert with_crouch[0]["pose_frames"] > 0


def test_aggregate_shape():
    frames, rallies, rows, cfg = _rows()
    agg = M.aggregate(rows, rallies, frames[-1].t, cfg,
                      pose_mode="mediapipe", court_mode="relative")
    summary = agg["summary"]
    assert summary["n_rallies"] == len(rows)
    assert summary["active_time"] > 0
    assert summary["active_ratio"] is not None
    assert 0.0 <= summary["active_ratio"] <= 1.0
    assert agg["shot_buckets"]
    assert agg["windows"]


def test_consistency_runs():
    _, _, rows, _ = _rows()
    cons = M.consistency(rows)
    assert "corr" in cons and "suspicious" in cons
