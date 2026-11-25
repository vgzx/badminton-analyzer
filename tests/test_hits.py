"""挥拍事件与拍数测试。"""

from badminton import hits as H
from badminton.config import Config
from badminton.features import FrameFeat
from badminton.synthetic import _player

FPS = 10.0


def test_robust_z_handles_zero_mad():
    z = H.robust_z([0.0] * 20 + [10.0])
    assert z[-1] > 3.0
    assert abs(float(z[:20].max())) < 3.0


def test_decode_enforces_alternation():
    cfg = Config()
    cands = [
        {"t": 1.0, "side": "near", "score": 3.0},
        {"t": 1.1, "side": "near", "score": 2.5},   # 同侧且间隔过小，必须被丢弃
        {"t": 2.2, "side": "far", "score": 3.1},
        {"t": 3.4, "side": "near", "score": 2.9},
    ]
    out = H.decode(cands, cfg)
    assert len(out) == 3, out
    for a, b in zip(out, out[1:]):
        assert a["side"] != b["side"]


def test_decode_respects_gap_bounds():
    cfg = Config()
    cands = [
        {"t": 1.0, "side": "near", "score": 3.0},
        {"t": 1.2, "side": "far", "score": 3.0},    # 间隔 0.2s < 0.35s
    ]
    assert len(H.decode(cands, cfg)) == 1


def _spike_frames(spikes, n=50):
    """spikes: {帧号: 'near'|'far'}"""
    frames = []
    t = 0.0
    for i in range(n):
        side = spikes.get(i)
        near = _player(0.5, 0.5, 0.5,
                       wrist_v=4.0 if side == "near" else 0.0,
                       accel=1.8 if side == "near" else 0.0,
                       energy=0.5 if side == "near" else 0.1)
        far = _player(0.5, 1.7, 0.5,
                      accel=1.6 if side == "far" else 0.0,
                      energy=0.5 if side == "far" else 0.1)
        frames.append(FrameFeat(t, i, 1.0 / FPS, near, far))
        t += 1.0 / FPS
    return frames


def test_attach_counts_serve_as_first_shot():
    frames = _spike_frames({10: "near", 20: "far", 30: "near"})
    rally = {"id": 0, "t0": 0.5, "t1": 4.0, "duration": 3.5, "kind": "rally"}
    H.attach_to_rallies([rally], frames, Config())
    assert rally["shots"] == 4, rally["shots"]     # 3 次击球 + 发球


def test_attach_collapses_same_side_spikes():
    frames = _spike_frames({10: "near", 20: "near", 30: "near"})
    rally = {"id": 0, "t0": 0.5, "t1": 4.0, "duration": 3.5, "kind": "rally"}
    H.attach_to_rallies([rally], frames, Config())
    assert rally["shots"] == 2, rally["shots"]     # 同侧连击不合法，只保留一次


def test_scores_length_and_finiteness():
    frames = _spike_frames({5: "near"})
    scores = H.swing_scores(frames, "near", Config())
    assert len(scores) == 50
    assert all(abs(float(v)) < 1e6 for v in scores)
