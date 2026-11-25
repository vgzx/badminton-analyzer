"""回合状态机测试：用合成特征流验证切分行为。"""

from badminton.config import Config
from badminton.features import FrameFeat
from badminton.rally import RallySegmenter
from badminton.synthetic import _player

FPS = 10.0


def _stream(segments):
    """segments: [(时长, 近端x, 近端y, 近端速度, 远端x, 远端y, 远端速度)]"""
    frames = []
    t = 0.0
    idx = 0
    for dur, nx, ny, nsp, fx, fy, fsp in segments:
        for _ in range(int(round(dur * FPS))):
            near = _player(nx, ny, nsp, pose=True)
            far = _player(fx, fy, fsp)
            frames.append(FrameFeat(t, idx, 1.0 / FPS, near, far))
            t += 1.0 / FPS
            idx += 1
    return frames


def _run(frames, cfg=None):
    segmenter = RallySegmenter(cfg or Config())
    for frame in frames:
        segmenter.update(frame.t, frame)
    segmenter.finish()
    return segmenter.rallies


SERVE = (1.2, 0.50, 0.45, 0.01, 0.50, 1.75, 0.01)
LIVE = (2.0, 0.50, 0.45, 0.60, 0.50, 1.75, 0.55)
QUIET = (1.2, 0.50, 0.45, 0.00, 0.50, 1.75, 0.00)
STOP = (1.8, 0.50, 0.60, 0.00, 0.50, 1.70, 0.00)
WALK = (1.5, 0.50, 0.25, 0.16, 0.50, 1.35, 0.14)


def test_single_rally():
    rallies = _run(_stream([SERVE, LIVE, STOP, WALK]))
    assert len(rallies) == 1, rallies
    rally = rallies[0]
    assert rally["kind"] == "rally"
    assert 1.5 < rally["duration"] < 3.5, rally


def test_quiet_gap_does_not_split():
    # 活球中间双方静止 1.2s（短于 quiet_duration=1.5）不应被切成两个回合
    rallies = _run(_stream([SERVE, LIVE, QUIET, LIVE, STOP, WALK]))
    assert len(rallies) == 1, rallies


def test_two_rallies():
    rallies = _run(_stream([SERVE, LIVE, STOP, WALK, SERVE, LIVE, STOP, WALK]))
    assert len(rallies) == 2, rallies
    assert rallies[0]["t1"] <= rallies[1]["t0"] + 0.01


def test_serve_fault_kept_but_labelled():
    short_live = (0.3, 0.50, 0.45, 0.60, 0.50, 1.75, 0.55)
    rallies = _run(_stream([SERVE, short_live, STOP, WALK]))
    assert len(rallies) == 1
    assert rallies[0]["kind"] == "serve_fault", rallies[0]


def test_cold_start():
    moving = [(3.0, 0.50, 0.45, 0.60, 0.50, 1.75, 0.55)]
    rallies = _run(_stream(moving + [STOP, WALK]))
    assert len(rallies) == 1
    assert rallies[0]["duration"] > 2.0


def test_no_rally_when_everything_still():
    still = [(20.0, 0.50, 0.45, 0.0, 0.50, 1.75, 0.0)]
    rallies = _run(_stream(still))
    assert rallies == []
