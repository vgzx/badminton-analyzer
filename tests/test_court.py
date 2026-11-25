"""球场几何测试。"""

import numpy as np

from badminton.court import COURT_CORNERS_M, Court, apply_homography, dlt_homography

# 一个典型的底线后机位：近端宽、远端窄（梯形）
IMAGE_POINTS = [(120.0, 940.0), (1040.0, 940.0), (330.0, 150.0), (830.0, 150.0)]


def test_dlt_roundtrip():
    H = dlt_homography(COURT_CORNERS_M, IMAGE_POINTS)
    H_inv = np.linalg.inv(H)
    for (xm, ym), px in zip(COURT_CORNERS_M, IMAGE_POINTS):
        back = apply_homography(H_inv, [px])[0]
        assert abs(back[0] - xm) < 1e-6, (back, xm, ym)
        assert abs(back[1] - ym) < 1e-6, (back, xm, ym)


def test_dlt_rejects_degenerate():
    collinear = [(0, 0), (10, 10), (20, 20), (30, 30)]
    try:
        dlt_homography(COURT_CORNERS_M, collinear)
    except ValueError:
        return
    raise AssertionError("共线点应当报错")


def test_calibrated_corners():
    court = Court.from_points(IMAGE_POINTS, 1280, 960)
    assert court.mode == "calibrated"
    x, y = court.to_unit(*IMAGE_POINTS[0])
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    x, y = court.to_unit(*IMAGE_POINTS[3])
    assert abs(x - 1.0) < 1e-6 and abs(y - 2.1967) < 1e-3


def test_calibrated_pixel_roundtrip():
    court = Court.from_points(IMAGE_POINTS, 1280, 960)
    px, py = court.to_px(0.5, 1.1)   # 网附近
    x, y = court.to_unit(px, py)
    assert abs(x - 0.5) < 1e-6 and abs(y - 1.1) < 1e-6


def test_relative_mode():
    court = Court.relative(1920, 1080)
    assert court.mode == "relative"
    x, y = court.to_unit(960, 1080)
    assert abs(x - 0.5) < 1e-9 and abs(y) < 1e-9
    x, y = court.to_unit(0, 0)
    assert abs(x) < 1e-9 and abs(y - 2.1967) < 1e-3
    px, py = court.to_px(0.5, 0.0)
    assert abs(px - 960) < 1e-6 and abs(py - 1080) < 1e-6


def test_save_load():
    import json
    import os

    from _util import tmpdir

    court = Court.from_points(IMAGE_POINTS, 1280, 960)
    path = os.path.join(tmpdir("court"), "court.json")
    court.save(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert len(data["points"]) == 4
    loaded = Court.load(path, 1280, 960)
    assert loaded.mode == "calibrated"
    x, y = loaded.to_unit(*IMAGE_POINTS[2])
    assert abs(x) < 1e-6 and abs(y - 2.1967) < 1e-3
