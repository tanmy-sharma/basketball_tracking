from __future__ import annotations

import numpy as np

from src.crops.geometry import facing, max_iou_per_box, torso_box


def kp(l_sho, r_sho, l_hip, r_hip, conf=0.9):
    k = np.zeros((17, 3), np.float32)
    for i, p in zip((5, 6, 11, 12), (l_sho, r_sho, l_hip, r_hip)):
        k[i] = (*p, conf)
    return k


FRONT = kp((70, 20), (30, 20), (65, 80), (35, 80))  # left shoulder at image right


def test_torso_box_padding():
    assert torso_box(FRONT, 0.15, 0.4, 4, 200, 200) == (24, 11, 76, 89)  # x 30..70 +-6, y 20..80 +-9


def test_torso_box_clamps_to_image():
    assert torso_box(FRONT, 0.15, 0.4, 4, 60, 50) == (24, 11, 60, 50)


def test_torso_box_widens_side_view():
    side = kp((50, 20), (52, 20), (50, 80), (52, 80))
    x0, y0, x1, y1 = torso_box(side, 0.15, 0.4, 4, 200, 200)
    assert (x1 - x0) >= 0.4 * (y1 - y0) - 1
    assert (x0 + x1) / 2 == 51


def test_torso_box_degenerate():
    assert torso_box(np.zeros((17, 3), np.float32), 0.15, 0.4, 8, 100, 100) is None
    bad = FRONT.copy()
    bad[5, 0] = np.nan
    assert torso_box(bad, 0.15, 0.4, 4, 200, 200) is None
    assert torso_box(FRONT, 0.15, 0.4, 100, 200, 200) is None  # below min_side


def test_facing():
    assert facing(FRONT) == "front"
    assert facing(kp((30, 20), (70, 20), (35, 80), (65, 80))) == "back"
    assert facing(kp((50, 20), (53, 20), (50, 80), (53, 80))) == "side"
    assert facing(kp((70, 20), (30, 20), (65, 80), (35, 80), conf=0.1)) == "unknown"
    assert facing(kp((70, 20), (30, 20), (65, 20), (35, 20))) == "unknown"  # zero torso height


def test_max_iou():
    assert max_iou_per_box(np.array([[0, 0, 10, 10]])).tolist() == [0.0]
    assert max_iou_per_box(np.zeros((0, 4))).shape == (0,)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [100, 100, 10, 10]], float)
    got = max_iou_per_box(b)
    assert np.allclose(got, [50 / 150, 50 / 150, 0])
    assert max_iou_per_box(np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)).tolist() == [1.0, 1.0]
