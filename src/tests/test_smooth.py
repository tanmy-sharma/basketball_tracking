"""Checks for src/tracker/smooth.py: depth-only smoothing, per identity, cut at gaps and jumps."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from tracker.smooth import registration_for, smooth_depth  # noqa: E402
from trackid import io  # noqa: E402

H = registration_for("IMG_0115")
HI = np.linalg.inv(H)


def _rows(ident, frames, u, v):
    xy = io.apply_h(H, np.c_[u, v])
    return [(int(f), float(x), float(y), ident[0], ident[1:]) for f, (x, y) in zip(frames, xy)]


def test_only_depth_noise_is_removed():
    rng = np.random.default_rng(0)
    fr = np.arange(1, 301)
    u_true, v_true = 600 + 0.5 * fr, 400 + 0.2 * fr
    rows = _rows("A10", fr, u_true, v_true + rng.normal(0, 3, len(fr)))
    out = smooth_depth(rows, H, 4.0)
    uv = io.apply_h(HI, np.array([(r[1], r[2]) for r in out]))
    assert np.allclose(uv[:, 0], u_true)  # sideways untouched
    mid = slice(20, -20)
    assert np.std(uv[mid, 1] - v_true[mid]) < 0.5 * 3  # depth noise cut by more than half
    assert [(r[0], r[3], r[4]) for r in out] == [(r[0], r[3], r[4]) for r in rows]  # frames, teams, jerseys, order


def test_runs_are_cut_by_identity_gap_and_jump():
    fr = np.arange(1, 41)
    a = _rows("A10", fr, np.full(40, 600.0), np.where(fr <= 20, 350.0, 380.0))  # big depth jump at frame 21
    b = _rows("B11", fr, np.full(40, 700.0), np.full(40, 450.0))
    gap = _rows("A9", np.r_[np.arange(1, 21), np.arange(25, 45)], np.full(40, 500.0), np.r_[np.full(20, 360.0), np.full(20, 420.0)])
    rows = [r for t in zip(a, b) for r in t] + gap  # interleaved like a frame-sorted file
    out = smooth_depth(rows, H, 4.0)
    for r, o in zip(rows, out):
        jump_cm = 1.0  # within a constant run, nothing moves
        assert np.hypot(r[1] - o[1], r[2] - o[2]) < jump_cm, (r, o)


def test_off():
    rows = _rows("A10", np.arange(1, 11), np.full(10, 600.0), np.arange(10) * 5.0 + 300)
    assert smooth_depth(rows, H, 0) == rows
