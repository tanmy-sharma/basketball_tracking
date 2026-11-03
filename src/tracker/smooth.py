"""Offline smoothing of exported submission rows along the camera's depth direction.

Each row's court point is projected back into the image, the image-v coordinate (depth, along the line of sight, where
the box-bottom foot point is noisiest) is Gaussian-smoothed over time, and the point is projected back to the court.
Image-u (sideways) is left as is: smoothing it lowered the score. Runs are per identity (team + jersey) over consecutive
frames, cut at a missing frame or a one-frame jump over break_cm, so different players and stitch jumps never blend.

Chosen on the 8 training games (outputs/reports/smoothing/): sigma 4 frames raised mean TI-HOTA 0.9025 -> 0.9052 and
made no game worse (IMG_0108 held out: 0.9069 -> 0.9091). Larger sigmas cost training score. Positions only: team,
jersey, frames and row order are unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from trackid import io


def registration_for(video: str):
    """Image->court homography of the camera the video was filmed with."""
    final = io.load_roster(io.FINAL / "games_roster.csv")
    return io.load_homographies((io.FINAL if video in final else io.TRAIN) / "court_registration.csv")[0]


def smooth_depth(rows: list, H: np.ndarray, sigma: float = 4.0, break_cm: float = 100.0) -> list:
    """rows: (frame, x, y, team, jersey) as identity.make_rows returns. H: image->court. sigma in frames, 0 = off."""
    if sigma <= 0 or not rows:
        return list(rows)
    f = np.array([r[0] for r in rows], np.int64)
    xy = np.array([(r[1], r[2]) for r in rows], np.float64)
    key = np.array([f"{r[3]}{r[4]}" for r in rows])
    order = np.lexsort((f, key))
    fo, xyo, ko = f[order], xy[order], key[order]
    uv = io.apply_h(np.linalg.inv(H), xyo)
    cut = np.r_[True, (ko[1:] != ko[:-1]) | (np.diff(fo) != 1) | (np.hypot(*np.diff(xyo, axis=0).T) > break_cm)]
    starts = np.flatnonzero(cut)
    for s, e in zip(starts, np.r_[starts[1:], len(fo)]):
        uv[s:e, 1] = gaussian_filter1d(uv[s:e, 1], sigma, mode="nearest")
    new = np.empty_like(xy)
    new[order] = io.apply_h(H, uv)
    return [(r[0], float(x), float(y), r[3], r[4]) for r, (x, y) in zip(rows, new)]
