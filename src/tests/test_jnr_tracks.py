"""Synthetic checks for src/tracker/jnr_tracks.py: join, number-switch split, stitching, bridging, both layouts."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.constants import NUMBERS

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("jnr_tracks", ROOT / "src/tracker/jnr_tracks.py")
jt = importlib.util.module_from_spec(spec)
sys.modules["jnr_tracks"] = jt
spec.loader.exec_module(jt)

TEAMS = {"A": ["6", "9", "10"], "B": ["6", "9", "11"]}  # IMG_0114: 6 and 9 on both teams


def _jnr_row(frame, box, number, elig=0.9, conf=0.9):
    p = {f"p_{n}": 0.01 for n in NUMBERS}
    if number:
        p[f"p_{number}"] = conf
    return dict(det_row=0, frame=frame, x=box[0], y=box[1], w=box[2], h=box[3], det_conf=0.9, max_iou=0.0,
                pose_conf=0.9, facing="back", status="ok", crop_w=20, crop_h=30, elig=elig, read=number or "", **p)


def _scene(fmt):
    """Track 1: A10 frames 1-100.  Track 2: A10 again 131-200 (fragment).  Track 3: B11 1-100 then switches to B6
    101-200.  Track 4: unreadable piece 105-125 sitting between tracks 1 and 2 (bridge).  Track 5: never read."""
    rows, jnr = [], []

    def add(frame, tid, xcm, team, number, read=True):
        box = (100.0 * tid + frame * 0.37, 200.0 + tid, 50.0 + tid, 120.0)
        if fmt == "ucmc":
            rows.append([frame, tid, f"{xcm:.2f}", "500.00", "0", "0", "det", "0.90", *(f"{b:.2f}" for b in box),
                         {"A": 1, "B": -1}[team], "0.00", 0])
        else:
            rows.append([frame, tid, *(f"{b:.2f}" for b in box), "0.90", *(f"{b:.2f}" for b in box), team])
        if read:
            jnr.append(_jnr_row(frame, box, number))

    for f in range(1, 101):
        add(f, 1, 300 + f, "A", "10", f % 2 == 1)
        add(f, 3, 900, "B", "11", f % 2 == 1)
        add(f, 5, 1200, "B", None, False)
    for f in range(101, 201):
        add(f, 3, 900, "B", "6", f % 2 == 1)
    for f in range(105, 126):
        add(f, 4, 400 + (f - 100), "A", "10", False)
    for f in range(131, 201):
        add(f, 2, 430 + (f - 130), "A", "10", f % 2 == 1)
    return rows, pd.DataFrame(jnr)


@pytest.fixture(params=["ucmc", "mot"])
def files(request, tmp_path):
    rows, jnr = _scene(request.param)
    t = tmp_path / "tracks.txt"
    pd.DataFrame(rows).to_csv(t, header=False, index=False)
    j = tmp_path / "jnr.csv.gz"
    jnr.to_csv(j, index=False, compression="gzip", float_format="%.5g")
    return request.param, t, j


def test_end_to_end(files):
    fmt, t, j = files
    out, segs, scores, stats = jt.run(t, j, TEAMS)
    assert stats["format"] == fmt
    assert stats["rows_with_jnr"] == stats["rows_eligible"] > 0  # every read joined despite 5-digit rounding
    ids = {f"{a}{n}": jt.IDENTITY_ID_BASE + i for i, (a, n) in enumerate(jt.identities(TEAMS))}
    o = out.assign(frame=out[0].astype(int), id=out[1].astype(int))
    raw = pd.read_csv(t, header=None)
    o["orig"] = raw[1].to_numpy()[out.index]
    # fragments of A10 (tracks 1 and 2) and the unread bridge (track 4) all become one id, in every row
    for tid in (1, 2, 4):
        assert set(o[o.orig == tid].id) == {ids["A10"]}, tid
    assert segs.set_index("orig_id").loc[4, "how"] == "bridge"
    # track 3 switched from B11 to B6 at frame ~100: split, each side relabelled over its whole history
    s3 = o[o.orig == 3]
    assert set(s3[s3.frame <= 99].id) == {ids["B11"]}
    assert set(s3[s3.frame >= 102].id) == {ids["B6"]}
    assert stats["splits"] == 1
    # never read: keeps a non-identity id
    assert set(o[o.orig == 5].id) == {jt.OTHER_ID_BASE}
    # no id twice in a frame, and the file keeps its layout
    assert not o.duplicated(["frame", "id"]).any()
    assert out.shape[1] == raw.shape[1]
    assert scores[ids["A10"]]["10"] > 0.9


def test_conflict_takes_second_choice(tmp_path):
    """Two simultaneous tracks both read as A10: the weaker one may not take A10 at the same time."""
    rows, jnr = [], []
    for f in range(1, 61):
        for tid, n_reads in ((1, 60), (2, 20)):
            box = (100.0 * tid, 200.0, 50.0, 120.0 + f)
            rows.append([f, tid, f"{300 * tid:.2f}", "500.00", "0", "0", "det", "0.90", *box, 1, "0.00", 0])
            if f <= n_reads:
                jnr.append(_jnr_row(f, box, "10", conf=0.6 if tid == 2 else 0.9))
    t, j = tmp_path / "t.txt", tmp_path / "j.csv.gz"
    pd.DataFrame(rows).to_csv(t, header=False, index=False)
    pd.DataFrame(jnr).to_csv(j, index=False, compression="gzip")
    _, segs, _, _ = jt.run(t, j, TEAMS)
    s = segs.set_index("orig_id")
    assert s.loc[1, "identity"] == "A10" and s.loc[1, "how"] == "vote"
    assert s.loc[2, "identity"] != "A10"


def test_attach_is_one_to_one():
    tracks = pd.DataFrame({"frame": [1, 1, 2], "dx": [10.0, 10.2, np.nan], "dy": [0.0, 0.0, 0.0],
                           "dw": [5.0, 5.0, 5.0], "dh": [9.0, 9.0, 9.0]})
    jnr = pd.DataFrame({"frame": [1, 2], "x": [10.0, 10.0], "y": [0.0, 0.0], "w": [5.0, 5.0], "h": [9.0, 9.0]})
    assert list(jt.attach(tracks, jnr)) == [0, -1, -1]


def test_change_points():
    L = np.log(np.vstack([np.tile([0.9, 0.1], (10, 1)), np.tile([0.1, 0.9], (10, 1))]))
    assert jt.change_points(L, 6, 8.0) == [10]
    assert jt.change_points(L[:10], 6, 8.0) == []
