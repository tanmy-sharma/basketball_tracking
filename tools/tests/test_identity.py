import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tracker.identity import BASE_COLUMNS, TEAM_COLUMNS, make_rows, split_tracks  # noqa: E402

TEAMS = {"A": ["6", "9", "10"], "B": ["6", "9", "11"]}
N = 400


def fake(n_tracks=10, with_team=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(1, n_tracks + 1):
        a, b = sorted(rng.integers(1, N, 2))
        if b - a < 40:
            b = min(N, a + 60)
        team = 1 if t % 2 else -1
        x0, y0 = rng.uniform(100, 1400), rng.uniform(100, 1000)
        for f in range(a, b + 1):
            rows.append((f, t, x0 + f, y0, x0, y0, "det", 0.9, 0, 0, 10, 10, team if rng.random() > .1 else 0, team * 2.0, 0))
    df = pd.DataFrame(rows, columns=BASE_COLUMNS + TEAM_COLUMNS)
    return df if with_team else df[BASE_COLUMNS].assign(team=0, belief=0.0, det_row=-1)


def check(rows):
    assert {r[0] for r in rows} == set(range(1, N + 1))
    seen = Counter((r[0], r[3], r[4]) for r in rows)
    assert max(seen.values()) == 1
    per = Counter((r[0], r[3]) for r in rows)
    assert max(per.values()) <= 3
    for r in rows:
        assert r[3] in TEAMS and r[4] in TEAMS[r[3]]
        assert np.isfinite(r[1]) and np.isfinite(r[2])


def test_valid_with_teams():
    check(make_rows(fake(), TEAMS, N))


def test_valid_without_team_calls():
    check(make_rows(fake(12, with_team=False), TEAMS, N))


def test_empty_frames_filled_and_empty_input():
    df = fake(1)
    check(make_rows(df[df.frame < 100], TEAMS, N))
    check(make_rows(df.iloc[:0], TEAMS, N))


def test_split_on_team_flip():
    f = np.arange(1, 201)
    team = np.where(f <= 100, 1, -1)
    df = pd.DataFrame(dict(frame=f, id=1, x=1.0, y=1.0, conf=.9, team=team, belief=0.0))
    assert split_tracks(df, k=30).seg.nunique() == 2
    df["team"] = 1
    assert split_tracks(df, k=30).seg.nunique() == 1


def test_overlapping_same_team_get_different_numbers_and_scores_used():
    rows = make_rows(fake(), TEAMS, N, jersey_scores={1: {"10": 5.0}})
    check(rows)
    assert any(r[4] == "10" for r in rows)
