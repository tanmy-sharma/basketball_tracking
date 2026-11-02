#!/usr/bin/env python
"""Where do cross-team identity errors happen? Training videos only: GT is used to find and describe them, never as input.

  python src/tracker/cross_team.py IMG_0107 TRACKS.txt [--labels LABELS.csv.gz --detections DETS.txt] [--csv OUT.csv]

A row is cross-team wrong when its box matches (IoU >= 0.5) a GT player of the other team from the track's main player.
Consecutive wrong rows of one track (gaps up to 15 frames) form an episode. Each episode is described by how it began
and what the colour labels said during it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from eval_detections import load_dets  # noqa: E402
from tracker import diagnose as dg  # noqa: E402
from tracker.run_ucmc import COLUMNS, OVERLAP_IOU, TEAM_COLUMNS, max_iou  # noqa: E402
from trackid import io  # noqa: E402

GAP = 15


def load(path) -> pd.DataFrame:
    t = pd.read_csv(path, header=None)
    t.columns = (COLUMNS + TEAM_COLUMNS)[: t.shape[1]]
    return t


def rows_with_owner(video: str, t: pd.DataFrame) -> pd.DataFrame:
    """Each det row with its GT owner (-1 if none), the track's main player, and teams."""
    t = t.reset_index(drop=True)
    o = dg.owners(video, t)
    o = o[o.track >= 0]
    # owners() is per GT box; map back to the det row of that track in that frame
    key = t[t.kind == "det"].reset_index().set_index(["frame", "id"])["index"]
    t["owner"] = -1
    idx = key.reindex(pd.MultiIndex.from_arrays([o.frame, o.track])).to_numpy()
    ok = ~pd.isna(idx)
    t.loc[idx[ok].astype(int), "owner"] = o.player.to_numpy()[ok]
    m = t[t.owner >= 0]
    main = m.groupby("id").owner.agg(lambda s: s.mode().iloc[0])
    team = {k: v[0] for k, v in io.track_identities(video).items()}
    t["main"] = t.id.map(main).fillna(-1).astype(int)
    t["owner_team"] = t.owner.map(team)
    t["main_team"] = t.main.map(team)
    t["cross"] = (t.owner >= 0) & (t.main >= 0) & (t.owner_team != t.main_team)
    return t


def episodes(t: pd.DataFrame, dets: pd.DataFrame | None, labels: pd.DataFrame | None) -> pd.DataFrame:
    over = None
    if dets is not None:
        feet_ok = dets.index.isin(t.det_row[t.det_row >= 0]) if "det_row" in t else None
        sub = dets[feet_ok] if feet_ok is not None else dets
        over = pd.Series(0.0, index=sub.index)
        for _, g in sub.groupby("frame"):
            over.loc[g.index] = max_iou(g[["x", "y", "w", "h"]].to_numpy())
    out = []
    for tid, g in t[t.kind == "det"].groupby("id"):
        g = g.sort_values("frame")
        w = g[g.cross]
        if w.empty:
            continue
        run = (w.frame.diff().fillna(GAP + 1) > GAP).cumsum()
        for _, e in w.groupby(run):
            f0, f1 = int(e.frame.min()), int(e.frame.max())
            before = g[g.frame < f0]
            prev_gap = int(f0 - before.frame.max() - 1) if len(before) else -1
            prev_owner = before.owner[before.owner >= 0].iloc[-1] if (before.owner >= 0).any() else -1
            sign = {"A": 1, "B": -1}
            mt = sign[e.main_team.iloc[0]]
            calls = e.team.to_numpy() if "team" in e else np.zeros(len(e))
            rec = dict(track=tid, start=f0, end=f1, rows=len(e), main=int(e.main.iloc[0]), owner=int(e.owner.mode().iloc[0]),
                       gap_before=prev_gap, came_from_owner=bool(prev_owner == e.owner.iloc[0]),
                       start_x=float(e.x.iloc[0]), start_y=float(e.y.iloc[0]), conf=float(e.conf.median()),
                       calls_track_team=float((calls == mt).mean()), calls_owner_team=float((calls == -mt).mean()),
                       no_call=float((calls == 0).mean()),
                       belief_at_start=float(e.belief.iloc[0] * mt) if "belief" in e else np.nan)
            if over is not None:
                rec["overlapped"] = float((over.reindex(e.det_row).to_numpy() >= OVERLAP_IOU).mean())
            out.append(rec)
    ep = pd.DataFrame(out)
    # How it began: the track was coasting (lost its player) and was picked up by someone else, or it slid across directly
    ep["how"] = np.where(ep.gap_before > 0, "re-acquired after a gap", np.where(ep.gap_before < 0, "track born on wrong player", "slid across with no gap"))
    return ep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("tracks")
    ap.add_argument("--detections")
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    t = rows_with_owner(a.video, load(a.tracks))
    dets = load_dets(a.detections) if a.detections else None
    ep = episodes(t, dets, None)
    n = int(t.cross.sum())
    print(f"{a.video}: {n} cross-team rows of {int((t.owner >= 0).sum())} matched rows, {len(ep)} episodes")
    print("\nBy how the episode began (rows):\n", ep.groupby("how").rows.agg(["count", "sum", "median"]).to_string())
    long = ep[ep.rows >= 30]
    print(f"\nEpisodes of 30+ rows: {len(long)}, holding {long.rows.sum()} rows ({long.rows.sum() / max(n, 1):.0%})")
    w = ep.rows / ep.rows.sum()
    print("\nRow-weighted, during episodes:")
    for c in ("calls_track_team", "calls_owner_team", "no_call", "overlapped", "conf"):
        if c in ep:
            print(f"  {c:18s} {np.average(ep[c].fillna(0), weights=w):.2f}")
    print(f"  track sure of its team at start (belief>3): {np.average(ep.belief_at_start > 3, weights=w):.2f}")
    print(f"  gap before (frames, median over row-weighted long episodes): {long.gap_before.median()}")
    zx = pd.cut(ep.start_x, [-1e9, 0, 500, 1000, 1505, 1e9], labels=["out-left", "0-5m", "5-10m", "10-15m", "out-right"])
    zy = pd.cut(ep.start_y, [-1e9, 0, 550, 1105, 1e9], labels=["out-top", "0-5.5m", "5.5-11m", "out-bottom"])
    print("\nWhere episodes start (rows):\n", pd.crosstab(zy, zx, values=ep.rows, aggfunc="sum").fillna(0).astype(int).to_string())
    print("\nWorst 12 episodes:\n", ep.sort_values("rows", ascending=False).head(12)[
        ["track", "start", "end", "rows", "main", "owner", "how", "gap_before", "calls_track_team", "calls_owner_team", "no_call",
         "overlapped", "belief_at_start"]].round(2).to_string(index=False))
    if a.csv:
        ep.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
