#!/usr/bin/env python
"""Say what each identity break in a court-track file actually is, using the training ground truth.

  python src/tracker/diagnose.py IMG_0106 TRACKS.txt [more.txt ...]

Each time a GT player's matched track changes (IoU >= 0.5), the event is one of:
  takeover     the new track had already been on another player: a real identity swap
  handed_over  the old track moves on to another player: also a swap, seen from the other side
  restart      the new track is fresh and the old one has ended: identity lost, but no other player involved
  duplicate    the new track is fresh while the old one lives on unmatched
Also counted: recovered (same track returns after a gap) and wrong_frames (rows on a player other than the track's main one).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from eval_detections import match  # noqa: E402
from tracker.evaluate import load_tracks, score  # noqa: E402
from trackid import eval_mot, io  # noqa: E402

KINDS = ["takeover", "handed_over", "restart", "duplicate"]


def owners(video: str, t: pd.DataFrame) -> pd.DataFrame:
    """One row per GT box with the track matched to it (-1 if none)."""
    bb = io.load_bboxes(video)
    by = {f: g for f, g in t[t.kind == "det"].groupby("frame")}
    rows = []
    for f, G in bb.groupby("frame"):
        g = by.get(f)
        if g is None:
            rows += [(f, tid, -1) for tid in G.track_id]
            continue
        idx, _ = match(G[["x", "y", "w", "h"]].to_numpy(), g[["bx", "by", "bw", "bh"]].to_numpy())
        ids = g.id.to_numpy()
        rows += [(f, tid, int(ids[k]) if k >= 0 else -1) for tid, k in zip(G.track_id, idx)]
    return pd.DataFrame(rows, columns=["frame", "player", "track"])


def classify(o: pd.DataFrame, t: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Returns (events, recovered count, wrong-player rows)."""
    matched = o[o.track >= 0].sort_values("frame")
    first_seen, last_row = {}, t.groupby("id").frame.max().to_dict()
    per_track = {tr: g for tr, g in matched.groupby("track")}
    for tr, g in per_track.items():
        first_seen[tr] = (g.frame.to_numpy(), g.player.to_numpy())

    def on_other_before(tr, player, frame):
        fr, pl = first_seen[tr]
        return bool(((fr < frame) & (pl != player)).any())

    def on_other_after(tr, player, frame):
        fr, pl = first_seen[tr]
        return bool(((fr >= frame) & (pl != player)).any())

    events, recovered = [], 0
    for player, g in matched.groupby("player"):
        frames, tracks = g.frame.to_numpy(), g.track.to_numpy()
        for i in range(1, len(g)):
            gap = frames[i] - frames[i - 1] - 1
            a, b = tracks[i - 1], tracks[i]
            if a == b:
                recovered += gap > 0
                continue
            if on_other_before(b, player, frames[i]):
                kind = "takeover"
            elif on_other_after(a, player, frames[i]):
                kind = "handed_over"
            elif last_row.get(a, 0) < frames[i]:
                kind = "restart"
            else:
                kind = "duplicate"
            events.append((player, frames[i], gap, kind))
    main = matched.groupby("track").player.agg(lambda s: s.mode().iloc[0])
    wrong = int((matched.player != matched.track.map(main)).sum())
    return pd.DataFrame(events, columns=["player", "frame", "gap", "kind"]), recovered, wrong


def box_metrics(video: str, t: pd.DataFrame) -> dict:
    gt = eval_mot._read_mot_txt(io.GT_BBOX / f"{video}.txt")
    r = t[t.kind == "det"]
    a = np.c_[r.frame, r.id, r.bx, r.by, r.bw, r.bh, np.ones((len(r), 2))].astype(float)
    d = eval_mot.build_sequence_data(seq_name="s", gt=gt, tracker=a, eval_class_id=None, distractor_ids=set(), pred_class_id=None)
    s = eval_mot.summarize_bundle(eval_mot.eval_bundle(d, 0.5))
    return {k: s[k] for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag")}


def diagnose(video: str, t: pd.DataFrame) -> dict:
    t = t.reset_index(drop=True)
    o = owners(video, t)
    ev, recovered, wrong = classify(o, t)
    counts = ev.kind.value_counts().reindex(KINDS, fill_value=0) if len(ev) else pd.Series(0, index=KINDS)
    res = dict(ids=t.id.nunique(), breaks=len(ev), **counts.to_dict(), recovered=recovered,
               wrong_frames=wrong, covered=float((o.track >= 0).mean()), coast_rows=int((t.kind == "coast").sum()))
    res["restart_gap_median"] = float(ev.gap[ev.kind == "restart"].median()) if (ev.kind == "restart").any() else 0.0
    res.update(box_metrics(video, t))
    orc = score(video, t)
    res.update(TI=orc["TI"], tracks_per_player=orc["tracks_per_player"], purity=float(orc["purity"]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("tracks", nargs="+")
    a = ap.parse_args()
    for p in a.tracks:
        print(p, {k: round(v, 3) if isinstance(v, float) else v for k, v in diagnose(a.video, load_tracks(p)).items()})


if __name__ == "__main__":
    main()
