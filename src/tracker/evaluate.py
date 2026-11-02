#!/usr/bin/env python
"""Score court tracks on a training video: TI-HOTA when each whole track is given its true identity.

Identity comes from ground truth, so this measures how clean the track pieces are and is an upper limit, not a prediction.
A track's identity is the GT player its boxes match most (IoU >= 0.5); tracks matching no player over half the time count
as spurious and are dropped.

  python src/tracker/evaluate.py IMG_0106 TRACKS.txt [more.txt ...]
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
from tracker.run_ucmc import COLUMNS  # noqa: E402
from trackid import io  # noqa: E402
from trackid.metric import ti_hota_fast  # noqa: E402


def load_tracks(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=COLUMNS)


def score(video: str, t: pd.DataFrame) -> dict:
    bb = io.load_bboxes(video)
    ids = io.track_identities(video)
    court = io.gt_rows(io.load_court_gt(video))
    gt = {f: g for f, g in bb.groupby("frame")}

    t = t.reset_index(drop=True)
    owner = np.full(len(t), -1)
    has_box = t[t.kind == "det"]
    for f, g in has_box.groupby("frame"):
        G = gt.get(f)
        if G is None:
            continue
        idx, _ = match(g[["bx", "by", "bw", "bh"]].to_numpy(), G[["x", "y", "w", "h"]].to_numpy())
        owner[g.index.to_numpy()[idx >= 0]] = G.track_id.to_numpy()[idx[idx >= 0]]
    t["owner"] = owner

    label, purity = {}, {}
    for tid, g in t[t.kind == "det"].groupby("id"):
        known = g.owner[g.owner >= 0]
        label[tid] = int(known.mode().iloc[0]) if len(known) and len(known) / len(g) >= 0.5 else -1
        if label[tid] >= 0:
            purity[tid] = (known == label[tid]).sum(), len(known)
    t["label"] = t.id.map(label).fillna(-1).astype(int)
    kept = t[t.label >= 0]
    name = kept.label.map(lambda i: "".join(ids[i]))

    res = dict(rows=len(t), tracks=t.id.nunique(), spurious_tracks=int((pd.Series(label) < 0).sum()),
               spurious_rows=float((t.label < 0).mean()), tracks_per_player=float(kept.groupby("label").id.nunique().median()),
               median_track_len=float(t.groupby("id").size().median()),
               purity=sum(a for a, _ in purity.values()) / max(sum(b for _, b in purity.values()), 1))
    r = ti_hota_fast(court, list(zip(kept.frame, kept.x, kept.y, name)))
    res.update(TI=r["TI-HOTA"], DetA=r["DetA"], AssA=r["AssA"])
    no_coast = kept.kind == "det"
    res["TI_no_coast"] = ti_hota_fast(court, list(zip(kept.frame[no_coast], kept.x[no_coast], kept.y[no_coast], name[no_coast])))["TI-HOTA"]
    res["TI_raw"] = ti_hota_fast(court, list(zip(kept.frame[no_coast], kept.raw_x[no_coast], kept.raw_y[no_coast], name[no_coast])))["TI-HOTA"]
    return res


def from_mot(path: Path | str, H: np.ndarray) -> pd.DataFrame:
    """Load another tracker's MOT output into this format, keeping only rows inside the court."""
    m = pd.read_csv(path, header=None, names=["frame", "id", "bx", "by", "bw", "bh", "a", "b"])
    p = io.apply_h(H, np.c_[m.bx + m.bw / 2, m.by + m.bh])
    inside = (p[:, 0] >= 0) & (p[:, 0] <= io.COURT_W) & (p[:, 1] >= 0) & (p[:, 1] <= io.COURT_H)
    return m.assign(x=p[:, 0], y=p[:, 1], raw_x=p[:, 0], raw_y=p[:, 1], kind="det", conf=1.0)[inside]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("tracks", nargs="+")
    a = ap.parse_args()
    for path in a.tracks:
        print(path, {k: round(v, 3) if isinstance(v, float) else v for k, v in score(a.video, load_tracks(path)).items()})


if __name__ == "__main__":
    main()
