#!/usr/bin/env python
"""How right is a box's team call, by how much it overlaps its nearest neighbour and what that neighbour was called?

  python src/tracker/overlap_rates.py [--videos ...] [--out outputs/reports/overlap_rates.json]

Neighbour = the labelled box with the highest IoU in the same frame. Relation: 'same' (neighbour has the same call), 'other'
(opposite call) or 'nocall'. Accuracy is measured against GT on training videos (boxes matching a GT player at IoU >= 0.5);
the table is what the tracker would use, with no GT at run time. Laplace (+1/+2) so no bin is exactly 0 or 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from eval_detections import iou_matrix, load_dets, match  # noqa: E402
from trackid import io  # noqa: E402

VIDEOS = [f"IMG_01{n:02d}" for n in range(5, 12)]
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
SIGN = {"A": 1, "B": -1}


def one(video: str) -> pd.DataFrame:
    det = ROOT / f"outputs/colab_all/tracking_results/{video}/8f5defcb/detections/{video}__rfdetr-large__720p.txt"
    d = load_dets(det)
    lab = pd.read_csv(ROOT / f"outputs/labels/{video}__rfdetr-large__720p.csv.gz", usecols=["det_row", "team"]).set_index("det_row")
    d = d.loc[lab.index].assign(call=lab.team.map(SIGN).fillna(0).astype(int))
    bb, ids = io.load_bboxes(video), io.track_identities(video)
    gt = {f: g for f, g in bb.groupby("frame")}
    out = []
    for f, g in d.groupby("frame"):
        box, call = g[["x", "y", "w", "h"]].to_numpy(), g.call.to_numpy()
        m = iou_matrix(box, box) if len(g) > 1 else np.zeros((1, 1))
        np.fill_diagonal(m, 0)
        nb = m.argmax(1)
        iou = m.max(1)
        true = np.zeros(len(g), int)
        G = gt.get(f)
        if G is not None:
            idx, _ = match(box, G[["x", "y", "w", "h"]].to_numpy())
            tids = G.track_id.to_numpy()
            true = np.array([SIGN[ids[tids[k]][0]] if k >= 0 and tids[k] in ids else 0 for k in idx])
        for i in np.flatnonzero((call != 0) & (true != 0)):
            rel = "none" if iou[i] == 0 else "nocall" if call[nb[i]] == 0 else "same" if call[nb[i]] == call[i] else "other"
            out.append((video, iou[i], rel, call[i] == true[i]))
    return pd.DataFrame(out, columns=["video", "iou", "rel", "right"])


def table(r: pd.DataFrame) -> pd.DataFrame:
    r = r.assign(bin=pd.cut(r.iou, BINS, right=False))
    t = r.groupby(["rel", "bin"], observed=True).right.agg(["size", "sum"])
    t["q"] = (t["sum"] + 1) / (t["size"] + 2)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", default=VIDEOS)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/reports/overlap_rates.json")
    a = ap.parse_args()
    with Pool(len(a.videos)) as p:
        r = pd.concat(p.map(one, a.videos))
    t = table(r)
    print(t.round(4).to_string())
    # Leave-one-video-out spread of q per cell, to see whether the table holds on an unseen game
    lovo = pd.concat({v: table(r[r.video != v]).q for v in a.videos}, axis=1)
    print("\nLeave-one-video-out q, min and max per cell:\n", pd.concat([lovo.min(1), lovo.max(1)], axis=1, keys=["min", "max"]).round(4).to_string())
    a.out.write_text(json.dumps({"bins": BINS, "table": [dict(rel=k[0], lo=k[1].left, hi=k[1].right, n=int(v["size"]), q=float(v.q))
                                                          for k, v in t.iterrows()]}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
