#!/usr/bin/env python
"""Compare cached detections with the ground-truth player boxes, and show which boxes get missed.

  python eval_detections.py IMG_0106 ../outputs/detections/IMG_0106__rfdetr-small__720p.txt [more.txt ...]

"on-court" keeps detections whose bottom-centre projects within --margin cm of the court, since GT only contains
the 6 players and raw precision is dominated by spectators and officials. Only frames in the det file are scored.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from trackid import io

THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def load_dets(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, header=None, usecols=[0, 2, 3, 4, 5, 6], names=["frame", "x", "y", "w", "h", "conf"])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: Nx4 xywh. Pairwise IoU."""
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = (np.minimum(ax2[:, None], bx2) - np.maximum(a[:, 0, None], b[:, 0])).clip(0)
    ih = (np.minimum(ay2[:, None], by2) - np.maximum(a[:, 1, None], b[:, 1])).clip(0)
    inter = iw * ih
    return inter / (a[:, 2, None] * a[:, 3, None] + b[:, 2] * b[:, 3] - inter + 1e-9)


def match(gt: np.ndarray, det: np.ndarray, thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian on IoU. Returns (matched det index per GT or -1, IoU per GT against the matched det)."""
    idx, ious = np.full(len(gt), -1), np.zeros(len(gt))
    if len(gt) and len(det):
        m = iou_matrix(gt, det)
        for g, d in zip(*linear_sum_assignment(-m)):
            if m[g, d] >= thr:
                idx[g], ious[g] = d, m[g, d]
    return idx, ious


def on_court(dets: pd.DataFrame, H: np.ndarray, margin: float) -> np.ndarray:
    p = io.apply_h(H, np.c_[dets.x + dets.w / 2, dets.y + dets.h])
    return (p[:, 0] > -margin) & (p[:, 0] < io.COURT_W + margin) & (p[:, 1] > -margin) & (p[:, 1] < io.COURT_H + margin)


def score(gt: dict, dets: pd.DataFrame, frames: np.ndarray) -> tuple[int, int, int]:
    by = {f: d[["x", "y", "w", "h"]].to_numpy() for f, d in dets.groupby("frame")}
    tp = fn = fp = 0
    for f in frames:
        g, d = gt[f], by.get(f, np.empty((0, 4)))
        n = int((match(g, d)[0] >= 0).sum())
        tp, fn, fp = tp + n, fn + len(g) - n, fp + len(d) - n
    return tp, fn, fp


def rate_by(df: pd.DataFrame, key: pd.Series) -> str:
    g = df.groupby(key, observed=True).hit.agg(["mean", "size"])
    return "\n".join(f"  {k!s:>14}: recall {r['mean']:.3f}  (n={int(r['size'])})" for k, r in g.iterrows())


def evaluate(video: str, path: str, margin: float, conf: float | None) -> None:
    H, _ = io.load_homographies(io.TRAIN / "court_registration.csv")
    roster = io.load_roster(io.TRAIN / "games_roster.csv")[video]
    all_dets = load_dets(path)
    frames = np.arange(all_dets.frame.min(), all_dets.frame.max() + 1)
    every = int(np.median(np.diff(np.sort(all_dets.frame.unique()))))
    frames = frames[(frames - frames[0]) % every == 0]
    bb = io.load_bboxes(video)
    bb = bb[bb.frame.isin(frames)]
    gt = {f: g[["x", "y", "w", "h"]].to_numpy() for f, g in bb.groupby("frame")}
    frames = np.array([f for f in frames if f in gt])
    court = all_dets[on_court(all_dets, H, margin)]

    print(f"\n=== {Path(path).stem}: {len(frames)} frames, {len(bb)} GT boxes, median height "
          f"{bb.h.median():.0f}px at full resolution")
    print("conf | raw recall precision | on-court recall precision F1 FP/frame")
    best = (0.0, 0.5)
    for t in THRESHOLDS:
        tp, fn, fp = score(gt, all_dets[all_dets.conf >= t], frames)
        tc, fc, pc = score(gt, court[court.conf >= t], frames)
        rc, pr = tc / max(tc + fc, 1), tc / max(tc + pc, 1)
        f1 = 2 * rc * pr / max(rc + pr, 1e-9)
        best = max(best, (f1, t))
        print(f"{t:4.2f} | {tp / max(tp + fn, 1):.3f} {tp / max(tp + fp, 1):.3f} | "
              f"{rc:.3f} {pr:.3f} {f1:.3f} {pc / len(frames):.2f}")

    t = best[1] if conf is None else conf
    used = court[court.conf >= t]
    by = {f: d[["x", "y", "w", "h"]].to_numpy() for f, d in used.groupby("frame")}
    rows = []
    for f, g in bb.groupby("frame"):
        box = g[["x", "y", "w", "h"]].to_numpy()
        d = by.get(f, np.empty((0, 4)))
        idx, ious = match(box, d)
        overlap = iou_matrix(box, box) - np.eye(len(box))
        for i, tid in enumerate(g.track_id):
            hit = idx[i] >= 0
            err, herr = np.nan, np.nan
            if hit:
                m = d[idx[i]]
                a = io.apply_h(H, [[box[i, 0] + box[i, 2] / 2, box[i, 1] + box[i, 3]]])[0]
                b = io.apply_h(H, [[m[0] + m[2] / 2, m[1] + m[3]]])[0]
                err, herr = float(np.hypot(*(a - b))), (m[1] + m[3]) - (box[i, 1] + box[i, 3])
            rows.append((f, tid, box[i, 3], overlap[i].max(), hit, ious[i], err, herr))
    r = pd.DataFrame(rows, columns=["frame", "tid", "h", "overlap", "hit", "iou", "err", "bottom"])
    e = r.err.dropna()
    print(f"\nAt conf >= {t} (best on-court F1 unless set with --conf): recall {r.hit.mean():.3f}")
    print(f"Matched boxes: mean IoU {r.iou[r.hit].mean():.3f}. Foot-point error against GT in cm: median "
          f"{e.median():.1f}, p90 {e.quantile(.9):.1f}, p99 {e.quantile(.99):.1f}; within 13 cm {(e <= 13).mean():.3f}, "
          f"30 cm {(e <= 30).mean():.3f}, 100 cm {(e <= 100).mean():.3f}. Box-bottom offset median "
          f"{r.bottom.median():+.1f}px (positive = below GT).")
    print("Recall by overlap with another player's box (max IoU):")
    print(rate_by(r, pd.cut(r.overlap, [-1, 0, .1, .3, .5, 1], labels=["none", "0-0.1", "0.1-0.3", "0.3-0.5", ">0.5"])))
    print("Recall by GT box height (full-resolution px):")
    print(rate_by(r, pd.qcut(r.h, 4, duplicates="drop").astype(str)))
    print("Recall by player:")
    print(rate_by(r, r.tid.map(lambda i: f"#{roster.track_jersey.get(i, '?')} (id {i})")))
    print("Recall before/after game start:")
    print(rate_by(r, np.where(r.frame < (roster.game_start or 0), "pre-game", "in game")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("dets", nargs="+")
    ap.add_argument("--margin", type=float, default=150.0)
    ap.add_argument("--conf", type=float)
    a = ap.parse_args()
    assert re.fullmatch(r"IMG_\d{4}", a.video), "video is a name like IMG_0106"
    for p in a.dets:
        evaluate(a.video, p, a.margin, a.conf)


if __name__ == "__main__":
    main()
