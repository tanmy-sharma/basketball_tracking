#!/usr/bin/env python
"""Team labels on vs off in the tracker, on identical detections, across the training games.

  python src/tracker/sweep_team.py [--videos IMG_0105 ...] [--max-frame N] [--out outputs/reports/team_sweep.jsonl]

Every variant runs at both settings: current (cdt 8, ambiguity 1) and loose (cdt 20, ambiguity 0). Metrics come from
diagnose(): box AssA and IDF1, coverage, purity, wrong-player rows (split here into cross-team and same-team) and oracle
TI-HOTA. GT is used for scoring only, except in the 'perfect' variant, a labelled diagnostic where team calls come from GT.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from eval_detections import load_dets, match  # noqa: E402
from tracker import diagnose as dg  # noqa: E402
from tracker.cross_team import episodes, rows_with_owner  # noqa: E402
from tracker.run_ucmc import Params, load_labels, track  # noqa: E402
from trackid import io  # noqa: E402

VIDEOS = [f"IMG_01{n:02d}" for n in range(5, 12)]
REG = ROOT / "data/Training/court_registration.csv"
SETTINGS = {"current": dict(cdt=8, ambiguity=1.0), "loose": dict(cdt=20, ambiguity=0.0)}
QT = str(ROOT / "outputs/reports/overlap_rates.json")
QB = "0:0.9999,0.001:0.9996,0.2:0.9926,0.4:0.9375,0.5:0.8615,0.6:0.8215"
VARIANTS = {  # name: (label source, team params)
    "off": (None, {}),
    "s0.5": ("colour", dict(team_scale=0.5)),
    "s1": ("colour", dict(team_scale=1.0)),
    "s2": ("colour", dict(team_scale=2.0)),
    "s3": ("colour", dict(team_scale=3.0)),
    "s4": ("colour", dict(team_scale=4.0)),
    "s6": ("colour", dict(team_scale=6.0)),
    "s4_lo0": ("colour", dict(team_scale=4.0, team_learn_overlap=0.0)),
    "s4_lo.25": ("colour", dict(team_scale=4.0, team_learn_overlap=0.25)),
    "s4_cap20": ("colour", dict(team_scale=4.0, team_cap=20.0)),
    "s4_lo.25_cap20": ("colour", dict(team_scale=4.0, team_learn_overlap=0.25, team_cap=20.0)),
    "off_nms.5": (None, dict(nms_iou=0.5)),
    "off_nms.7": (None, dict(nms_iou=0.7)),
    "s6_nms.5": ("colour", dict(team_scale=6.0, nms_iou=0.5)),
    "s6_nms.6": ("colour", dict(team_scale=6.0, nms_iou=0.6)),
    "s6_nms.7": ("colour", dict(team_scale=6.0, nms_iou=0.7)),
    "s6_lo.25": ("colour", dict(team_scale=6.0, team_learn_overlap=0.25)),
    "s6_lo.25_nms.6": ("colour", dict(team_scale=6.0, team_learn_overlap=0.25, nms_iou=0.6)),
    "s6_qt_nms.6": ("colour", dict(team_scale=6.0, q_table=QT, nms_iou=0.6)),
    "s4_qt": ("colour", dict(team_scale=4.0, q_table=QT)),
    "s6_qt": ("colour", dict(team_scale=6.0, q_table=QT)),
    "s4_qt_cap20": ("colour", dict(team_scale=4.0, q_table=QT, team_cap=20.0)),
    "s1_veto": ("colour", dict(team_scale=1.0, team_veto=5.0)),
    "s1_birth": ("colour", dict(team_scale=1.0, team_birth=1)),
    "s1_veto_birth": ("colour", dict(team_scale=1.0, team_veto=5.0, team_birth=1)),
    "perfect_s1 (diagnostic)": ("gt", dict(team_scale=1.0)),
    # Colour safeguard (docs/15). q by max IoU over all boxes, measured on the fine-tuned boxes of IMG_0105-0111
    "s6_raw": ("colour", dict(team_scale=6.0, overlap_raw=1)),
    "s6_qb": ("colour", dict(team_scale=6.0, overlap_raw=1, q_bins=QB)),
    "s6_qb_back.4": ("colour", dict(team_scale=6.0, overlap_raw=1, q_bins=QB, guard_iou=0.4, guard_front_px=20.0)),
    "s6_qb_back.6": ("colour", dict(team_scale=6.0, overlap_raw=1, q_bins=QB, guard_iou=0.6, guard_front_px=20.0)),
    "s6_qb_all.6": ("colour", dict(team_scale=6.0, overlap_raw=1, q_bins=QB, guard_iou=0.6)),
    "s6_back.4": ("colour", dict(team_scale=6.0, overlap_raw=1, guard_iou=0.4, guard_front_px=20.0)),
    "s6_lock30": ("colour", dict(team_scale=6.0, team_lock_frames=30)),
    "s6_qb_back.4_lock30": ("colour", dict(team_scale=6.0, overlap_raw=1, q_bins=QB, guard_iou=0.4, guard_front_px=20.0,
                                           team_lock_frames=30)),
}


DET_SOURCES = {  # name: (detection file, label file) per video
    "large": (lambda v: ROOT / f"outputs/colab_all/tracking_results/{v}/8f5defcb/detections/{v}__rfdetr-large__720p.txt",
              lambda v: ROOT / f"outputs/labels/{v}__rfdetr-large__720p.csv.gz"),
    # Fine-tuned on 0104-0107 and 0109-0111, so only IMG_0108 is a fair test
    "ft": (lambda v: ROOT / f"outputs/detections_ft/{v}__rfdetr-small-ft-471b3fff__720p.txt",
           lambda v: ROOT / f"outputs/labels/{v}__rfdetr-small-ft-471b3fff__720p.csv.gz"),
}


def det_path(video: str, source: str = "large") -> Path:
    return DET_SOURCES[source][0](video)


def gt_labels(video: str, dets: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic only: each box's team from the GT player it matches (IoU >= 0.5), no call otherwise."""
    bb, ids = io.load_bboxes(video), io.track_identities(video)
    gt = {f: g for f, g in bb.groupby("frame")}
    team = pd.Series(index=dets.index, dtype=object)
    for f, g in dets.groupby("frame"):
        G = gt.get(f)
        if G is None:
            continue
        idx, _ = match(g[["x", "y", "w", "h"]].to_numpy(), G[["x", "y", "w", "h"]].to_numpy())
        tids = G.track_id.to_numpy()
        hit = [(r, ids[tids[k]][0]) for r, k in zip(g.index, idx) if k >= 0 and tids[k] in ids]  # skip stray GT tracks
        if hit:
            team.loc[[r for r, _ in hit]] = [t for _, t in hit]
    return pd.DataFrame({"team": team}).rename_axis("det_row")


def wrong_split(video: str, t: pd.DataFrame) -> tuple[int, int]:
    """Wrong-player rows (box on a player other than the track's main one), split by whether that player is on the other team."""
    o = dg.owners(video, t.reset_index(drop=True))
    m = o[o.track >= 0]
    main = m.groupby("track").player.agg(lambda s: s.mode().iloc[0])
    team = {k: v[0] for k, v in io.track_identities(video).items()}
    w = m[m.player != m.track.map(main)]
    cross = (w.player.map(team) != w.track.map(main).map(team)).to_numpy()
    return int(cross.sum()), int((~cross).sum())


def swap_stats(video: str, t: pd.DataFrame, labels) -> dict:
    """Cross-team episodes: runs of frames a track spends on an other-team player (gaps <= cross_team.GAP joined).
    Their length is how long the swap lasted before the track got back or ended."""
    if labels is None:
        t = t.assign(team=0, belief=0.0, det_row=-1)
    ep = episodes(rows_with_owner(video, t), None, None)
    if not len(ep):
        return dict(swaps=0, swaps_over_100=0, swap_mean=0.0, swap_max=0)
    return dict(swaps=len(ep), swaps_over_100=int((ep.rows > 100).sum()), swap_mean=float(ep.rows.mean()),
                swap_max=int(ep.rows.max()))


def job(a: tuple) -> dict:
    video, setting, variant, max_frame, dsrc = a
    try:
        source, extra = VARIANTS[variant]
        p = replace(Params(), **SETTINGS[setting], **extra)
        dets = load_dets(det_path(video, dsrc))
        n = int(dets.frame.max()) if not max_frame else max_frame
        dets = dets[dets.frame <= n]
        labels = None
        if source == "colour":
            labels = load_labels(DET_SOURCES[dsrc][1](video), det_path(video, dsrc))
        elif source == "gt":
            labels = gt_labels(video, dets)
        H, _ = io.load_homographies(REG)
        t = track(dets, H, p, n, labels)
        r = dg.diagnose(video, t)
        cross, same = wrong_split(video, t)
        keep = ("ids", "AssA", "IDF1", "HOTA", "DetA", "covered", "purity", "wrong_frames", "TI", "tracks_per_player")
        res = dict(video=video, setting=setting, variant=variant, max_frame=max_frame, dets=dsrc,
                   **{k: float(r[k]) for k in keep}, wrong_cross=cross, wrong_same=same, **swap_stats(video, t, labels))
    except Exception:
        res = dict(video=video, setting=setting, variant=variant, error=traceback.format_exc()[-1500:])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", default=VIDEOS)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--max-frame", type=int, default=0, help="only frames up to this (smoke test); GT is not cut, so coverage is low")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/reports/team_sweep.jsonl")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--dets", choices=list(DET_SOURCES), default="large", help="which detector's boxes (and their labels)")
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    a = ap.parse_args()
    jobs = [(v, s, k, a.max_frame, a.dets) for v in a.videos for s in a.settings for k in a.variants]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.unlink(missing_ok=True)
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job, jobs), 1):
            with open(a.out, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(i, len(jobs), res["video"], res["setting"], res["variant"], "ERROR" if "error" in res else
                  f"AssA {res['AssA']:.1f} IDF1 {res['IDF1']:.1f} TI {res['TI']:.3f}", flush=True)
    summarise(a.out)


def summarise(path: Path) -> None:
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    if "error" in d:
        bad = d[d.error.notna()]
        for e in bad.itertuples():
            print("ERROR", e.video, e.setting, e.variant, e.error[-400:])
        d = d[d.error.isna()]
    cols = ["TI", "AssA", "IDF1", "purity", "covered", "wrong_cross", "wrong_same", "swaps", "swaps_over_100", "swap_mean", "tracks_per_player"]
    print(d.groupby(["setting", "variant"], sort=False)[cols].mean().round(3).to_string())
    off = d[d.variant == "off"].set_index(["video", "setting"])
    for col in ("AssA", "TI"):
        delta = d.set_index(["video", "setting"]).assign(base=off[col])
        delta = delta.assign(gain=delta[col] - delta.base).reset_index()
        wins = delta.groupby(["setting", "variant"], sort=False).gain.agg(["mean", "min", lambda g: int((g > 0).sum())])
        print(f"\n{col} change vs off (mean, worst video, videos improved):\n", wins.round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarise":
        summarise(Path(sys.argv[2]))
    else:
        main()
