#!/usr/bin/env python
"""Accuracy of team-colour calls for clear vs overlapped boxes (q_clear, q_overlap for run_ucmc.attach_teams).

  uv run python src/tracker/team_rates.py [--videos IMG_0105 ...] [--max-frame N]

A call = labelled box with team A/B. Correct if it matches a GT player (IoU>=0.5) of that team. Overlapped = max IoU with
another labelled box in the frame >= OVERLAP_IOU (same set and threshold as attach_teams). Laplace (+1/+2) estimates.
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
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src" / "tracker")]
import eval_detections as ed  # noqa: E402
from trackid import io  # noqa: E402

OVERLAP_IOU = 0.2
VIDEOS = [f"IMG_{i:04d}" for i in range(105, 112)]
KEYS = ["clear", "over", "over_same", "over_cross", "nogt_clear", "nogt_over"]


def det_path(v: str) -> Path:
    return ROOT / f"outputs/colab_all/tracking_results/{v}/8f5defcb/detections/{v}__rfdetr-large__720p.txt"


def one_video(args: tuple[str, int | None]) -> tuple[str, dict]:
    v, max_frame = args
    dets = ed.load_dets(det_path(v))
    lab = pd.read_csv(ROOT / f"outputs/labels/{v}__rfdetr-large__720p.csv.gz")
    if max_frame:
        lab = lab[lab.frame <= max_frame]
    bb = io.load_bboxes(v)
    ident = io.track_identities(v)
    gt_team = {t: tm for t, (tm, _) in ident.items()}
    c = {k: [0, 0] for k in KEYS}  # [n, correct]
    gby = {f: g for f, g in bb.groupby("frame")}
    for f, g in lab.groupby("frame"):
        boxes = dets.loc[g.det_row.to_numpy(), ["x", "y", "w", "h"]].to_numpy(float)
        m = ed.iou_matrix(boxes, boxes)
        np.fill_diagonal(m, 0)
        over = m.max(1) if len(g) > 1 else np.zeros(len(g))
        partner = m.argmax(1) if len(g) > 1 else np.zeros(len(g), int)
        gt = gby.get(f)
        owner = np.full(len(g), -1)  # GT track_id matched to each box
        if gt is not None and len(gt):
            idx, _ = ed.match(gt[["x", "y", "w", "h"]].to_numpy(float), boxes)
            for tid, d in zip(gt.track_id.to_numpy(), idx):
                if d >= 0:
                    owner[d] = tid
        teams = g.team.to_numpy(object)
        for i in range(len(g)):
            if teams[i] not in ("A", "B"):
                continue
            ov = over[i] >= OVERLAP_IOU
            if owner[i] < 0:
                c["nogt_over" if ov else "nogt_clear"][0] += 1
                continue
            ok = int(gt_team.get(owner[i]) == teams[i])
            ks = ["over" if ov else "clear"]
            if ov:
                po = owner[partner[i]]
                if po >= 0 and gt_team.get(po) is not None and gt_team.get(owner[i]) is not None:
                    ks.append("over_cross" if gt_team[po] != gt_team[owner[i]] else "over_same")
            for k in ks:
                c[k][0] += 1
                c[k][1] += ok
    return v, c


def q(n: int, k: int) -> float:
    return (k + 1) / (n + 2)


def fmt(c: dict, k: str) -> str:
    n, ok = c[k]
    return f"{n:7d} {ok / n * 100 if n else float('nan'):6.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=VIDEOS)
    ap.add_argument("--max-frame", type=int, default=None)
    a = ap.parse_args()
    with Pool(min(len(a.videos), 7)) as p:
        res = dict(p.map(one_video, [(v, a.max_frame) for v in a.videos]))
    pool = {k: [sum(res[v][k][0] for v in res), sum(res[v][k][1] for v in res)] for k in KEYS}
    print(f"{'video':9s} {'clear':>15s} {'overlap':>15s} {'ov same':>15s} {'ov cross':>15s} {'noGT share':>10s} {'q_clear':>8s} {'q_over':>8s}  (LOVO q)")
    out = {"videos": {}}
    rows = {**res, "POOLED": pool}
    for v, c in rows.items():
        others = {k: [pool[k][0] - c[k][0], pool[k][1] - c[k][1]] for k in KEYS} if v != "POOLED" else pool
        tot = sum(c[k][0] for k in ("clear", "over", "nogt_clear", "nogt_over"))
        nogt = (c["nogt_clear"][0] + c["nogt_over"][0]) / max(tot, 1)
        qc, qo = q(*others["clear"]), q(*others["over"])
        print(f"{v:9s} {fmt(c, 'clear'):>15s} {fmt(c, 'over'):>15s} {fmt(c, 'over_same'):>15s} {fmt(c, 'over_cross'):>15s} {nogt*100:9.1f}% {qc:8.4f} {qo:8.4f}")
        out["videos" if v != "POOLED" else "pooled"] = out.get("videos", {}) if v != "POOLED" else {}
        entry = {k: {"n": c[k][0], "correct": c[k][1]} for k in KEYS}
        entry.update(no_gt_share=nogt, q_clear=qc, q_overlap=qo)
        if v == "POOLED":
            out["pooled"] = entry
        else:
            out["videos"][v] = entry
    out["note"] = "per-video q_* are leave-one-video-out; pooled q_* use all videos"
    dst = ROOT / "outputs/reports/team_rates.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
