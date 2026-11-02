#!/usr/bin/env python
"""Run the court-space UCMCTrack on cached detections and keep only what is inside the court.

  python src/tracker/run_ucmc.py DETECTIONS.txt REGISTRATION.csv OUT.txt [--margin 0] [--input-margin 100]

Detections up to --input-margin cm outside the court are tracked, so a player stepping over the line keeps their track,
but only rows within --margin cm of the court are written. Output, positions in cm:
  frame,track_id,court_x,court_y,raw_x,raw_y,kind,conf,box_x,box_y,box_w,box_h
kind is det (matched a detection) or coast (predicted, no detection; raw and box fields are -1). By default no coast
rows are written (--emit-coast 0), though tracks stay alive unobserved for --cdt frames. Everything here is causal: no
future frames, smoothing, interpolation or track linking.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from eval_detections import iou_matrix, load_dets  # noqa: E402
from tracker.kalman import KalmanTracker, TrackStatus  # noqa: E402
from tracker.mapper import CM_PER_M, HomographyMapper  # noqa: E402
from tracker.ucmc import UCMCTrack  # noqa: E402
from trackid import io  # noqa: E402

FPS = 30000 / 1001
COLUMNS = ["frame", "id", "x", "y", "raw_x", "raw_y", "kind", "conf", "bx", "by", "bw", "bh"]


@dataclass
class Params:
    """Tuned on IMG_0106 only, trading identity purity against track count. Check on other games before relying on them."""
    conf: float = 0.1
    high_score: float = 0.7
    a: float = 20.0
    wx: float = 25.0
    wy: float = 25.0
    vmax: float = 10.0  # m/s
    cdt: int = 8  # frames a track may go unobserved before deletion (deleted on the next miss)
    emit_coast: int = 0  # frames of unobserved track to write out; retention can be longer than this
    gate: float = 0.0  # squared Mahalanobis limit, 0 = off
    ambiguity: float = 1.0  # cost margin below which a match is not committed, 0 = off
    recency: int = 0  # 1 = match recently seen tracks first
    birth_hits: int = 3  # observations needed to confirm a track
    conf_noise_scale: float = 3.0  # above 0: noise by confidence (mapper.CONF_NOISE) times this, replacing the ratios
    u_ratio: float = 0.02  # fixed foot-point noise as a fraction of box height; used only if conf_noise_scale is 0
    v_ratio: float = 0.02
    margin: float = 0.0
    input_margin: float = 100.0
    # Team labels (used only with --labels). See UCMCTrack for what each does.
    team_scale: float = 0.0  # multiplies the team mismatch penalty, 0 = off
    # Chance a team call is right, on a box overlapping no other (IoU < 0.2) and on an overlapped one. Pooled over
    # IMG_0105-0111 RF-DETR boxes by src/tracker/team_rates.py; leave-one-video-out gives 0.9992-0.9993 and 0.937-0.942.
    q_clear: float = 0.9993
    q_overlap: float = 0.939
    team_cap: float = 10.0  # limit on a track's team log-odds
    team_decay: float = 1.0  # track log-odds is multiplied by this before each matched box is added
    team_veto: float = 0.0  # above 0: a clear box can't join a track this sure (log-odds) of the other team
    team_birth: int = 0  # 1 = no new track from a box whose team already has 3 sure tracks matched this frame
    team_learn_overlap: float = 1.0  # weight of an overlapped box's call in the track's log-odds (its penalty is unchanged)
    nms_iou: float = 0.0  # above 0: before tracking, drop a box overlapping a higher-confidence box by at least this IoU
    q_table: str = ""  # JSON from src/tracker/overlap_rates.py: q by overlap and neighbour's call, replacing q_clear/q_overlap
    # Colour safeguard. A crop overlapped by another player mostly shows whoever is in front, so its call is unreliable.
    overlap_raw: int = 0  # 1 = measure overlap on every box in the file (conf >= LABEL_MIN_CONF), before the court filter
    #                       and NMS, so dropping a neighbour can't make a contaminated crop count as clear
    q_bins: str = ""  # "lo:q,lo:q,...": chance a call is right by the box's max IoU, replacing q_clear/q_overlap
    guard_iou: float = 0.0  # above 0: a box overlapping another at least this much gets no call (no penalty, no learning)
    guard_front_px: float = -1.0  # >= 0: except the front box, whose feet are lower in the image than every such
    #                               neighbour's by more than this many px. < 0: no call for any of them
    team_lock_frames: int = 0  # above 0: a track sure of one team for this many matched frames in a row keeps it for good


@dataclass
class Det:
    y: np.ndarray
    R: np.ndarray
    conf: float
    bb_width: float
    bb_height: float
    box: np.ndarray
    track_id: int = 0
    team: int = 0  # +1 team A, -1 team B, 0 no call (none, unsure, or no label)
    evidence: float = 0.0  # logit of the chance the call is right; 0 with no call
    row: int = -1  # 0-based line in the detection file
    overlapped: bool = False  # IoU >= OVERLAP_IOU with another box this frame (set only with labels)


TEAM_SIGN = {"A": 1, "B": -1}
OVERLAP_IOU = 0.2  # same threshold as src/team: a box overlapping another this much is "overlapped"


def max_iou(boxes: np.ndarray) -> np.ndarray:
    """Per box, the highest IoU with any other box in the same frame (boxes xywh)."""
    if len(boxes) < 2:
        return np.zeros(len(boxes))
    m = iou_matrix(boxes, boxes)
    np.fill_diagonal(m, 0)
    return m.max(1)


def load_labels(path: Path | str, detections: Path | str) -> pd.DataFrame:
    """Team label file from src/team, checked against the detection file it was made from. Index = det_row."""
    path = Path(path)
    want = Path(str(path) + ".sha256").read_text().split()[0]
    have = hashlib.sha256(Path(detections).read_bytes()).hexdigest()
    if want != have:
        sys.exit(f"{path} was made from a different detection file ({want[:12]} != {have[:12]})")
    lab = pd.read_csv(path, usecols=["det_row", "team"], dtype={"team": str})
    return lab.set_index("det_row")


def q_from_table(path: Path | str, iou: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Chance a call is right, looked up by (neighbour relation, IoU bin) from src/tracker/overlap_rates.py's JSON."""
    import json
    t = json.loads(Path(path).read_text())
    q = np.full(len(iou), np.nan)
    for c in t["table"]:
        q[(rel == c["rel"]) & (iou >= c["lo"]) & (iou < c["hi"])] = c["q"]
    return np.nan_to_num(q, nan=0.5)  # a cell the table lacks: no evidence


LABEL_MIN_CONF = 0.1  # src/team labels boxes down to this confidence


def overlap_info(raw: pd.DataFrame, guard_iou: float) -> pd.DataFrame:
    """Per box of the whole detection file: max IoU with any other box in its frame, and whether it is in front of every
    box it overlaps by >= guard_iou (by how far its feet are below theirs, in px; inf if it overlaps none)."""
    raw = raw[raw.conf >= LABEL_MIN_CONF]
    over = pd.Series(0.0, index=raw.index)
    lead = pd.Series(np.inf, index=raw.index)
    for _, g in raw.groupby("frame"):
        if len(g) < 2:
            continue
        b = g[["x", "y", "w", "h"]].to_numpy()
        m = iou_matrix(b, b)
        np.fill_diagonal(m, 0)
        over.loc[g.index] = m.max(1)
        if guard_iou:
            feet = b[:, 1] + b[:, 3]
            gap = np.where(m >= guard_iou, feet[:, None] - feet[None, :], np.inf)
            lead.loc[g.index] = gap.min(1)
    return pd.DataFrame({"over": over, "lead": lead})


def q_from_bins(spec: str, iou: np.ndarray) -> np.ndarray:
    """spec "lo:q,lo:q,...", bins sorted by lo; each box gets the q of the last bin whose lo <= its IoU."""
    bins = sorted((float(lo), float(q)) for lo, q in (s.split(":") for s in spec.split(",")))
    q = np.full(len(iou), bins[0][1])
    for lo, v in bins:
        q[iou >= lo] = v
    return q


def attach_teams(dets: pd.DataFrame, labels: pd.DataFrame, q_clear: float, q_overlap: float, q_table: str = "",
                 info: pd.DataFrame | None = None, q_bins: str = "", guard_iou: float = 0.0,
                 guard_front_px: float = -1.0) -> pd.DataFrame:
    """Adds team (+1/-1/0) and evidence (logit of the call being right) to each detection.

    Without q_table: q_clear or q_overlap by whether the box overlaps another (IoU >= 0.2). With q_table: by the IoU with its
    most-overlapping neighbour and whether that neighbour got the same call, the opposite call or none.
    dets must still have its original row numbers as index. Boxes without a label get no call. Overlap is measured among
    the boxes the tracker sees, the same set src/team labelled.
    """
    team = dets.index.map(labels.team.map(TEAM_SIGN)).to_numpy(dtype=float)
    team = np.nan_to_num(team).astype(int)
    over = np.zeros(len(dets))
    rel = np.full(len(dets), "none", dtype=object)
    pos = {r: i for i, r in enumerate(dets.index)}
    for _, g in dets.groupby("frame"):
        i = np.array([pos[r] for r in g.index])
        if len(g) < 2:
            continue
        m = iou_matrix(*(g[["x", "y", "w", "h"]].to_numpy(),) * 2)
        np.fill_diagonal(m, 0)
        over[i] = m.max(1)
        nb = team[i[m.argmax(1)]]
        rel[i] = np.where(over[i] == 0, "none", np.where(nb == 0, "nocall", np.where(nb == team[i], "same", "other")))
    if info is not None:  # overlap among all boxes, including ones the tracker dropped
        over = info.over.reindex(dets.index).fillna(0.0).to_numpy()
    if q_table:
        q = q_from_table(q_table, over, rel)
    elif q_bins:
        q = q_from_bins(q_bins, over)
    else:
        q = np.where(over >= OVERLAP_IOU, q_overlap, q_clear)
    if guard_iou:
        heavy = over >= guard_iou
        if guard_front_px >= 0 and info is not None:
            heavy &= ~(info.lead.reindex(dets.index).to_numpy() > guard_front_px)
        team = np.where(heavy, 0, team)
    evidence = np.where(team != 0, np.log(q / (1 - q)), 0.0)
    return dets.assign(team=team, evidence=evidence, overlapped=over >= OVERLAP_IOU)


def nms(dets: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Greedy per-frame NMS by confidence, keeping the original row index (labels are keyed by it)."""
    keep = []
    for _, g in dets.groupby("frame"):
        g = g.sort_values("conf", ascending=False, kind="stable")
        b = g[["x", "y", "w", "h"]].to_numpy()
        m = iou_matrix(b, b)
        alive = np.ones(len(g), bool)
        for i in range(len(g)):
            if alive[i]:
                alive[i + 1:] &= m[i, i + 1:] < thr
        keep.append(g.index[alive])
    return dets.loc[np.sort(np.concatenate(keep))] if keep else dets


def in_court(x: np.ndarray, y: np.ndarray, margin: float) -> np.ndarray:
    return (x >= -margin) & (x <= io.COURT_W + margin) & (y >= -margin) & (y <= io.COURT_H + margin)


TEAM_COLUMNS = ["team", "belief", "det_row"]  # added only when labels are given: the box's call, the track's log-odds


def track(dets: pd.DataFrame, H: np.ndarray, p: Params, n_frames: int, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Returns one row per output track position, inside the court only.

    dets keeps its original row numbers as index (load_dets does this). With labels, each box carries its team call and
    the output gets TEAM_COLUMNS; without, the output is exactly what it was before team labels existed.
    """
    KalmanTracker.count = 1
    mapper = HomographyMapper(H, p.u_ratio, p.v_ratio, p.conf_noise_scale)
    tracker = UCMCTrack(p.a, p.a, p.wx, p.wy, p.vmax, p.cdt, FPS, p.high_score, p.gate, p.ambiguity, bool(p.recency), p.birth_hits,
                        team_scale=p.team_scale, team_cap=p.team_cap, team_decay=p.team_decay, team_veto=p.team_veto,
                        team_birth=bool(p.team_birth), team_learn_overlap=p.team_learn_overlap,
                        team_lock_frames=p.team_lock_frames)
    teams = labels is not None
    info = overlap_info(dets, p.guard_iou) if teams and (p.overlap_raw or p.guard_iou) else None
    feet = io.apply_h(H, np.c_[dets.x + dets.w / 2, dets.y + dets.h])
    dets = dets[(dets.conf >= p.conf) & in_court(feet[:, 0], feet[:, 1], p.input_margin)]
    if p.nms_iou:
        dets = nms(dets, p.nms_iou)
    if teams:
        dets = attach_teams(dets, labels, p.q_clear, p.q_overlap, p.q_table, info, p.q_bins, p.guard_iou, p.guard_front_px)
    else:
        dets = dets.assign(team=0, evidence=0.0, overlapped=False)
    dets = dets.assign(row=dets.index)
    by = {f: d[["x", "y", "w", "h", "conf", "team", "evidence", "row", "overlapped"]].to_numpy(dtype=float) for f, d in dets.groupby("frame")}
    rows = []
    for f in range(1, n_frames + 1):
        frame_dets = []
        for b in by.get(f, ()):
            y, R = mapper.mapto(b[:4], float(b[4]))
            frame_dets.append(Det(y, R, float(b[4]), b[2], b[3], b[:4], team=int(b[5]), evidence=float(b[6]), row=int(b[7]),
                                  overlapped=bool(b[8])))
        tracker.update(frame_dets, f)
        for t in tracker.trackers:
            if t.status == TrackStatus.Tentative:
                continue
            x, y = t.kf.x[0, 0] * CM_PER_M, t.kf.x[2, 0] * CM_PER_M
            if t.detidx >= 0:
                d = frame_dets[t.detidx]
                row = (f, t.id, x, y, d.y[0, 0] * CM_PER_M, d.y[1, 0] * CM_PER_M, "det", d.conf, *d.box)
                extra = (d.team, t.team_belief, d.row)
            elif t.death_count <= p.emit_coast:
                row = (f, t.id, x, y, -1.0, -1.0, "coast", -1.0, -1.0, -1.0, -1.0, -1.0)
                extra = (0, t.team_belief, -1)
            else:
                continue
            rows.append(row + extra if teams else row)
    out = pd.DataFrame(rows, columns=COLUMNS + TEAM_COLUMNS if teams else COLUMNS)
    return out[in_court(out.x.to_numpy(), out.y.to_numpy(), p.margin)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("detections")
    ap.add_argument("registration")
    ap.add_argument("out", type=Path)
    ap.add_argument("--labels", type=Path, help="team label file from src/team for these detections; adds team columns")
    for name, default in vars(Params()).items():
        ap.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    a = ap.parse_args()
    p = Params(**{k: getattr(a, k) for k in vars(Params())})

    dets = load_dets(a.detections)
    H, _ = io.load_homographies(a.registration)
    labels = load_labels(a.labels, a.detections) if a.labels else None
    out = track(dets, H, p, int(dets.frame.max()), labels)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, header=False, float_format="%.2f")
    print(f"{a.out}: {len(out)} rows, {out.id.nunique()} tracks")


if __name__ == "__main__":
    main()
