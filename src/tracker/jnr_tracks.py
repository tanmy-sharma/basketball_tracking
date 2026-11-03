#!/usr/bin/env python
"""Tracker extension: attach jersey numbers (JNR) to tracks, split tracks where the number switches, and stitch fragments
of the same player into one id. The new ids are written over the whole file, before and after each read.

  python src/tracker/jnr_tracks.py TRACKS.txt JNR.csv.gz --video IMG_0114 --out OUT.txt
         [--format auto|ucmc|mot] [--submission SUB.txt --n-frames N [--score]]

Inputs
  TRACKS.txt  one row per (frame, track) that STORES THE DETECTION BOX the tracker matched in that frame next to its own,
              so each tracker row joins exactly one JNR row. Two layouts, headerless:
                ucmc  what src/tracker/run_ucmc.py writes: frame,id,court_x,court_y,raw_x,raw_y,kind,conf,bx,by,bw,bh
                      [,team,belief,det_row]. bx..bh is the detection box (-1 on coast rows); team is the box's colour
                      call (+1 A, -1 B, 0 none). Positions are court cm, so stitching can check running speed.
                mot   frame,id,x,y,w,h,conf,dx,dy,dw,dh[,team]: tracker box, then the matched detection box (-1 if none),
                      full-resolution pixels; optional team A/B or +1/-1/0. No speed check (pixels, not court).
  JNR.csv.gz  src/infer/jnr_video.py output for the same detection file: x,y,w,h are the detection boxes, elig, p_<n>.

Steps
  1. join: per frame, each tracker row's detection box to the JNR row with the same box (max coordinate gap < --tol px).
  2. per read: P(team) from the box's colour call; P(number | team) = ViTSTR probabilities renormalised over that team's
     3 roster numbers. Identity = (team, number), so 6 and 9 on both teams of IMG_0114 stay distinct by team.
  3. split: a track whose confident reads change identity persistently (binary segmentation on eligible reads, at least
     --min-side reads each side, log-likelihood gain >= --split-gain) is cut midway between the two runs of reads.
  4. vote per piece: mean P(number | team) over reads with elig >= 0.5 (all non-degenerate reads if none), times the
     piece's P(team) from all its colour calls. Labelled if >= --min-reads reads, top score >= --min-score and at least
     --name-min-eligible eligible reads (a vote from unreadable reads only is recorded but never names a piece).
  5. assign: strongest pieces first, each takes its best identity that no placed piece holds at the same time (up to
     --max-overlap frames) and that it could have run to (<= 10 m/s plus slack, ucmc only); else its next best, but only
     if that also scores >= --min-score, otherwise it stays unnamed.
  6. bridge: an unlabelled piece between two pieces of one identity, which fits in time, speed and team, takes it if only
     one identity fits, and not one its own clear vote scores below --bridge-veto.
  7. write: same file, same columns and formatting, id replaced. Identity ids are 1..6 (team A numbers, then B, in roster
     order); other pieces get 100+. Where two pieces of one id overlap, the weaker piece's rows in those frames are dropped.
     OUT.segments.csv explains every piece; OUT.scores.json holds per-id number scores for identity.py.
  8. --submission: identity.make_rows on OUT, without the unnamed pieces that have fewer than --export-min-eligible
     eligible reads (stats no_export_ids). Unnamed pieces with readable reads still get a number there. Positions are
     then smoothed along the camera's depth direction (tracker/smooth.py, --smooth-sigma frames, 0 = off).
Nothing here reads ground truth; --score only grades a training video afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT)]

from trackid import io  # noqa: E402

from src.core.constants import NUMBERS  # noqa: E402

FPS = 30000 / 1001
UCMC_BOX = slice(8, 12)
MOT_BOX = slice(7, 11)
IDENTITY_ID_BASE, OTHER_ID_BASE = 1, 100
SEG_STRIDE = 1_000_000  # matches identity.py: piece k of track t is t + k * SEG_STRIDE


@dataclass
class Params:
    tol: float = 1.0  # px: detection box in the track file vs in the JNR file (JNR keeps 5 significant digits)
    elig: float = 0.5  # eligibility cut-off, as validated on IMG_0108
    team_q: float = 0.95  # P(call right) for one colour call when scoring a single read
    team_nats: float = 0.2  # log-odds per colour call when pooling a whole piece (calls are correlated)
    p_floor: float = 0.01  # per-read probability floor, so one bad crop can't veto an identity
    min_side: int = 6  # eligible reads needed on each side of a split
    split_gain: float = 8.0  # nats a split must gain over one identity for the whole track
    min_reads: int = 3  # reads a piece needs to be labelled by its own vote
    min_score: float = 0.6  # top identity score a piece needs to be labelled
    max_overlap: int = 30  # frames two pieces of one identity may share: tracker duplicates at births, and a short swap at
    #                        a track's end onto someone else (IMG_0104 track 42: 11 frames; IMG_0109 track 32: 19)
    vmax: float = 1000.0  # cm/s, ucmc only
    slack: float = 150.0  # cm added to the reach check
    bridge_gap: int = 90  # frames: bridge only between neighbours this close in time
    # Identity admission (docs/20, audit/astra_final_results_verification_2026-11-03.md). 0 restores the old behaviour.
    name_min_eligible: int = 1  # eligible reads a piece needs to be named by its own vote. Unreadable reads renormalised
    #                             over 3 roster numbers look certain (a sideline non-player took B1 on IMG_0115)
    bridge_veto: float = 0.1  # a bridge may not give a piece an identity its own labelled vote scores below this
    export_min_eligible: int = 1  # --submission: unnamed pieces with fewer eligible reads are left out, not guessed


# ----------------------------------------------------------------------------- reading
def read_tracks(path: Path | str, fmt: str = "auto") -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(raw string table to rewrite, numeric view, format). The view has frame,id,fx,fy (foot / court position),
    dx,dy,dw,dh (detection box, NaN if none) and call (+1 A, -1 B, 0)."""
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if fmt == "auto":
        fmt = "ucmc" if raw.shape[1] >= 12 and raw[6].isin(["det", "coast"]).all() else "mot"
    num = lambda c: pd.to_numeric(raw[c], errors="coerce").to_numpy(float)  # noqa: E731
    v = pd.DataFrame({"frame": num(0).astype(int), "id": num(1).astype(int)})
    if fmt == "ucmc":
        v["fx"], v["fy"] = num(2), num(3)
        box, call_col = UCMC_BOX, 12
    elif fmt == "mot":
        if raw.shape[1] < 11:
            raise SystemExit(f"{path}: mot layout needs the matched detection box in columns 8-11 (got {raw.shape[1]} columns)")
        v["fx"], v["fy"] = num(2) + num(4) / 2, num(3) + num(5)
        box, call_col = MOT_BOX, 11
    else:
        raise ValueError(fmt)
    b = np.stack([num(c) for c in range(box.start, box.stop)], 1)
    b[(b[:, 2] <= 0) | (b[:, 3] <= 0)] = np.nan
    v[["dx", "dy", "dw", "dh"]] = b
    if raw.shape[1] > call_col:
        c = raw[call_col].str.strip().replace({"A": "1", "B": "-1", "": "0"})
        v["call"] = pd.to_numeric(c, errors="coerce").fillna(0).clip(-1, 1).astype(int).to_numpy()
    else:
        v["call"] = 0
    return raw, v, fmt


def read_jnr(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"read": str, "status": str, "facing": str})


def find_roster(video: str, roster: Path | None) -> dict[str, list[str]]:
    paths = [roster] if roster else [io.FINAL / "games_roster.csv", io.TRAIN / "games_roster.csv"]
    for p in paths:
        r = io.load_roster(p)
        if video in r:
            return r[video].teams
    raise SystemExit(f"{video} is in none of {paths}")


# ----------------------------------------------------------------------------- 1. join
def attach(tracks: pd.DataFrame, jnr: pd.DataFrame, tol: float = 1.0) -> np.ndarray:
    """For each track row, the JNR row index with the same detection box in the same frame, or -1. One to one."""
    out = np.full(len(tracks), -1, np.int64)
    jb = jnr[["x", "y", "w", "h"]].to_numpy(float)
    jby = {int(f): idx for f, idx in jnr.groupby("frame").indices.items()}
    tb = tracks[["dx", "dy", "dw", "dh"]].to_numpy(float)
    for f, ti in tracks.groupby("frame").indices.items():
        ji = jby.get(int(f))
        if ji is None:
            continue
        ti = ti[~np.isnan(tb[ti, 0])]
        if not len(ti):
            continue
        gap = np.abs(tb[ti, None, :] - jb[None, ji, :]).max(-1)  # (tracks, jnr rows)
        for a in np.argsort(gap.min(1)):
            b = int(gap[a].argmin())
            if gap[a, b] < tol:
                out[ti[a]] = ji[b]
                gap[:, b] = np.inf
    return out


# ----------------------------------------------------------------------------- 2. per-read identity evidence
def identities(teams: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [(t, j) for t in sorted(teams) for j in teams[t]]


def number_given_team(jnr: pd.DataFrame, teams: dict[str, list[str]]) -> np.ndarray:
    """(rows, identities): ViTSTR probability renormalised over the identity's team roster. NaN on degenerate rows."""
    p = jnr[[f"p_{n}" for n in NUMBERS]].to_numpy(float)
    col = {n: i for i, n in enumerate(NUMBERS)}
    out = []
    for t in sorted(teams):
        sub = p[:, [col[j] for j in teams[t]]]
        out.append(sub / sub.sum(1, keepdims=True))
    return np.concatenate(out, 1)


def team_prob(calls: np.ndarray, ids: list[tuple[str, str]], nats: float) -> np.ndarray:
    """P(team of each identity) from summed colour calls (+1 A, -1 B). Teams other than A/B get no evidence."""
    s = float(np.sum(calls)) * nats
    pa = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    names = sorted({t for t, _ in ids})
    if names != ["A", "B"]:
        return np.full(len(ids), 1.0 / len(names))
    return np.array([pa if t == "A" else 1 - pa for t, _ in ids])


# ----------------------------------------------------------------------------- 3. split on identity change
def change_points(L: np.ndarray, min_side: int, gain_min: float) -> list[int]:
    """Binary segmentation of per-read log-probabilities L (reads, identities). Returns read indices starting a new piece."""
    n = len(L)
    if n < 2 * min_side:
        return []
    C = np.cumsum(L, 0)
    whole = C[-1].max()
    left, right = C[:-1], C[-1] - C[:-1]  # split after read i
    gain = left.max(1) + right.max(1) - whole
    differ = left.argmax(1) != right.argmax(1)
    i = np.arange(n - 1)
    ok = differ & (i >= min_side - 1) & (i <= n - 1 - min_side)
    if not ok.any():
        return []
    gain = np.where(ok, gain, -np.inf)
    t = int(gain.argmax())
    if gain[t] < gain_min:
        return []
    t += 1
    return (change_points(L[:t], min_side, gain_min) + [t]
            + [t + k for k in change_points(L[t:], min_side, gain_min)])


def split_by_number(v: pd.DataFrame, P: np.ndarray, p: Params, ids) -> np.ndarray:
    """seg id per track row. P: (rows, identities) per-read identity probability (NaN where no usable read)."""
    seg = v["id"].to_numpy().copy()
    usable = ~np.isnan(P[:, 0]) & v["eligible"].to_numpy()
    frames = v["frame"].to_numpy()
    for tid, idx in v.groupby("id").indices.items():
        idx = idx[np.argsort(frames[idx], kind="stable")]
        reads = idx[usable[idx]]
        cps = change_points(np.log(np.maximum(P[reads], p.p_floor)), p.min_side, p.split_gain)
        for piece, c in enumerate(cps, 1):
            cut = (frames[reads[c - 1]] + frames[reads[c]]) / 2  # the switch happened somewhere in between
            seg[idx[frames[idx] > cut]] = tid + piece * SEG_STRIDE
    return seg


# ----------------------------------------------------------------------------- 4. vote per piece
def piece_table(v: pd.DataFrame, NgT: np.ndarray, ids, p: Params) -> pd.DataFrame:
    rows = []
    has = v["jnr_row"].to_numpy() >= 0
    for sid, idx in v.groupby("seg").indices.items():
        g = v.iloc[idx].sort_values("frame")
        r = idx[has[idx]]
        r = r[~np.isnan(NgT[r, 0])]
        el = r[v["eligible"].to_numpy()[r]]
        use, source = (el, "eligible") if len(el) else (r, "all" if len(r) else "none")
        pt = team_prob(g["call"].to_numpy(), ids, p.team_nats)
        score = (NgT[use].mean(0) * pt) if len(use) else pt * 0
        if score.sum() > 0:
            score = score / score.sum()
        order = np.argsort(-score)
        top, second = float(score[order[0]]), float(score[order[1]]) if len(score) > 1 else 0.0
        rows.append(dict(seg=int(sid), orig_id=int(g["id"].iloc[0]), start=int(g.frame.iloc[0]), end=int(g.frame.iloc[-1]),
                         n_rows=len(g), n_reads=len(r), n_eligible=len(el), source=source,
                         p_team_a=float(pt[0]),  # ids are sorted by team, so pt[0] is the first team (A)
                         best=f"{ids[order[0]][0]}{ids[order[0]][1]}" if len(use) else "",
                         # below name_min_eligible the vote is kept for the record but cannot name the piece
                         score=top if len(use) and len(el) >= p.name_min_eligible else 0.0,
                         second=second if len(use) else 0.0,
                         strength=len(use) * (top - second) if len(use) else 0.0,
                         x0=g.fx.iloc[0], y0=g.fy.iloc[0], x1=g.fx.iloc[-1], y1=g.fy.iloc[-1],
                         **{f"s_{t}{j}": float(s) for (t, j), s in zip(ids, score)}))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 5/6. assign and bridge
def reachable(a: pd.Series, b: pd.Series, p: Params, court: bool) -> bool:
    """Could the player at the end of piece a be at the start of piece b (a ends before b starts)?"""
    if not court:
        return True
    gap = max(int(b.start) - int(a.end), 1)
    return float(np.hypot(b.x0 - a.x1, b.y0 - a.y1)) <= p.slack + p.vmax * gap / FPS


def _fits(sid, k, frames_of, placed, segs, p, court) -> bool:
    me = segs.loc[sid]
    for other in placed[k]:
        o = segs.loc[other]
        if len(np.intersect1d(frames_of[sid], frames_of[other], assume_unique=True)) > p.max_overlap:
            return False
        if o.end < me.start and not reachable(o, me, p, court):
            return False
        if me.end < o.start and not reachable(me, o, p, court):
            return False
    return True


def assign(segs: pd.DataFrame, frames_of: dict, ids, p: Params, court: bool) -> pd.DataFrame:
    segs = segs.set_index("seg", drop=False).copy()
    segs["identity"], segs["how"] = "", "none"
    placed: dict[int, list[int]] = {k: [] for k in range(len(ids))}
    sc = segs[[f"s_{t}{j}" for t, j in ids]].to_numpy()
    pos = {s: i for i, s in enumerate(segs.index)}
    labelled = segs[(segs.n_reads >= p.min_reads) & (segs.score >= p.min_score)]
    for sid in labelled.sort_values(["strength", "n_rows"], ascending=False).index:
        for rank, k in enumerate(np.argsort(-sc[pos[sid]])):
            # A blocked piece may only fall back to an identity its own reads support. Taking any positive score let one
            # blocked piece take another player's number, which blocked that player's piece in turn (IMG_0109: a rotation of
            # A4/A10/A12 over ~12,600 rows, docs/18). Unsupported: stay unnamed and leave it to the bridge.
            if sc[pos[sid], k] < p.min_score:
                break
            if _fits(sid, k, frames_of, placed, segs, p, court):
                placed[k].append(sid)
                segs.loc[sid, ["identity", "how"]] = [f"{ids[k][0]}{ids[k][1]}", "vote" if rank == 0 else "vote_2nd"]
                break
        if segs.loc[sid, "how"] == "none":  # a candidate whose supported identities were all taken
            segs.loc[sid, "how"] = "conflict"
    # bridge: unlabelled pieces between two placed pieces of one identity
    team_of = {k: ids[k][0] for k in range(len(ids))}
    first_team = sorted({t for t, _ in ids})[0]
    for sid in segs[segs.identity == ""].sort_values("n_rows", ascending=False).index:
        me, fits = segs.loc[sid], []
        voted = me.n_eligible >= p.min_reads and me.score >= p.min_score  # read clearly, but lost its own identity
        for k in range(len(ids)):
            pt = me.p_team_a if team_of[k] == first_team else 1 - me.p_team_a
            if pt < 0.5 or not placed[k]:
                continue
            if voted and sc[pos[sid], k] < p.bridge_veto:  # its own reads say another number (IMG_0115 piece 32: B1 0.99)
                continue
            prev = [o for o in placed[k] if segs.loc[o].end < me.start and me.start - segs.loc[o].end <= p.bridge_gap]
            nxt = [o for o in placed[k] if segs.loc[o].start > me.end and segs.loc[o].start - me.end <= p.bridge_gap]
            if prev and _fits(sid, k, frames_of, placed, segs, p, court):
                fits.append((k, 2 if nxt else 1))  # 2: neighbours on both sides; 1: a continuation only
        both = [k for k, s in fits if s == 2]
        pick = both if both else [k for k, _ in fits]
        if len(pick) == 1:
            k = pick[0]
            placed[k].append(sid)
            segs.loc[sid, ["identity", "how"]] = [f"{ids[k][0]}{ids[k][1]}", "bridge"]
    return segs.reset_index(drop=True)


# ----------------------------------------------------------------------------- 7. write
def new_ids(segs: pd.DataFrame, ids) -> dict[int, int]:
    ident_id = {f"{t}{j}": IDENTITY_ID_BASE + i for i, (t, j) in enumerate(ids)}
    out, nxt = {}, OTHER_ID_BASE
    for r in segs.sort_values(["start", "seg"]).itertuples():
        if r.identity:
            out[r.seg] = ident_id[r.identity]
        else:
            out[r.seg], nxt = nxt, nxt + 1
    return out


def run(tracks_path, jnr_path, teams, p: Params = Params(), fmt: str = "auto"):
    raw, v, fmt = read_tracks(tracks_path, fmt)
    jnr = read_jnr(jnr_path)
    ids = identities(teams)
    v["jnr_row"] = attach(v, jnr, p.tol)
    ok = v.jnr_row.to_numpy() >= 0
    j = v.jnr_row.to_numpy()[ok]
    NgT = np.full((len(v), len(ids)), np.nan)
    NgT[ok] = number_given_team(jnr.iloc[j], teams)
    elig = np.zeros(len(v), bool)
    elig[ok] = jnr["elig"].to_numpy(float)[j] >= p.elig
    v["eligible"] = elig
    # per-read identity probability, with the read's own colour call as the team prior
    call = v["call"].to_numpy()
    first = sorted({t for t, _ in ids})[0]
    pa = np.where(call > 0, p.team_q, np.where(call < 0, 1 - p.team_q, 0.5))
    pt = np.stack([pa if t == first else 1 - pa for t, _ in ids], 1)
    v["seg"] = split_by_number(v, NgT * pt, p, ids)
    segs = piece_table(v, NgT, ids, p)
    frames_of = {s: np.unique(v.frame.to_numpy()[i]) for s, i in v.groupby("seg").indices.items()}
    segs = assign(segs, frames_of, ids, p, court=(fmt == "ucmc"))
    m = new_ids(segs, ids)
    segs["new_id"] = segs.seg.map(m)
    v["new_id"] = v.seg.map(m)
    # one row per id per frame: keep the strongest piece where two pieces of one identity overlap
    rank = v.seg.map(dict(zip(segs.seg, segs.strength))).to_numpy()
    keep = (pd.DataFrame({"f": v.frame, "i": v.new_id, "r": -rank, "o": np.arange(len(v))})
            .sort_values(["f", "i", "r", "o"]).drop_duplicates(["f", "i"]).o.sort_values().to_numpy())
    out = raw.iloc[keep].copy()
    out[1] = v.new_id.to_numpy()[keep].astype(str)
    stats = dict(format=fmt, rows=len(v), rows_written=len(out), rows_dropped_overlap=len(v) - len(out),
                 rows_with_jnr=int(ok.sum()), rows_eligible=int(elig.sum()), tracks=int(v["id"].nunique()),
                 pieces=len(segs), splits=int(len(segs) - v["id"].nunique()),
                 labelled={h: int((segs.how == h).sum()) for h in ("vote", "vote_2nd", "bridge", "conflict", "none")},
                 rows_identified=int(v.new_id.lt(OTHER_ID_BASE).to_numpy()[keep].sum()), ids_out=int(out[1].nunique()))
    scores = {int(r.new_id): {j: float(getattr(r, f"s_{t}{j}")) for t, j in ids if t == r.identity[:1]}
              for r in segs[segs.identity != ""].drop_duplicates("new_id").itertuples()}
    # unnamed pieces without readable evidence: kept in OUT (overlays, debugging), left out of the submission
    stats["no_export_ids"] = sorted(int(i) for i in segs.new_id[(segs.identity == "")
                                                                & (segs.n_eligible < p.export_min_eligible)].unique())
    return out, segs, scores, stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("jnr", type=Path)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--roster", type=Path, help="default: Final challenge roster, then Training")
    ap.add_argument("--format", default="auto", choices=["auto", "ucmc", "mot"])
    for k, d in asdict(Params()).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(d), default=d)
    ap.add_argument("--submission", type=Path, help="ucmc only: also write a submission via identity.make_rows")
    ap.add_argument("--n-frames", type=int)
    ap.add_argument("--smooth-sigma", type=float, default=4.0,
                    help="submission: depth smoothing in frames (tracker/smooth.py), 0 = off")
    ap.add_argument("--score", action="store_true", help="training videos only: grade the submission against GT")
    a = ap.parse_args(argv)
    teams = find_roster(a.video, a.roster)
    p = Params(**{k: getattr(a, k) for k in asdict(Params())})
    out, segs, scores, stats = run(a.tracks, a.jnr, teams, p, a.format)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, header=False, index=False)
    stem = a.out.with_suffix("")
    segs.drop(columns=[c for c in segs if c.startswith(("x0", "y0", "x1", "y1"))]).to_csv(
        f"{stem}.segments.csv", index=False, float_format="%.4g")
    Path(f"{stem}.scores.json").write_text(json.dumps({"teams": teams, "scores": scores, "stats": stats}, indent=1))
    print(json.dumps(stats))
    if a.submission:
        if stats["format"] != "ucmc":
            raise SystemExit("--submission needs ucmc tracks (court positions)")
        if not a.n_frames:
            raise SystemExit("--submission needs --n-frames")
        from tracker.identity import make_rows, read_tracks as read_ucmc, score
        tr = read_ucmc(a.out)
        tr = tr[~tr.id.isin(stats["no_export_ids"])]
        rows = make_rows(tr, teams, a.n_frames, jersey_scores=scores)
        if a.smooth_sigma > 0:
            from tracker.smooth import registration_for, smooth_depth
            rows = smooth_depth(rows, registration_for(a.video), a.smooth_sigma)
        io.write_submission(a.submission, rows)
        print(f"{a.submission}: {len(rows)} rows")
        if a.score:
            r = score(rows, a.video, a.n_frames)
            print("real:", {k: round(float(x), 4) for k, x in r["real"].items() if np.ndim(x) == 0})


if __name__ == "__main__":
    main()
