#!/usr/bin/env python
"""Turn tracker output into a TrackID3x3 submission: (frame, x_cm, y_cm, team, jersey) rows.

  python src/tracker/identity.py TRACKS.txt --video IMG_0106 --roster data/Training/games_roster.csv \
      --n-frames 3000 --out OUT.txt [--score]

TRACKS.txt is what run_ucmc.py writes (COLUMNS, plus TEAM_COLUMNS when run with --labels). Steps:
 1. split tracks where the per-box team calls flip persistently (binary segmentation on the calls),
 2. one team per segment (confidence-weighted vote; belief sign if no calls; dropped if nothing, or too short),
 3. one jersey per segment: same-team segments sharing a frame get different numbers; longest first, preferring
    jersey_scores (a future jersey reader) and then spatial-temporal continuity with the segment that last held the number,
 4. rows that would repeat an identity in a frame are dropped (so <= 3 per team per frame),
 5. frames left empty are filled from the nearest known positions of each identity.
Nothing here reads ground truth; --score only uses it to grade a training video afterwards.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from trackid import io  # noqa: E402

BASE_COLUMNS = ["frame", "id", "x", "y", "raw_x", "raw_y", "kind", "conf", "bx", "by", "bw", "bh"]
TEAM_COLUMNS = ["team", "belief", "det_row"]
SEG_STRIDE = 1_000_000  # split piece k of track t gets id t + k * SEG_STRIDE


def read_tracks(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    names = BASE_COLUMNS + TEAM_COLUMNS if df.shape[1] >= 15 else BASE_COLUMNS
    df = df.iloc[:, : len(names)]
    df.columns = names
    if "team" not in df:
        df["team"], df["belief"], df["det_row"] = 0, 0.0, -1
    return df


# ----------------------------------------------------------------------------- 1. splitting
def _change_points(c: np.ndarray, min_calls: int, margin: float) -> list[int]:
    """Binary segmentation of a +-1 call sequence. Returns indices (into c) where a new segment starts."""
    n = len(c)
    if n < 2 * min_calls:
        return []
    S = np.cumsum(c)
    best, best_t = 0.0, -1
    for s in (1, -1):  # left majority sign s, right majority -s
        gain = s * (2 * S - S[-1])  # sum_left*s - sum_right*s, at split after index t
        gain[: min_calls - 1] = -np.inf
        gain[n - min_calls:] = -np.inf
        t = int(np.argmax(gain))
        # compare to keeping one segment: |S_n|; require both halves to really disagree
        left, right = S[t], S[-1] - S[t]
        if gain[t] - abs(S[-1]) > best and left * s > 0 and right * s < 0 and min(abs(left), abs(right)) >= margin:
            best, best_t = gain[t] - abs(S[-1]), t
    if best_t < 0 or best < margin:
        return []
    t = best_t + 1
    return (_change_points(c[:t], min_calls, margin)
            + [t]
            + [t + k for k in _change_points(c[t:], min_calls, margin)])


def split_tracks(df: pd.DataFrame, k: int = 30, margin: float = 0.4) -> pd.DataFrame:
    """Adds column seg (segment id). A track splits where >= k consecutive calls flip team persistently.
    margin: minimum |vote sum| on each side as a fraction of k."""
    df = df.sort_values(["id", "frame"], kind="stable").reset_index(drop=True)
    seg = df["id"].to_numpy().copy()
    if (df.team != 0).any():
        for tid, idx in df.groupby("id").indices.items():
            calls = df.team.to_numpy()[idx]
            has = np.flatnonzero(calls != 0)
            cps = _change_points(calls[has].astype(float), k, margin * k)
            if not cps:
                continue
            starts = [0] + [has[i] for i in cps]  # row offsets within the track
            for piece, st in enumerate(starts[1:], 1):
                seg[idx[st:]] = tid + piece * SEG_STRIDE
    df["seg"] = seg
    return df


# ----------------------------------------------------------------------------- 2. team per segment
def segment_table(df: pd.DataFrame, min_len: int, has_calls: bool) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("seg"):
        if len(g) < min_len:
            continue
        w = np.where(g.conf.to_numpy() > 0, g.conf.to_numpy(), 0.0)
        vote = float((g.team.to_numpy() * w).sum())
        if vote != 0:
            team = "A" if vote > 0 else "B"
        elif has_calls and g.belief.mean() != 0:
            team = "A" if g.belief.mean() > 0 else "B"
        else:
            team = None
        rows.append(dict(seg=sid, team=team, start=int(g.frame.min()), end=int(g.frame.max()), n=len(g),
                         conf=float(g.conf[g.conf > 0].sum()), x0=g.x.iloc[0], y0=g.y.iloc[0],
                         x1=g.x.iloc[-1], y1=g.y.iloc[-1]))
    return pd.DataFrame(rows)


def balance_unknown(segs: pd.DataFrame, per_team: int = 3) -> pd.DataFrame:
    """No colour information at all: spread segments over the two teams so neither exceeds 3 at once."""
    segs = segs.copy()
    load = {"A": {}, "B": {}}
    for i in segs.sort_values("n", ascending=False).index:
        r = segs.loc[i]
        fr = range(r.start, r.end + 1)

        def peak(t):
            return max((load[t].get(f, 0) for f in fr), default=0)
        t = min("AB", key=peak)
        segs.loc[i, "team"] = t
        for f in fr:
            load[t][f] = load[t].get(f, 0) + 1
    return segs


# ----------------------------------------------------------------------------- 3. jerseys
def assign_jerseys(df: pd.DataFrame, segs: pd.DataFrame, teams: dict[str, list[str]],
                   jersey_scores: dict | None = None, tau_t: float = 150.0, tau_d: float = 300.0,
                   w_cont: float = 1.0) -> dict[int, str]:
    """seg -> jersey. Longest segments first; each takes the jersey with the best utility
    (score + continuity - overlap penalty). Overlap with an already placed same-team segment costs that many frames
    (those rows get dropped later), so a conflict-free number always wins when one exists."""
    out: dict[int, str] = {}
    frames_of = {sid: g.frame.to_numpy() for sid, g in df.groupby("seg")}
    for team, jerseys in teams.items():
        ts = segs[segs.team == team].sort_values("n", ascending=False)
        placed: dict[str, list] = {j: [] for j in jerseys}  # jersey -> [(seg, frames, row)]
        for _, r in ts.iterrows():
            fr = frames_of[r.seg]
            best, best_u = None, -np.inf
            for j in jerseys:
                overlap = sum(len(np.intersect1d(fr, f2, assume_unique=True)) for _, f2, _ in placed[j])
                u = -1000.0 * overlap / len(fr)
                if jersey_scores:
                    sc = jersey_scores.get(r.seg) or jersey_scores.get(r.seg % SEG_STRIDE) or {}
                    u += 10.0 * sc.get(j, 0.0)
                # continuity: nearest earlier / later placed neighbours on this number
                for _, f2, r2 in placed[j]:
                    if r2.end < r.start:
                        gap, d = r.start - r2.end, np.hypot(r.x0 - r2.x1, r.y0 - r2.y1)
                    elif r.end < r2.start:
                        gap, d = r2.start - r.end, np.hypot(r2.x0 - r.x1, r2.y0 - r.y1)
                    else:
                        continue
                    u += w_cont * np.exp(-gap / tau_t) * np.exp(-d / tau_d)
                if u > best_u:
                    best, best_u = j, u
            out[r.seg] = best
            placed[best].append((r.seg, fr, r))
    return out


# ----------------------------------------------------------------------------- 4/5. rows
def build_rows(df: pd.DataFrame, segs: pd.DataFrame, jersey: dict[int, str], n_frames: int, hold: int = 0, fallback=("A", "0")):
    team_of = dict(zip(segs.seg, segs.team))
    d = df[df.seg.isin(jersey)].copy()
    d["team_id"] = d.seg.map(team_of)
    d["jersey"] = d.seg.map(jersey)
    d["len"] = d.seg.map(dict(zip(segs.seg, segs.n)))
    d = d[(d.frame >= 1) & (d.frame <= n_frames)]
    # one row per identity per frame; keep the longest segment (then the most confident)
    d = d.sort_values(["frame", "team_id", "jersey", "len", "conf"], ascending=[True, True, True, False, False])
    d = d.drop_duplicates(["frame", "team_id", "jersey"])
    ident = {(t, j): g.sort_values("frame") for (t, j), g in d.groupby(["team_id", "jersey"])}

    rows = [(int(f), float(x), float(y), t, j) for f, x, y, t, j in zip(d.frame, d.x, d.y, d.team_id, d.jersey)]
    have = set(d.frame)
    if hold > 0 and ident:  # keep each identity alive for a few frames after it was last seen
        present = set(zip(d.frame, d.team_id, d.jersey))
        for (t, j), g in ident.items():
            fr, xs, ys = g.frame.to_numpy(), g.x.to_numpy(), g.y.to_numpy()
            for i in range(len(fr)):
                end = min(fr[i] + hold, fr[i + 1] - 1 if i + 1 < len(fr) else n_frames, n_frames)
                for f in range(fr[i] + 1, end + 1):
                    if (f, t, j) not in present:
                        rows.append((f, float(xs[i]), float(ys[i]), t, j))
        have = {r[0] for r in rows}
    missing = [f for f in range(1, n_frames + 1) if f not in have]
    if missing:
        if ident:
            m = np.asarray(missing)
            for (t, j), g in ident.items():
                fr = g.frame.to_numpy()
                # nearest known row in time (before preferred: carry forward)
                pos = np.searchsorted(fr, m) - 1
                before, after = np.clip(pos, 0, len(fr) - 1), np.clip(pos + 1, 0, len(fr) - 1)
                use = np.where((pos >= 0), before, after)
                for f, u in zip(m, use):
                    rows.append((int(f), float(g.x.iloc[u]), float(g.y.iloc[u]), t, j))
        else:  # nothing identified at all: a valid but empty-handed fallback
            rows += [(f, 752.5, 552.5, *fallback) for f in missing]
    rows.sort(key=lambda r: (r[0], r[3], r[4]))
    return rows


def make_rows(tracks: pd.DataFrame, teams: dict[str, list[str]], n_frames: int, k: int = 30, min_len: int = 15,
              jersey_scores: dict | None = None, hold: int = 0, drop_unknown: bool = True):
    fb = ("A", teams["A"][0])
    has_calls = bool((tracks.team != 0).any())
    tracks = tracks[tracks.frame <= n_frames]
    if tracks.empty:
        return build_rows(tracks.assign(seg=[]), pd.DataFrame(columns=["seg", "team", "n"]), {}, n_frames, fallback=fb)
    df = split_tracks(tracks[tracks.frame <= n_frames], k=k)
    segs = segment_table(df, min_len, has_calls)
    if len(segs):
        if not has_calls:
            segs = balance_unknown(segs)
        elif drop_unknown:
            segs = segs[segs.team.notna()]
        else:
            segs = balance_unknown(segs[segs.team.notna()])
    if not len(segs):
        return build_rows(df, segs.assign(team=[]), {}, n_frames, fallback=fb)
    jersey = assign_jerseys(df, segs, teams, jersey_scores)
    return build_rows(df, segs, jersey, n_frames, hold)


# ----------------------------------------------------------------------------- scoring (training videos)
def score(rows, video: str, n_frames: int, n_perm: int = 10, seed: int = 0) -> dict:
    from trackid.metric import ti_hota_fast
    gt = io.load_court_gt(video)
    gt = gt[gt.frame <= n_frames]
    g = io.gt_rows(gt)
    pred = [(f, x, y, t + j) for f, x, y, t, j in rows]
    res = {"real": ti_hota_fast(g, pred)}
    rng = np.random.default_rng(seed)
    teams = io.load_roster(io.TRAIN / "games_roster.csv")[video].teams
    vals = []
    for _ in range(n_perm):
        mp = {t: dict(zip(js, rng.permutation(js))) for t, js in teams.items()}
        vals.append(ti_hota_fast(g, [(f, x, y, t + mp[t][j]) for f, x, y, t, j in rows])["TI-HOTA"]
                    if "TI-HOTA" in res["real"] else np.nan)
    res["perm_mean"], res["perm_min"], res["perm_max"] = float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--video", required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=30, help="calls on each side needed to split a track")
    ap.add_argument("--min-len", type=int, default=15)
    ap.add_argument("--hold", type=int, default=0, help="frames to keep an identity alive after its last row")
    ap.add_argument("--score", action="store_true", help="training videos only: print TI-HOTA against GT")
    a = ap.parse_args()
    teams = io.load_roster(a.roster)[a.video].teams
    rows = make_rows(read_tracks(a.tracks), teams, a.n_frames, a.k, a.min_len, hold=a.hold)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    io.write_submission(a.out, rows)
    print(f"{a.out}: {len(rows)} rows, {len({r[0] for r in rows})}/{a.n_frames} frames")
    if a.score:
        r = score(rows, a.video, a.n_frames)
        print("real:", {k: round(float(v), 4) for k, v in r["real"].items() if np.ndim(v) == 0})
        print(f"random jersey permutation (team right): mean {r['perm_mean']:.4f} min {r['perm_min']:.4f} max {r['perm_max']:.4f}")


if __name__ == "__main__":
    main()
