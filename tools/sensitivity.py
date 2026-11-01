#!/usr/bin/env python
"""Score sensitivity of TI-HOTA to typical errors, using training GT (perturbed) as predictions."""
from __future__ import annotations

import numpy as np

from trackid import io
from trackid.metric import ti_hota_fast

VIDEOS = [f"IMG_{i:04d}" for i in range(104, 112)]
BLOCK = 30  # frames per corrupted segment (errors are temporally correlated in practice)
PCTS = [5, 10, 25, 50]


class Game:
    def __init__(self, video: str):
        df = io.load_court_gt(video).sort_values(["frame", "team", "jersey"], kind="stable")
        self.n = 6
        # a few GT frames have 5 or 7 rows: perturb only 6-row frames, pass the rest through verbatim
        ok = df.groupby("frame").frame.transform("size") == self.n
        self.other = io.gt_rows(df[~ok])
        df = df[ok]
        self.frames = df.frame.to_numpy().reshape(-1, self.n)[:, 0]
        self.X = df[["x", "y"]].to_numpy().reshape(-1, self.n, 2)
        self.A = (df.team + df.jersey).to_numpy().reshape(-1, self.n).astype(object)
        self.T = len(self.frames)
        self.rng = np.random.default_rng(0)

    def rows(self, X=None, A=None, keep=None, extra=()):
        X = self.X if X is None else X
        A = self.A if A is None else A
        keep = np.ones(A.shape, bool) if keep is None else keep
        f = np.broadcast_to(self.frames[:, None], A.shape)[keep]
        out = list(zip(f.tolist(), X[keep, 0].tolist(), X[keep, 1].tolist(), A[keep].tolist()))
        return out + self.other + list(extra)

    def block_mask(self, pct):
        m = np.zeros(self.T, bool)
        nb = -(-self.T // BLOCK)
        for b in self.rng.permutation(nb)[: round(nb * pct / 100)]:
            m[b * BLOCK:(b + 1) * BLOCK] = True
        return m


def scenarios(g: Game):
    r = g.rng
    for s in (5, 10, 20, 30, 50, 75):
        yield f"gaussian noise sigma={s}cm", g.rows(X=g.X + r.normal(0, s, g.X.shape))
    for b in (10, 20, 30, 50):
        yield f"constant bias {b}cm (x)", g.rows(X=g.X + [b, 0])
    for p in PCTS:
        m = g.block_mask(p)
        A = g.A.copy()
        s = A[m].astype(str)
        A[m] = np.char.add(np.where(np.char.startswith(s, "A"), "B", "A"), np.char.lstrip(s, "AB"))
        yield f"team labels swapped in {p}% frames", g.rows(A=A)
    first_team = [k for k in range(g.n) if str(g.A[0, k]).startswith("A")][:2]
    for p in PCTS:
        m = g.block_mask(p)
        A = g.A.copy()
        i, j = first_team
        A[m, i], A[m, j] = g.A[m, j], g.A[m, i]
        yield f"2 same-team ids swapped in {p}% frames", g.rows(A=A)
    k = 0  # dropped player
    for p in PCTS:
        m = g.block_mask(p)
        yield f"player dropped in {p}% frames", g.rows(keep=~(m[:, None] & (np.arange(g.n) == k)))
        X = g.X.copy()  # stale: hold last position seen before each dropped block
        idx = np.where(~m, np.arange(g.T), -1)
        idx = np.maximum.accumulate(idx)
        idx[idx < 0] = np.argmax(~m) if (~m).any() else 0
        X[:, k] = g.X[idx, k]
        yield f"player stale (held) in {p}% frames", g.rows(X=X)
    x = r.uniform(0, io.COURT_W, g.T)
    y = r.uniform(0, io.COURT_H, g.T)
    yield "7th spurious row/frame (unknown id)", g.rows(extra=zip(g.frames.tolist(), x, y, ["X0"] * g.T))
    d = np.array([g.X[t, 0] + [300, 0] for t in range(g.T)])
    yield "7th row/frame = duplicate id 300cm away", g.rows(extra=zip(g.frames.tolist(), d[:, 0], d[:, 1], g.A[:, 0].tolist()))


def main() -> None:
    games = [Game(v) for v in VIDEOS]
    res: dict[str, list] = {}
    base = []
    for g in games:
        gt = g.rows()
        base.append(ti_hota_fast(gt, gt)["TI-HOTA"])
        for name, pred in scenarios(g):
            r = ti_hota_fast(gt, pred)
            res.setdefault(name, []).append((r["TI-HOTA"], r["DetA"], r["AssA"]))
    print(f"Games: {len(games)} ({', '.join(VIDEOS)}); frames/game: {[g.T for g in games]}")
    print(f"perfect prediction TI-HOTA = {np.mean(base):.4f}\n")
    lines = ["| scenario | TI-HOTA | DetA | AssA |", "|---|---|---|---|"]
    for name, v in res.items():
        m = np.mean(v, 0)
        lines.append(f"| {name} | {m[0]:.4f} | {m[1]:.4f} | {m[2]:.4f} |")
    table = "\n".join(lines)
    print(table)
    from pathlib import Path
    Path(__file__).with_name("sensitivity_results.md").write_text(
        f"# TI-HOTA sensitivity (training GT as predictions)\n\nMean over {len(games)} games, tau=100cm, "
        f"fast metric. Corruptions applied in {BLOCK}-frame blocks, seed 0.\n\n{table}\n")


if __name__ == "__main__":
    main()
