"""Pure metric functions. `y` is the true number index, `probs` is (N, 9) in NUMBERS order, `allowed` is a bool (N, 9)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.core.constants import NUMBER_TO_IDX, NUMBERS

SWAP = {"6": "9", "9": "6", "10": "12"}


def masked_pred(probs: np.ndarray, allowed: np.ndarray | None) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    return (p if allowed is None else np.where(allowed, p, -np.inf)).argmax(1)


def per_crop_accuracy(y, probs) -> float:
    return float((np.asarray(probs).argmax(1) == np.asarray(y)).mean())


def within_team_accuracy(y, probs, allowed) -> float:
    return float((masked_pred(probs, allowed) == np.asarray(y)).mean())


def confusion(y, pred, n: int = 9) -> np.ndarray:
    m = np.zeros((n, n), int)
    np.add.at(m, (np.asarray(y, int), np.asarray(pred, int)), 1)
    return m


def accuracy_by_bib(df: pd.DataFrame, probs, allowed, numbers=("6", "9", "10")) -> pd.DataFrame:
    """Per (bib, number): 9-way and within-team accuracy, and how often the number is read as its look-alike."""
    p9, pt = np.asarray(probs).argmax(1), masked_pred(probs, allowed)
    rows = []
    for (bib, num), g in df.reset_index(drop=True).groupby(["bib", "jersey"]):
        if num not in numbers:
            continue
        i, y = g.index.to_numpy(), NUMBER_TO_IDX[num]
        other = SWAP.get(num)
        rows.append({"bib": bib, "number": num, "n": len(g), "acc_9way": float((p9[i] == y).mean()),
                     "acc_team": float((pt[i] == y).mean()), "swap_to": other,
                     "swap_rate": float((p9[i] == NUMBER_TO_IDX[other]).mean()) if other else np.nan})
    return pd.DataFrame(rows, columns=["bib", "number", "n", "acc_9way", "acc_team", "swap_to", "swap_rate"])


def tracklet_vote(df: pd.DataFrame, probs, weights=None, window_s: float = np.inf, allowed=None,
                  fps: float = 29.97) -> pd.DataFrame:
    """Sum weighted probabilities over (video, track, window). A window with no weight abstains (pred_idx -1).

    `window_s=inf` is the whole track. `weights` (N,) are e.g. 0/1 eligibility.
    """
    d = df.reset_index(drop=True)
    p = np.asarray(probs, dtype=float)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=float)
    win = np.zeros(len(d), int) if not np.isfinite(window_s) else (d.frame.to_numpy() / fps // window_s).astype(int)
    rows = []
    for (v, t, k), g in d.assign(_w=win).groupby(["video", "track_id", "_w"]):
        i = g.index.to_numpy()
        tot = (p[i] * w[i, None]).sum(0)
        if allowed is not None:
            tot = np.where(allowed[i[0]], tot, -np.inf)
        abst = w[i].sum() <= 0
        pred = -1 if abst else int(tot.argmax())
        truth = int(g.number_idx.iloc[0])
        rows.append({"video": v, "track_id": t, "window": k, "n_crops": len(i), "n_voting": int((w[i] > 0).sum()),
                     "truth": truth, "pred_idx": pred, "abstained": bool(abst), "correct": pred == truth})
    return pd.DataFrame(rows, columns=["video", "track_id", "window", "n_crops", "n_voting", "truth", "pred_idx",
                                       "abstained", "correct"])


def summarize_votes(votes: pd.DataFrame) -> dict:
    voted = votes[~votes.abstained]
    return {"n_windows": len(votes), "abstain_rate": float(votes.abstained.mean()) if len(votes) else np.nan,
            "accuracy_voted": float(voted.correct.mean()) if len(voted) else np.nan,
            "accuracy_all": float(votes.correct.mean()) if len(votes) else np.nan}


def coverage_curve(y, probs, scores, thresholds, allowed=None) -> pd.DataFrame:
    """Keep crops with score >= threshold; report coverage and accuracy on the kept crops."""
    ok = masked_pred(probs, allowed) == np.asarray(y)
    s = np.asarray(scores, dtype=float)
    rows = []
    for t in thresholds:
        keep = s >= t
        rows.append({"threshold": t, "coverage": float(keep.mean()),
                     "accuracy": float(ok[keep].mean()) if keep.any() else np.nan})
    return pd.DataFrame(rows)


def auc(y, scores) -> float:
    """ROC AUC for binary y via rank sums (ties averaged); NaN if only one class."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(scores, dtype=float))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def accuracy_by_track(df: pd.DataFrame, pred) -> pd.DataFrame:
    d = df.reset_index(drop=True).assign(_ok=(np.asarray(pred) == df.number_idx.to_numpy()))
    g = d.groupby(["video", "track_id", "jersey"])._ok
    return pd.DataFrame({"n": g.size(), "accuracy": g.mean()}).reset_index()


def accuracy_by(df: pd.DataFrame, pred, col: str) -> pd.DataFrame:
    d = df.reset_index(drop=True).assign(_ok=(np.asarray(pred) == df.number_idx.to_numpy()))
    g = d.groupby(col)._ok
    return pd.DataFrame({"n": g.size(), "accuracy": g.mean()}).reset_index()


def free_read_stats(df: pd.DataFrame, reads: pd.DataFrame, allowed: np.ndarray) -> dict:
    """Free-decoding rates over crops present in `reads` (columns crop_id, text).

    exact: text equals the true number. off_roster: non-empty text that is not one of the team's 3 numbers.
    """
    d = df.reset_index(drop=True)
    txt = d.crop_id.map(reads.drop_duplicates("crop_id").set_index("crop_id").text.astype(str))
    have = txt.notna().to_numpy()
    if not have.any():
        return {"n": 0}
    t = txt[have].str.strip()
    roster = np.array([{NUMBERS[j] for j in np.flatnonzero(a)} for a in allowed], dtype=object)[have]
    nonempty = (t != "").to_numpy()
    off = np.array([x != "" and x not in r for x, r in zip(t, roster)])
    return {"n": int(have.sum()), "exact_rate": float((t == d.jersey[have]).mean()),
            "empty_rate": float((~nonempty).mean()), "off_roster_rate": float(off.mean()),
            "off_roster_of_nonempty": float(off.sum() / nonempty.sum()) if nonempty.any() else float("nan")}
