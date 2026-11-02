"""Weak-label rules and the coverage table (pure functions)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import WeakCfg
from src.core.constants import NUMBERS

LABELS = {1: "eligible", 0: "ineligible", -1: "ambiguous", -2: "uncovered", -3: "overlap"}
WEAK_COLUMNS = ["crop_id", "video", "number_idx", "bib", "facing", "label", "p_true", "p_max", "pred_idx",
                "fold", "fold_saw_bib"]


def covered_numbers(train_df: pd.DataFrame) -> set[int]:
    return {int(i) for i in train_df.number_idx.unique()}


def assemble_labels(df: pd.DataFrame, probs: np.ndarray, covered: set[int], train_bibs: set[str],
                    weak: WeakCfg, max_iou: float | None, fold: str = "") -> pd.DataFrame:
    """Label held-out crops from a model that never saw their game. Overlap (-3) wins over uncovered (-2)."""
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (len(df), len(NUMBERS)):
        raise ValueError(f"probs shape {probs.shape} does not match {len(df)} rows x {len(NUMBERS)} numbers")
    y = df.number_idx.to_numpy(dtype=int)
    pred = probs.argmax(1)
    p_true = probs[np.arange(len(df)), y]
    cov = np.isin(y, sorted(covered))
    label = np.full(len(df), -1)
    label[cov & (pred == y) & (p_true >= weak.tau_pos)] = 1
    label[cov & (p_true < weak.tau_neg)] = 0
    label[~cov] = -2
    if max_iou is not None:
        label[(df.max_iou.to_numpy() > max_iou)] = -3
    return pd.DataFrame({"crop_id": df.crop_id.to_numpy(), "video": df.video.to_numpy(), "number_idx": y,
                         "bib": df.bib.to_numpy(), "facing": df.facing.to_numpy(), "label": label,
                         "p_true": p_true, "p_max": probs.max(1), "pred_idx": pred, "fold": fold,
                         "fold_saw_bib": df.bib.isin(train_bibs).to_numpy()}, columns=WEAK_COLUMNS)


def eligibility_labels(weak_df: pd.DataFrame) -> pd.DataFrame:
    return weak_df.loc[weak_df.label.isin([0, 1]), ["crop_id", "label"]].reset_index(drop=True)


def coverage_markdown(weak_df: pd.DataFrame) -> str:
    """Game-by-number table of label percentages, with a large warning when any number is uncovered."""
    lines = ["# Weak-label coverage", ""]
    unc = weak_df[weak_df.label == -2].groupby(["video", "number_idx"]).size()
    if len(unc):
        lines += ["**WARNING: some numbers were never seen by the fold model, so their crops are unlabelled (-2).**",
                  ""]
        lines += [f"- {v}: number {NUMBERS[i]} ({n} crops)" for (v, i), n in unc.items()] + [""]
    lines += ["Percent of crops per label (1 eligible, 0 ineligible, -1 ambiguous, -2 uncovered, -3 overlap).", "",
              "| game | number | bib | n | 1 | 0 | -1 | -2 | -3 | fold saw bib |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (v, i, b), g in weak_df.groupby(["video", "number_idx", "bib"]):
        pct = [f"{100 * (g.label == k).mean():.0f}" for k in (1, 0, -1, -2, -3)]
        lines.append(f"| {v} | {NUMBERS[i]} | {b} | {len(g)} | " + " | ".join(pct)
                     + f" | {'yes' if g.fold_saw_bib.all() else 'no' if not g.fold_saw_bib.any() else 'part'} |")
    return "\n".join(lines) + "\n"
