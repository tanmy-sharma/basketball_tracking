"""Contiguous-block validation split with a guard band.

The val block is [val_start_frac, val_end_frac) of each video's max sampled frame (inclusive at 1.0).
Frames within guard_frames of an interior block edge are 'guard' on both sides and used by neither.
Val shares players, bibs and lighting with train: it is for early stopping only, never a generalisation score.
"""
from __future__ import annotations

import pandas as pd

from src.core.config import SplitCfg


def assign_split(df: pd.DataFrame, split: SplitCfg) -> pd.Series:
    out = pd.Series("train", index=df.index, dtype=object)
    for _, g in df.groupby("video"):
        fmax = g.frame.max()
        lo, hi = split.val_start_frac * fmax, split.val_end_frac * fmax
        f = g.frame
        in_val = (f >= lo) & ((f < hi) | (split.val_end_frac >= 1.0))
        out[g.index[in_val]] = "val"
        gd = split.guard_frames
        if gd > 0:
            guard = pd.Series(False, index=g.index)
            if split.val_start_frac > 0:
                guard |= (f >= lo - gd) & (f < lo + gd)
            if split.val_end_frac < 1.0:
                guard |= (f >= hi - gd) & (f < hi + gd)
            out[g.index[guard]] = "guard"
    return out
