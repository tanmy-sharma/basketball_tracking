"""Row selection for training: every filter in one place, plus the test-game guard."""
from __future__ import annotations

import pandas as pd

from src.core.config import Config
from src.core.constants import CHALLENGE_VIDEOS
from src.core.paths import resolve_path
from src.dataio.split import assign_split


def training_rows(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Filtered rows with a `split` column (train|val); guard rows are dropped.

    Split blocks are computed from the unfiltered frame range so filters cannot shift the val block.
    """
    g, f = cfg.games, cfg.filters
    videos = set(df.video)
    bad = videos & (set(g.test) | CHALLENGE_VIDEOS)
    if bad:
        raise ValueError(f"refusing to train on test/challenge games: {sorted(bad)}")
    extra = videos - set(g.train)
    if extra:
        raise ValueError(f"rows from games not in games.train: {sorted(extra)}")

    split = assign_split(df, cfg.split)
    keep = (df.status == "ok") & (df.pose_min_conf >= f.min_pose_conf)
    if f.max_iou is not None:
        keep &= df.max_iou <= f.max_iou
    for rule in f.exclude:
        keep &= ~((df.bib == rule.bib) & df.jersey.isin(rule.numbers))
    if f.include_csv:
        ids = pd.read_csv(resolve_path(f.include_csv), dtype=str).crop_id
        keep &= df.crop_id.isin(set(ids))
    out = df[keep].copy()
    out["split"] = split[keep]
    if f.subsample_every > 1:
        rank = out.groupby("video").frame.rank(method="dense").astype(int) - 1
        out = out[rank % f.subsample_every == 0]
    out = out[out.split != "guard"].reset_index(drop=True)
    for s in ("train", "val"):
        if not (out.split == s).any():
            raise ValueError(f"no {s} rows after filtering")
    return out
