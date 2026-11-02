"""Crop manifest columns and validation."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

CROP_COLUMNS = ["crop_id", "video", "frame", "track_id", "team", "jersey", "number_idx", "bib", "in_game",
                "box_x", "box_y", "box_w", "box_h", "max_iou", "pose_min_conf",
                "crop_x0", "crop_y0", "crop_x1", "crop_y1", "crop_w", "crop_h", "offset", "facing", "status"]
STR_COLUMNS = ["crop_id", "video", "team", "jersey", "bib", "facing", "status"]
INT_COLUMNS = ["frame", "track_id", "number_idx", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "crop_w", "crop_h",
               "offset"]
FLOAT_COLUMNS = ["box_x", "box_y", "box_w", "box_h", "max_iou", "pose_min_conf"]
FACING = ("front", "back", "side", "unknown")
STATUS = ("ok", "low_pose", "degenerate")
_ID = re.compile(r"(IMG_\d{4}):(\d{6}):(\d{2})")


def crop_id(video: str, frame: int, track_id: int) -> str:
    return f"{video}:{frame:06d}:{track_id:02d}"


def validate_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with columns in CROP_COLUMNS order and canonical dtypes, or raise ValueError."""
    missing = [c for c in CROP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    df = df[CROP_COLUMNS].copy()
    for c in STR_COLUMNS:
        df[c] = df[c].fillna("").astype(str)
    if df.in_game.dtype != bool:
        df["in_game"] = df.in_game.astype(str).str.lower().map({"true": True, "false": False})
        if df.in_game.isna().any():
            raise ValueError("in_game must be True/False")
        df["in_game"] = df.in_game.astype(bool)
    for c in INT_COLUMNS:
        df[c] = df[c].astype("int64")
    for c in FLOAT_COLUMNS:
        df[c] = df[c].astype("float64")
    for col, allowed in (("facing", FACING), ("status", STATUS)):
        bad = set(df[col]) - set(allowed)
        if bad:
            raise ValueError(f"bad {col} values: {sorted(bad)}")
    if df.crop_id.duplicated().any():
        raise ValueError("duplicate crop_id")
    want = [crop_id(v, f, t) for v, f, t in zip(df.video, df.frame, df.track_id)]
    if (df.crop_id.to_numpy() != np.array(want, dtype=object)).any():
        raise ValueError("crop_id does not match video/frame/track_id")
    return df

