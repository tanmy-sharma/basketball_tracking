"""Raw-RGB crop store: crops.bin + manifest.csv per video, memmapped lazily (safe across DataLoader workers)."""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import CropCfg, crop_hash
from src.core.paths import outputs_dir
from src.dataio.schema import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, validate_manifest


def crops_root(crops: CropCfg) -> Path:
    return outputs_dir() / "crops" / crop_hash(crops)


class CropWriter:
    """Appends HxWx3 uint8 RGB crops to <video_dir>/crops.bin and returns each crop's byte offset."""

    def __init__(self, video_dir: Path) -> None:
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.video_dir / "crops.bin", "wb")
        self._pos = 0

    def add(self, rgb: np.ndarray) -> int:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected HxWx3 uint8, got {rgb.dtype} {rgb.shape}")
        off = self._pos
        self._f.write(np.ascontiguousarray(rgb).tobytes())
        self._pos += rgb.size
        return off

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> CropWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CropStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._maps: dict[str, np.memmap] = {}

    def __getstate__(self) -> dict:
        return {"root": self.root, "_maps": {}}

    def read(self, video: str, offset: int, w: int, h: int) -> np.ndarray:
        mm = self._maps.get(video)
        if mm is None:
            mm = self._maps[video] = np.memmap(self.root / video / "crops.bin", dtype=np.uint8, mode="r")
        n = w * h * 3
        if offset < 0 or offset + n > len(mm):
            raise IndexError(f"crop at offset {offset} ({w}x{h}) outside {video}/crops.bin")
        return np.array(mm[offset:offset + n]).reshape(h, w, 3)


def mark_done(video_dir: Path, **meta) -> None:
    """Write done.json atomically; call after crops.bin, manifest.csv and keypoints are final."""
    tmp = Path(video_dir) / "done.json.tmp"
    tmp.write_text(json.dumps(meta, default=str))
    os.replace(tmp, Path(video_dir) / "done.json")


def save_manifest(video_dir: Path, df: pd.DataFrame) -> None:
    df = validate_manifest(df)
    tmp = Path(video_dir) / "manifest.csv.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, Path(video_dir) / "manifest.csv")


def load_manifest(root: Path, videos: Iterable[str]) -> pd.DataFrame:
    frames = []
    num = {c: [""] for c in INT_COLUMNS + FLOAT_COLUMNS}
    for v in videos:
        d = Path(root) / v
        if not (d / "done.json").exists():
            raise FileNotFoundError(f"crops for {v} not finished (no done.json in {d})")
        frames.append(pd.read_csv(d / "manifest.csv", dtype={c: str for c in STR_COLUMNS},
                                  keep_default_na=False, na_values=num))
    if not frames:
        raise ValueError("no videos given")
    return validate_manifest(pd.concat(frames, ignore_index=True))
