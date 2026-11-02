"""Crop dataset over a CropStore. Subclasses override `label` to change how targets are produced."""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.core.constants import BIB_OF_ROSTER, NUMBER_TO_IDX
from src.dataio.store import CropStore

_BIB_ALLOWED = {bib: np.array([n in roster for n in NUMBER_TO_IDX]) for roster, bib in BIB_OF_ROSTER.items()}


def roster_allowed(bibs: pd.Series | list[str]) -> np.ndarray:
    """(N, 9) bool mask of the numbers on each row's team; unknown bibs allow everything."""
    return np.stack([_BIB_ALLOWED.get(b, np.ones(len(NUMBER_TO_IDX), bool)) for b in bibs])


def worker_init(_: int) -> None:
    cv2.setNumThreads(0)


class CropDataset(Dataset):
    """Item i is (transform(rgb, rng), self.label(i)).

    `transform` is called as transform(rgb_uint8_HxWx3, rng: np.random.Generator). The memmap is opened
    lazily per worker and the rng is created per process, so DataLoader workers do not share state.
    With label_col=None, labels are -1 (prediction).
    """

    def __init__(self, df: pd.DataFrame, store: CropStore, transform: Callable[..., Any],
                 label_col: str | None = "number_idx", seed: int = 0) -> None:
        self.df = df.reset_index(drop=True)
        self.store, self.transform, self.label_col, self.seed = store, transform, label_col, seed
        self.labels = (self.df[label_col].to_numpy(np.int64) if label_col else np.full(len(self.df), -1, np.int64))
        self.allowed = roster_allowed(self.df.bib) if "bib" in self.df else None
        self._video = self.df.video.to_numpy()
        self._geo = self.df[["offset", "crop_w", "crop_h"]].to_numpy(np.int64)
        self._rng: tuple[int, np.random.Generator] | None = None

    def __len__(self) -> int:
        return len(self.df)

    @property
    def rng(self) -> np.random.Generator:
        pid = os.getpid()
        if self._rng is None or self._rng[0] != pid:
            self._rng = (pid, np.random.default_rng([self.seed, torch.initial_seed() % (2 ** 32), pid]))
        return self._rng[1]

    def label(self, i: int) -> Any:
        return int(self.labels[i])

    def image(self, i: int) -> np.ndarray:
        off, w, h = self._geo[i]
        return self.store.read(str(self._video[i]), int(off), int(w), int(h))

    def __getitem__(self, i: int) -> tuple[Any, Any]:
        return self.transform(self.image(i), self.rng), self.label(i)


def balanced_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    """Inverse class frequency, same number of draws as rows."""
    counts = np.bincount(labels)
    w = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(labels), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))
