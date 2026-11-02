"""Crop dataset with digit-token targets and the team roster mask of each crop."""
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.core.constants import BIB_OF_ROSTER, NUMBER_TO_IDX, NUMBERS
from src.dataio.store import CropStore
from src.recognizer.tokens import encode_number

ROSTER_OF_BIB = {b: frozenset(r) for r, b in BIB_OF_ROSTER.items()}


def roster_mask(bib: str) -> np.ndarray:
    """(9,) bool over NUMBERS for a bib colour; all True for an unknown bib."""
    roster = ROSTER_OF_BIB.get(bib)
    return np.array([n in roster if roster else True for n in NUMBERS])


def worker_init(_: int) -> None:
    cv2.setNumThreads(0)


class DigitCropDataset(Dataset):
    """Item: (image (1,S,S) float, tokens (seqlen,) long, number_idx long, roster (9,) bool)."""

    def __init__(self, df: pd.DataFrame, store: CropStore, transform: Callable, seqlen: int = 4,
                 augment: bool = False, seed: int = 0) -> None:
        self.df = df.reset_index(drop=True)
        self.store, self.transform, self.augment, self.seed = store, transform, augment, seed
        self.targets = np.array([encode_number(j, seqlen) for j in self.df.jersey], dtype=np.int64)
        self.idx = np.array([NUMBER_TO_IDX[j] for j in self.df.jersey], dtype=np.int64)
        self.rosters = np.stack([roster_mask(b) for b in self.df.bib]) if len(self.df) else np.zeros((0, 9), bool)
        self._calls = 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        rgb = self.store.read(r.video, int(r.offset), int(r.crop_w), int(r.crop_h))
        rng = None
        if self.augment:
            self._calls += 1
            rng = np.random.default_rng([self.seed, torch.initial_seed() % 2**32, i, self._calls])
        x = self.transform(rgb, rng)
        return (torch.from_numpy(x), torch.from_numpy(self.targets[i]), torch.tensor(self.idx[i]),
                torch.from_numpy(self.rosters[i]))
