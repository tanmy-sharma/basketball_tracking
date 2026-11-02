"""Greyscale 1-channel input at input_size. Mirroring is never applied; rotations are small (shared augmenter)."""
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from src.core.config import AugCfg, RecognizerCfg

PAD = (128, 128, 128)


def to_input(rgb: np.ndarray, cfg: RecognizerCfg) -> np.ndarray:
    """uint8 RGB crop -> float32 (1, S, S), normalised with (x / 255 - 0.5) / 0.5."""
    from src.classifier.transforms import letterbox
    s = cfg.input_size
    img = letterbox(rgb, s, PAD) if cfg.resize == "letterbox" else cv2.resize(rgb, (s, s), interpolation=cv2.INTER_CUBIC)
    grey = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
    return ((grey.astype(np.float32) / 255.0 - 0.5) / 0.5)[None]


def eval_transform(cfg: RecognizerCfg) -> Callable[[np.ndarray, np.random.Generator | None], np.ndarray]:
    return lambda rgb, rng=None: to_input(rgb, cfg)


def train_transform(cfg: RecognizerCfg, aug: AugCfg) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    from src.classifier.transforms import augment_native
    return lambda rgb, rng: to_input(augment_native(rgb, aug, rng), cfg)
