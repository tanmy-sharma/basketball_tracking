"""Crop augmentation and preprocessing. Digits are chirality-sensitive: no mirroring, rotation capped by AugCfg."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from src.core.config import AugCfg

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PAD_RGB = tuple(int(round(m * 255)) for m in MEAN)


def letterbox(img: np.ndarray, size: int, pad_value: int | tuple[int, int, int] = PAD_RGB) -> np.ndarray:
    """Fit into a size x size canvas keeping aspect ratio; works for HxW and HxWx3 uint8."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    out = out.reshape(nh, nw, *img.shape[2:])
    canvas = np.empty((size, size, *img.shape[2:]), img.dtype)
    canvas[...] = pad_value
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = out
    return canvas


def _subcrop(img: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    mx, my = int(w * frac), int(h * frac)
    x0, x1 = int(rng.integers(0, mx + 1)), w - int(rng.integers(0, mx + 1))
    y0, y1 = int(rng.integers(0, my + 1)), h - int(rng.integers(0, my + 1))
    return img[y0:y1, x0:x1]


def _rotate(img: np.ndarray, max_deg: float, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-max_deg, max_deg)), 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else img


def _color(img: np.ndarray, aug: AugCfg, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-aug.hue_deg, aug.hue_deg) / 2) % 180  # OpenCV hue is degrees / 2
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + rng.uniform(-aug.sat, aug.sat)), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out = out * (1 + rng.uniform(-aug.contrast, aug.contrast))
    out = out + 255 * rng.uniform(-aug.brightness, aug.brightness)
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_native(rgb: np.ndarray, aug: AugCfg, rng: np.random.Generator) -> np.ndarray:
    """Sub-crop jitter, small rotation, JPEG, blur and colour jitter at native size (uint8 RGB in and out)."""
    img = _subcrop(rgb, aug.subcrop_frac, rng)
    img = _rotate(img, aug.max_rotate_deg, rng)
    if rng.random() < aug.jpeg_p:
        img = _jpeg(img, int(rng.integers(aug.jpeg_q[0], aug.jpeg_q[1] + 1)))
    if rng.random() < aug.blur_p:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(.3, 1.2)))
    return _color(img, aug, rng)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


class CropTransform:
    """Callable (rgb uint8, rng) -> (3, size, size) float tensor. Train adds augmentation and random greyscale."""

    def __init__(self, size: int, aug: AugCfg | None = None) -> None:
        self.size, self.aug = size, aug

    def __call__(self, rgb: np.ndarray, rng: np.random.Generator | None = None) -> torch.Tensor:
        if self.aug is not None:
            rng = rng or np.random.default_rng()
            rgb = augment_native(rgb, self.aug, rng)
            rgb = letterbox(rgb, self.size, PAD_RGB)
            if rng.random() < self.aug.grey_p:
                g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                rgb = np.repeat(g[..., None], 3, axis=2)
        else:
            rgb = letterbox(rgb, self.size, PAD_RGB)
        return to_tensor(rgb)


def build_transform(size: int, aug: AugCfg | None, train: bool) -> CropTransform:
    return CropTransform(size, aug if train else None)
