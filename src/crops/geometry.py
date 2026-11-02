"""Torso box from COCO keypoints, facing direction and box overlap."""
from __future__ import annotations

import numpy as np

L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
TORSO = (L_SHO, R_SHO, L_HIP, R_HIP)
MIN_KP_CONF = 0.3
SIDE_GAP = 0.25


def torso_box(kps: np.ndarray, pad_frac: float, min_aspect: float, min_side: int, img_w: int, img_h: int
              ) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) pixel box around shoulders and hips, or None if degenerate.

    Pads pad_frac of width and height per side, widens to width >= min_aspect * height (side views), clamps.
    """
    pts = np.asarray(kps, np.float64)[list(TORSO), :2]
    if not np.isfinite(pts).all():
        return None
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - pad_frac * w, x1 + pad_frac * w, y0 - pad_frac * h, y1 + pad_frac * h
    need = min_aspect * (y1 - y0)
    if x1 - x0 < need:
        cx = (x0 + x1) / 2
        x0, x1 = cx - need / 2, cx + need / 2
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(img_w, int(np.ceil(x1))), min(img_h, int(np.ceil(y1)))
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None
    return x0, y0, x1, y1


def facing(kps: np.ndarray) -> str:
    """front | back | side | unknown. Person's left shoulder at image right means front."""
    k = np.asarray(kps, np.float64)
    if k.shape[1] < 3 or (k[list(TORSO), 2] < MIN_KP_CONF).any():
        return "unknown"
    torso_h = abs(k[[L_HIP, R_HIP], 1].mean() - k[[L_SHO, R_SHO], 1].mean())
    if torso_h <= 0:
        return "unknown"
    gap = k[L_SHO, 0] - k[R_SHO, 0]
    if abs(gap) < SIDE_GAP * torso_h:
        return "side"
    return "front" if gap > 0 else "back"


def max_iou_per_box(boxes_xywh: np.ndarray) -> np.ndarray:
    """For each box, the highest IoU with any other box in the array (0 for a lone box)."""
    b = np.asarray(boxes_xywh, np.float64).reshape(-1, 4)
    n = len(b)
    out = np.zeros(n)
    if n < 2:
        return out
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(x1[:, None], x1) - np.maximum(x0[:, None], x0), 0, None)
    ih = np.clip(np.minimum(y1[:, None], y1) - np.maximum(y0[:, None], y0), 0, None)
    inter = iw * ih
    area = b[:, 2] * b[:, 3]
    union = area[:, None] + area - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(iou, 0)
    return iou.max(1)
