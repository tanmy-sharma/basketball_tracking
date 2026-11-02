"""Pose estimation behind a small protocol, so extraction is testable with a fake."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch


class PoseEstimator(Protocol):
    def __call__(self, frame_rgb: np.ndarray, boxes_xywh: np.ndarray) -> np.ndarray:
        """(N, 17, 3): x, y in frame pixels and confidence, per COCO box."""
        ...


class ViTPose:
    """transformers VitPoseForPoseEstimation, one forward pass per frame. `model`/`processor` allow injection."""

    def __init__(self, model_id: str, revision: str | None, device: str, model: Any = None,
                 processor: Any = None) -> None:
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self.device = device
        self.proc = processor or AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = (model or VitPoseForPoseEstimation.from_pretrained(model_id, revision=revision))
        self.model.to(device).eval()

    @torch.inference_mode()
    def __call__(self, frame_rgb: np.ndarray, boxes_xywh: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes_xywh, np.float64).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, 17, 3), np.float32)
        lst = boxes.tolist()
        inp = self.proc(images=[frame_rgb], boxes=[lst], return_tensors="pt")
        out = self.model(pixel_values=inp["pixel_values"].to(self.device))
        res = self.proc.post_process_pose_estimation(out, boxes=[lst])[0]
        return np.stack([np.concatenate([r["keypoints"].float().cpu().numpy(),
                                         r["scores"].float().cpu().numpy()[:, None]], 1) for r in res]
                        ).astype(np.float32)
