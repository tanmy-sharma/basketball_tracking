"""Run a trained classifier (best.pt) over manifest rows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.classifier.dataset import CropDataset
from src.classifier.model import build_model
from src.classifier.transforms import build_transform
from src.core.checkpoint import load_checkpoint
from src.core.config import Config
from src.core.loop import pick_device
from src.dataio.store import CropStore


@torch.no_grad()
def predict_crops(best_pt: Path, df: pd.DataFrame, store: CropStore, device: str,
                  batch_size: int = 256) -> np.ndarray:
    """Softmax probabilities (N, C) in df row order. Builds the model from the checkpoint's own config."""
    ckpt = load_checkpoint(Path(best_pt))
    cfg = Config.model_validate(ckpt["config"])
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    dev = pick_device(device)
    model.to(dev).eval()
    head = ckpt["head"]
    ds = CropDataset(df, store, build_transform(cfg.model.input_size, None, False), label_col=None)
    out = [torch.softmax(model(x.to(dev), head).float(), 1).cpu().numpy()
           for x, _ in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)]
    return np.concatenate(out) if out else np.zeros((0, len(ckpt["classes"])), np.float32)
