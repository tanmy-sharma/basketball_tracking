from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, full: bool = False) -> dict:
    """best.pt loads with weights_only=True; `full=True` (last.pt: optimizer, rng) needs it off."""
    return torch.load(path, map_location="cpu", weights_only=not full)
