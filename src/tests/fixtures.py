"""Synthetic fixtures shared by every package's tests. No competition data, GPU, downloads or models."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.core.config import Config, PhaseCfg
from src.core.constants import BIB_OF_ROSTER, NUMBER_TO_IDX
from src.core.paths import add_tools_to_path
from src.core.plugin import LoopPlugin
from src.dataio.schema import CROP_COLUMNS, crop_id
from src.dataio.store import CropStore, CropWriter, mark_done, save_manifest

IMG_W, IMG_H = 3840, 2160
# (team A jerseys, team B jerseys) per game, cycled over the requested games
ROSTERS = [(("6", "9", "11"), ("1", "3", "5")), (("6", "9", "10"), ("1", "3", "5")),
           (("4", "10", "12"), ("6", "9", "11"))]


def tiny_dino_overrides() -> dict:
    return dict(image_size=56, patch_size=14, hidden_size=48, num_hidden_layers=4, num_attention_heads=2,
                mlp_ratio=2)


def make_training_tree(root: Path, games: list[str], n_frames: int = 60, stray_track: bool = False,
                       duplicate_row: bool = False) -> Path:
    """Build a Training-like dir at `root` (the value of trackid.io.TRAIN) and return it.

    Per game: 6 tracks (ids 1..6, jerseys from ROSTERS), boxes in 4K, GT court positions consistent with
    the registration so `track_identities` resolves. `stray_track` adds a box with id 0 (not on the roster);
    `duplicate_row` repeats the first (frame, track) row.
    """
    root = Path(root)
    court = root / "full player annotations" / "court position"
    bbox = root / "full player annotations" / "Image-space bounding boxes"
    for d in (court, bbox, root / "vidoes"):
        d.mkdir(parents=True, exist_ok=True)
    # image -> court is a pure scale, so bottom-centre of a box maps to court = pt * (1505/W, 1105/H)
    pd.DataFrame({"point_id": ["p0", "p1", "p2", "p3"],
                  "image_x": [0, IMG_W, IMG_W, 0], "image_y": [0, 0, IMG_H, IMG_H],
                  "court_x": [0.0, 1505.0, 1505.0, 0.0], "court_y": [0.0, 0.0, 1105.0, 1105.0]}
                 ).to_csv(root / "court_registration.csv", index=False)
    rows = []
    for gi, g in enumerate(games):
        ta, tb = ROSTERS[gi % len(ROSTERS)]
        jerseys = list(ta) + list(tb)
        teams = ["A"] * 3 + ["B"] * 3
        rows.append({"video": g, "game_start_frame": 1,
                     "track_id: jersey": ", ".join(f"{i + 1}: {j}" for i, j in enumerate(jerseys)),
                     "offense_track_ids": "1, 2, 3", "defense_track_ids": "4, 5, 6",
                     "team: jersey": f"A: {', '.join(ta)}; B: {', '.join(tb)}"})
        boxes, gt = [], []
        for f in range(1, n_frames + 1):
            for k in range(6):
                x, y, w, h = 300 + 450 * k + 2 * f, 600 + 100 * k, 140, 400
                boxes.append((f, k + 1, x, y, w, h, 1.0, -1, -1))
                gt.append((f, (x + w / 2) * 1505 / IMG_W, (y + h) * 1105 / IMG_H, teams[k], jerseys[k]))
        if stray_track:
            boxes.append((1, 0, 50, 50, 100, 300, 0.9, -1, -1))
        if duplicate_row:
            boxes.append(boxes[0])
        with open(bbox / f"{g}.txt", "w", newline="") as fh:
            csv.writer(fh).writerows(boxes)
        with open(court / f"{g}.txt", "w", newline="") as fh:
            csv.writer(fh).writerows(gt)
    pd.DataFrame(rows).to_csv(root / "games_roster.csv", index=False)
    return root


def patch_trackid_paths(monkeypatch, root: Path) -> None:
    """Point tools/trackid/io at a synthetic tree from make_training_tree."""
    add_tools_to_path()
    from trackid import io
    root = Path(root)
    monkeypatch.setattr(io, "TRAIN", root)
    monkeypatch.setattr(io, "GT_BBOX", root / "full player annotations" / "Image-space bounding boxes")
    monkeypatch.setattr(io, "GT_COURT", root / "full player annotations" / "court position")


def draw_digit(number: str, w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Noisy crop with the number written on it, so a tiny net can learn the classes."""
    bg = rng.integers(40, 215, 3)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    fg = (255 - bg).tolist()
    scale = min(w / (16.0 * len(number)), h / 22.0)
    (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(img, number, (max(0, (w - tw) // 2), (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 2)
    noise = rng.normal(0, 8, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def make_crop_store(root: Path, video: str, n: int, numbers: list[str], rng: np.random.Generator,
                    frame_step: int = 5) -> pd.DataFrame:
    """Write a finished crop store for `video` under `root` (a crops_root-like dir) and return its manifest.

    Row i has number numbers[i % len(numbers)], track_id = that index + 1, frame = 1 + frame_step * (i // len).
    All rows status ok, pose_min_conf .9, max_iou 0. Read it with CropStore(root).
    """
    vdir = Path(root) / video
    nums = list(numbers)
    bib = BIB_OF_ROSTER.get(frozenset(nums), "green")
    recs = []
    with CropWriter(vdir) as wr:
        for i in range(n):
            j = i % len(nums)
            num, frame, tid = nums[j], 1 + frame_step * (i // len(nums)), j + 1
            w, h = int(rng.integers(21, 30)), int(rng.integers(33, 45))
            off = wr.add(draw_digit(num, w, h, rng))
            recs.append({"crop_id": crop_id(video, frame, tid), "video": video, "frame": frame, "track_id": tid,
                         "team": "A", "jersey": num, "number_idx": NUMBER_TO_IDX[num], "bib": bib,
                         "in_game": True, "box_x": 100.0, "box_y": 100.0, "box_w": 40.0, "box_h": 110.0,
                         "max_iou": 0.0, "pose_min_conf": 0.9, "crop_x0": 110, "crop_y0": 110,
                         "crop_x1": 110 + w, "crop_y1": 110 + h, "crop_w": w, "crop_h": h, "offset": off,
                         "facing": "front", "status": "ok"})
    df = pd.DataFrame(recs, columns=CROP_COLUMNS)
    save_manifest(vdir, df)
    mark_done(vdir, synthetic=True)
    return df


def make_video(path: Path, n_frames: int, w: int = 64, h: int = 36, fps: int = 30) -> Path:
    """Lossless video whose frame i is a flat colour (R=i%256, G=i//256, B=77); see frame_index()."""
    frames = np.empty((n_frames, h, w, 3), np.uint8)
    for i in range(n_frames):
        frames[i] = (i % 256, i // 256, 77)
    path = Path(path)
    if path.suffix.lower() not in (".mkv", ".mov"):
        path = path.with_suffix(".mkv")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "ffv1", "-pix_fmt", "bgr0",
                    str(path)],
                   input=frames.tobytes(), check=True, capture_output=True)
    return path


def frame_index(rgb: np.ndarray) -> int:
    """Inverse of make_video's colour coding (zero-based frame index)."""
    r, g, _ = rgb[rgb.shape[0] // 2, rgb.shape[1] // 2]
    return int(r) + 256 * int(g)


class ToyPlugin(LoopPlugin):
    """Linear classifier on random data; `crash_at` (set on the class) raises once in evaluate at that epoch.

    Usable as plugin "src.tests.fixtures:ToyPlugin" or via register_plugin.
    """
    crash_at: int | None = None
    _n = 0

    def build_model(self) -> nn.Module:
        return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))

    def _data(self, seed: int):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(64, 8, generator=g)
        return torch.utils.data.TensorDataset(x, (x[:, 0] > 0).long() + (x[:, 1] > 0).long())

    def build_loaders(self, run):
        mk = lambda s, shuf: torch.utils.data.DataLoader(self._data(s), batch_size=16, shuffle=shuf)  # noqa: E731
        return mk(0, True), mk(1, False)

    def configure_phase(self, model: nn.Module, phase: PhaseCfg) -> list[dict]:
        for i, p in enumerate(model.parameters()):
            p.requires_grad = phase.unfreeze_blocks > 0 or i >= 2
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": phase.head_lr}]

    def loss(self, model, batch, device):
        x, y = batch
        return nn.functional.cross_entropy(model(x.to(device)), y.to(device))

    def evaluate(self, model, loader, device):
        self._n += 1
        if type(self).crash_at == self._n:
            type(self).crash_at = None
            raise RuntimeError("simulated crash")
        with torch.no_grad():
            x, y = loader.dataset.tensors
            out = model(x.to(device))
            return {"val_loss": nn.functional.cross_entropy(out, y.to(device)).item(),
                    "val_balanced_acc": (out.argmax(1).cpu() == y).float().mean().item()}


def toy_config_dict(run_name: str = "toy", phases: list[dict] | None = None, patience: int = 2) -> dict:
    phases = phases or [{"name": "head", "epochs": 2, "unfreeze_blocks": 0, "head_lr": 1e-2},
                        {"name": "ft", "epochs": 3, "unfreeze_blocks": 1, "head_lr": 1e-2}]
    return {"run_name": run_name, "plugin": "src.tests.fixtures:ToyPlugin", "device": "cpu",
            "games": {"train": ["IMG_0106"], "test": ["IMG_0108"]},
            "train": {"num_workers": 0, "batch_size": 16, "patience": patience, "phases": phases}}


def toy_config(**kw) -> Config:
    return Config.model_validate(toy_config_dict(**kw))
