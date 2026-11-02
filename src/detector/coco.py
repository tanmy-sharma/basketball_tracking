"""Roboflow-layout COCO export at 720p for RF-DETR: train/ and valid/ with _annotations.coco.json."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import pandas as pd

from src.core.config import Config
from src.core.constants import CHALLENGE_VIDEOS
from src.core.paths import outputs_dir
from src.dataio.labels import load_labelled_boxes
from src.dataio.split import assign_split
from src.dataio.video import iter_frames, probe, video_path

HEIGHT = 720
MIN_SIDE = 2.0
CATEGORIES = [{"id": 1, "name": "player", "supercategory": "player"}]  # own name as parent keeps it under filter_parent_categories


def dataset_hash(cfg: Config) -> str:
    d = cfg.detector
    key = {"train": sorted(cfg.games.train), "split": cfg.split.model_dump(mode="json"),
           "every": d.sample_every, "q": d.jpeg_quality, "height": HEIGHT}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]


def scale_boxes(boxes: pd.DataFrame, sx: float, sy: float, w: int, h: int) -> list[list[float]]:
    """4K xywh -> clamped 720p xywh, dropping boxes with a side under MIN_SIDE after clamping."""
    out = []
    for x, y, bw, bh in boxes[["x", "y", "w", "h"]].to_numpy(float):
        x0, y0 = min(max(x * sx, 0.0), w), min(max(y * sy, 0.0), h)
        x1, y1 = min(max((x + bw) * sx, 0.0), w), min(max((y + bh) * sy, 0.0), h)
        if x1 - x0 >= MIN_SIDE and y1 - y0 >= MIN_SIDE:
            out.append([round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)])
    return out


def _candidates(cfg: Config) -> pd.DataFrame:
    """Sampled frames that have labels, with a split column. Frames without labels are never exported."""
    every, rows = cfg.detector.sample_every, []
    for g in cfg.games.train:
        frames = load_labelled_boxes(g).frame.unique()
        rows += [(g, int(f)) for f in sorted(frames) if (f - 1) % every == 0]
    df = pd.DataFrame(rows, columns=["video", "frame"])
    df["split"] = assign_split(df, cfg.split)
    return df[df.split != "guard"].reset_index(drop=True)


def build_dataset(cfg: Config) -> Path:
    """Write (or reuse) outputs/detector_dataset/<hash>/ and return it."""
    banned = (set(cfg.games.train) & set(cfg.games.test)) | (set(cfg.games.train) & CHALLENGE_VIDEOS)
    if banned:
        raise ValueError(f"refusing to export test/challenge games: {sorted(banned)}")
    if cfg.detector is None:
        raise ValueError("config has no detector block")
    root = outputs_dir() / "detector_dataset" / dataset_hash(cfg)
    if (root / "done.json").exists():
        return root
    tmp = root.with_name(root.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    cand = _candidates(cfg)
    if cand.empty:
        raise ValueError("no labelled frames to export")
    coco = {s: {"info": {}, "images": [], "annotations": [], "categories": CATEGORIES} for s in ("train", "valid")}
    for s in coco:
        (tmp / s).mkdir(parents=True)
    quality = [cv2.IMWRITE_JPEG_QUALITY, cfg.detector.jpeg_quality]
    for g in cfg.games.train:
        sub = cand[cand.video == g].set_index("frame").split.map({"train": "train", "val": "valid"})
        if sub.empty:
            continue
        boxes = {f: b for f, b in load_labelled_boxes(g).groupby("frame") if f in sub.index}
        path = video_path(g)
        src_w, src_h = probe(path)
        last = max(sub.index)
        for frame, img in iter_frames(path, cfg.detector.sample_every, HEIGHT):
            if frame > last:
                break
            if frame not in sub.index:
                continue
            h, w = img.shape[:2]
            anns = scale_boxes(boxes[frame], w / src_w, h / src_h, w, h)
            if not anns:
                continue
            c = coco[sub[frame]]
            name = f"{g}_{frame:06d}.jpg"
            cv2.imwrite(str(tmp / sub[frame] / name), img[:, :, ::-1], quality)
            iid = len(c["images"]) + 1
            c["images"].append({"id": iid, "file_name": name, "width": w, "height": h})
            for b in anns:
                c["annotations"].append({"id": len(c["annotations"]) + 1, "image_id": iid, "category_id": 1,
                                         "bbox": b, "area": round(b[2] * b[3], 2), "iscrowd": 0})
    for s, c in coco.items():
        (tmp / s / "_annotations.coco.json").write_text(json.dumps(c))
    meta = {s: {"images": len(c["images"]), "boxes": len(c["annotations"])} for s, c in coco.items()}
    if not meta["train"]["images"] or not meta["valid"]["images"]:
        raise ValueError(f"export left an empty split: {meta}")
    (tmp / "done.json").write_text(json.dumps({"games": cfg.games.train, **meta}))
    shutil.rmtree(root, ignore_errors=True)
    tmp.rename(root)
    return root
