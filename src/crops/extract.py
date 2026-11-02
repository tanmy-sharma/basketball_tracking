"""One game -> crops.bin, manifest.csv, keypoints.npy, meta.json, done.json.

  python -m src.crops.extract --config C.yaml [--games IMG_0106 ...] [--set a.b=c]
  python -m src.crops.extract --overlay --games IMG_0104      # read-only alignment check, no pose, no crops
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.core.config import CropCfg, load_config
from src.core.constants import CHALLENGE_VIDEOS
from src.core.paths import outputs_dir
from src.crops.geometry import MIN_KP_CONF, TORSO, facing, max_iou_per_box, torso_box
from src.crops.pose import PoseEstimator
from src.dataio.labels import load_labelled_boxes
from src.dataio.schema import CROP_COLUMNS, crop_id
from src.dataio.store import CropWriter, crops_root, mark_done, save_manifest
from src.dataio.video import frame_count, iter_frames, read_frames, scaled_size, video_path

log = logging.getLogger(__name__)
LABEL_W = 3840  # labels are in 4K pixels whatever the video resolution
LOW_POSE_CONF = MIN_KP_CONF  # below this torso-keypoint confidence a row is 'low_pose' (pixels are kept)


def extract_game(video: str, crops: CropCfg, pose: PoseEstimator, out_root: Path | None = None) -> Path:
    """Extract crops for `video` into <out_root or crops_root(crops)>/<video> and return that directory.

    Labels may end before the video does (the gap is logged); a video shorter than its labels raises,
    except under limit_frames. Writes go to <video>.tmp, which is renamed into place before done.json.
    """
    if video in CHALLENGE_VIDEOS:
        raise ValueError(f"{video} is a challenge video")
    out_root = Path(out_root) if out_root else crops_root(crops)
    final, tmp = out_root / video, out_root / f"{video}.tmp"
    path = video_path(video)
    boxes = load_labelled_boxes(video)
    label_last = int(boxes.frame.max())
    n_video = frame_count(path)
    limited = crops.limit_frames is not None
    if not limited:
        if n_video < label_last:
            raise RuntimeError(f"{video}: video has {n_video} frames but labels reach frame {label_last}")
        if n_video > label_last:
            log.warning("%s: labels end at frame %d but the video has %d frames (%d unlabelled at the end)",
                        video, label_last, n_video, n_video - label_last)
    width, height, _ = scaled_size(path, crops.height)
    scale = LABEL_W / width
    by_frame = {int(f): g for f, g in boxes.groupby("frame")}
    n_sampled = math.ceil(crops.limit_frames / crops.every) if limited else None

    shutil.rmtree(tmp, ignore_errors=True)
    rows: list[dict] = []
    kps_all: list[np.ndarray] = []
    decoded = 0
    t0 = time.time()
    with CropWriter(tmp) as writer:
        for frame, img in iter_frames(path, crops.every, crops.height, n_sampled):
            decoded += 1
            g = by_frame.get(frame)
            if g is None:
                continue
            b = g[["x", "y", "w", "h"]].to_numpy(np.float64) / scale
            ious = max_iou_per_box(b)
            ok = (b[:, 2] >= 1) & (b[:, 3] >= 1)
            kps = np.zeros((len(g), 17, 3), np.float32)
            if ok.any():
                kps[ok] = pose(img, b[ok])
            for i, r in enumerate(g.itertuples()):
                tb = torso_box(kps[i], crops.pad_frac, crops.min_aspect, crops.min_side_px,
                               img.shape[1], img.shape[0]) if ok[i] else None
                conf = float(kps[i][list(TORSO), 2].min()) if ok[i] else 0.0
                if tb is None:
                    x0 = y0 = x1 = y1 = 0
                    status, off = "degenerate", -1
                else:
                    x0, y0, x1, y1 = tb
                    off = writer.add(np.ascontiguousarray(img[y0:y1, x0:x1]))
                    status = "ok" if conf >= LOW_POSE_CONF else "low_pose"
                rows.append({
                    "crop_id": crop_id(video, frame, r.track_id), "video": video, "frame": frame,
                    "track_id": r.track_id, "team": r.team, "jersey": r.jersey, "number_idx": r.number_idx,
                    "bib": r.bib, "in_game": r.in_game, "box_x": b[i, 0], "box_y": b[i, 1], "box_w": b[i, 2],
                    "box_h": b[i, 3], "max_iou": ious[i], "pose_min_conf": conf, "crop_x0": x0, "crop_y0": y0,
                    "crop_x1": x1, "crop_y1": y1, "crop_w": x1 - x0, "crop_h": y1 - y0, "offset": off,
                    "facing": facing(kps[i]) if ok[i] else "unknown", "status": status})
                kps_all.append(kps[i])
    expected = n_sampled if limited else math.ceil(n_video / crops.every)
    # Every labelled sampled frame must decode. The unlabelled tail may not: IMG_0104's container counts 11,400 frames
    # but ffmpeg decodes about 11,380, past its last label at 11,376.
    needed = n_sampled if limited else math.ceil(label_last / crops.every)
    if decoded > expected or (not limited and decoded < needed):
        raise RuntimeError(f"{video}: decoded {decoded} sampled frames, expected {needed}-{expected}")
    if not limited and decoded < expected:
        log.warning("%s: decoded %d of %d sampled frames; the missing ones are all after the last label (frame %d)",
                    video, decoded, expected, label_last)
    if not rows:
        raise RuntimeError(f"{video}: no labelled boxes on the sampled frames")

    df = pd.DataFrame(rows, columns=CROP_COLUMNS)
    save_manifest(tmp, df)
    np.save(tmp / "keypoints.tmp.npy", np.stack(kps_all).astype(np.float32))
    os.replace(tmp / "keypoints.tmp.npy", tmp / "keypoints.npy")
    counts = df.status.value_counts().to_dict()
    kept = df[df.crop_w > 0]
    med = (float(kept.crop_w.median()), float(kept.crop_h.median())) if len(kept) else (0.0, 0.0)
    (tmp / "meta.json").write_text(json.dumps({
        "video": video, "decoded_sampled_frames": decoded, "video_frames": n_video, "label_last_frame": label_last,
        "unlabelled_tail_frames": max(0, n_video - label_last), "limit_frames": crops.limit_frames,
        "rows": len(df), "status_counts": counts, "median_crop_wh": med, "scale": scale,
        "cfg": crops.model_dump(), "pose_revision": crops.pose_revision}, indent=1))
    log.info("%s: %d rows, status %s, median crop %.0fx%.0f, %.0fs", video, len(df), counts, *med, time.time() - t0)

    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    mark_done(final, video=video, rows=len(df))
    return final


def write_overlays(video: str, out_dir: Path | None = None, height: int = 720) -> list[Path]:
    """Read-only: PNGs of labelled boxes at the first, middle and last labelled frame, to check alignment.

    Decodes the video sequentially up to the last labelled frame, so it takes about as long as one full pass.
    """
    boxes = load_labelled_boxes(video)
    frames = sorted(boxes.frame.unique())
    targets = sorted({int(frames[0]), int(frames[len(frames) // 2]), int(frames[-1])})
    path = video_path(video)
    scale = LABEL_W / scaled_size(path, height)[0]
    out_dir = Path(out_dir) if out_dir else outputs_dir() / "overlays" / video
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    got = read_frames(path, targets, height)
    for f, img in got.items():
        canvas = np.ascontiguousarray(img[:, :, ::-1])
        for r in boxes[boxes.frame == f].itertuples():
            x0, y0 = int(r.x / scale), int(r.y / scale)
            cv2.rectangle(canvas, (x0, y0), (int((r.x + r.w) / scale), int((r.y + r.h) / scale)), (0, 255, 0), 2)
            cv2.putText(canvas, f"{r.track_id}:{r.jersey}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
        p = out_dir / f"{video}_f{f:06d}.png"
        cv2.imwrite(str(p), canvas)
        out.append(p)
    if missing := sorted(set(targets) - set(got)):
        log.warning("%s: frames %s not in the video", video, missing)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--games", nargs="+")
    ap.add_argument("--overlay", action="store_true", help="write alignment overlays instead of extracting")
    ap.add_argument("--out", help="overlay output dir")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Hub request otherwise
    cfg = load_config(a.config, a.overrides) if a.config else None
    games = a.games or (cfg.games.train + cfg.games.test + cfg.games.label_pool if cfg else [])
    if not games:
        ap.error("give --games or --config")
    if a.overlay:
        for g in games:
            for p in write_overlays(g, Path(a.out) / g if a.out else None):
                print(p)
        return
    if cfg is None:
        ap.error("--config is required to extract")
    import torch
    from src.crops.pose import ViTPose
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else
                                                 "mps" if torch.backends.mps.is_available() else "cpu")
    pose = None
    for g in dict.fromkeys(games):
        if (crops_root(cfg.crops) / g / "done.json").exists():
            print(f"{g}: done, skipping")
            continue
        pose = pose or ViTPose(cfg.crops.pose_model, cfg.crops.pose_revision, device)
        print(extract_game(g, cfg.crops, pose))


if __name__ == "__main__":
    main()
