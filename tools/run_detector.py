#!/usr/bin/env python
"""Run public COCO detectors on a video in one decode pass and cache their person boxes.

  python run_detector.py VIDEO.MOV [--models rfdetr-nano rfdetr-small] [--height 720] [--every 30] [--limit N]

Frames are downscaled to --height before detection, to match the 720p challenge videos. Output is MOT det format
in full-resolution pixels, 1-based frames, in ../outputs/detections/: frame,-1,x,y,w,h,conf,-1,-1,-1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from inference_models import AutoModel

OUT = Path(__file__).resolve().parents[1] / "outputs" / "detections"
CONF_FLOOR = 0.05  # low on purpose: ByteTrack-style trackers use low-score boxes, and eval sweeps thresholds


def probe(video: Path) -> tuple[int, int]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


def frame_count(video: Path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True)
    return int(r.stdout.strip().strip(","))


def frames(video: Path, width: int, height: int, every: int):
    """Yield (1-based frame number, BGR image) from an ffmpeg pipe, without dropping or repeating frames."""
    # -fps_mode replaced -vsync in ffmpeg 5.1, and Colab may ship an older build
    help_text = subprocess.run(["ffmpeg", "-h", "full"], capture_output=True, text=True).stdout
    sync = "-fps_mode" if "fps_mode" in help_text else "-vsync"
    filters = f"select='not(mod(n,{every}))',scale={width}:{height}:flags=area"
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "auto", "-i", str(video), "-vf", filters,
           sync, "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = width * height * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        n = 0
        while len(buf := p.stdout.read(size)) == size:
            yield n * every + 1, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            n += 1
    finally:
        p.kill()
        p.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--models", nargs="+", default=["rfdetr-small"])
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--limit", type=int, help="stop after N processed frames (smoke test, skips the frame-count check)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    W, H = probe(a.video)
    width = round(W * a.height / H / 2) * 2
    scale = W / width
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"__{a.height}p" + (f"__every{a.every}" if a.every > 1 else "")

    runs = []
    for name in a.models:
        model = AutoModel.from_pretrained(name, device=a.device)
        path = OUT / f"{a.video.stem}__{name}{suffix}.txt"
        runs.append(dict(name=name, model=model, person=list(model.class_names).index("person"), path=path,
                         tmp=path.with_suffix(".tmp"), f=open(path.with_suffix(".tmp"), "w"), secs=0.0))

    n = 0
    for frame, img in frames(a.video, width, a.height, a.every):
        for r in runs:
            t = time.perf_counter()
            det = r["model"](img, confidence=CONF_FLOOR)[0].to_supervision()
            r["secs"] += time.perf_counter() - t
            keep = det.class_id == r["person"]
            for (x1, y1, x2, y2), c in zip(det.xyxy[keep] * scale, det.confidence[keep]):
                r["f"].write(f"{frame},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f},-1,-1,-1\n")
        n += 1
        if n % 500 == 0:
            print(f"{n} frames: " + ", ".join(f"{r['name']} {n / r['secs']:.1f} fps" for r in runs), flush=True)
        if a.limit and n >= a.limit:
            break

    for r in runs:
        r["f"].close()
    # Frames are numbered in decode order with passthrough sync, so a shortfall can only be missing frames at the end. The
    # packet count can be higher than what decodes (IMG_0104: 11400 packets, 11376 frames, matching its GT), so warn only.
    expected = -(-frame_count(a.video) // a.every)
    if not a.limit and n > expected:
        sys.exit(f"decoded {n} frames but the video has only {expected} packets; frame numbers would not match the GT")
    if not a.limit and n < expected:
        print(f"WARNING: decoded {n} frames, {expected - n} fewer than the packet count. Kept: numbering is unaffected")
    for r in runs:
        r["tmp"].rename(r["path"])  # only a finished run appears under the final name
        print(f"{r['name']}: {n} frames, {n / r['secs']:.1f} fps -> {r['path']}")


if __name__ == "__main__":
    main()
