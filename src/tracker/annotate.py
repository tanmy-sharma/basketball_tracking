#!/usr/bin/env python
"""Draw court tracks on a video: a box and track ID per player, plus a circle where a track has no detection (coasting).

  python src/tracker/annotate.py VIDEO TRACKS.txt REGISTRATION.csv OUT.mp4 [--height 720] [--start FRAME] [--frames N]

TRACKS is the output of run_ucmc.py. The registration must be in the same pixel scale as the boxes.
"""
from __future__ import annotations

import argparse
import colorsys
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from tracker.evaluate import load_tracks  # noqa: E402
from trackid import io  # noqa: E402


def colour(track_id: int) -> tuple[int, int, int]:
    """A stable, distinct BGR colour per track (golden-ratio hue steps)."""
    r, g, b = colorsys.hsv_to_rgb((track_id * 0.618034) % 1.0, 0.9, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("tracks")
    ap.add_argument("registration")
    ap.add_argument("out", type=Path)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--start", type=int, default=1, help="first frame, 1-based")
    ap.add_argument("--frames", type=int, help="number of frames (default: to the end)")
    a = ap.parse_args()

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(a.video)], capture_output=True, text=True, check=True)
    W, H = map(int, probe.stdout.strip().split(",")[:2])
    width = round(W * a.height / H / 2) * 2
    scale = a.height / H

    _, court_to_image = io.load_homographies(a.registration)
    tracks = load_tracks(a.tracks)
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    outline = io.apply_h(court_to_image, [[0, 0], [io.COURT_W, 0], [io.COURT_W, io.COURT_H], [0, io.COURT_H]]) * scale
    outline = outline.round().astype(np.int32).reshape(-1, 1, 2)

    # Frames are selected by index, not seek time, so numbering matches the annotations
    last = f"+{a.frames}" if a.frames else ""
    select = f"select='gte(n,{a.start - 1})" + (f"*lt(n,{a.start - 1 + a.frames})'" if last else "'")
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-hwaccel", "auto", "-i", str(a.video), "-vf",
                            f"{select},scale={width}:{a.height}:flags=area", "-fps_mode", "passthrough",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{a.height}",
                            "-r", "30000/1001", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)], stdin=subprocess.PIPE)

    size = width * a.height * 3
    frame, written = a.start, 0
    while len(buf := dec.stdout.read(size)) == size:
        img = np.frombuffer(buf, np.uint8).reshape(a.height, width, 3).copy()
        cv2.polylines(img, [outline], True, (0, 255, 255), 1, cv2.LINE_AA)
        g = by_frame.get(frame)
        if g is not None:
            for r in g.itertuples():
                c = colour(int(r.id))
                if r.kind == "det":
                    x, y = int(r.bx * scale), int(r.by * scale)
                    cv2.rectangle(img, (x, y), (int((r.bx + r.bw) * scale), int((r.by + r.bh) * scale)), c, 2, cv2.LINE_AA)
                else:
                    u, v = io.apply_h(court_to_image, [[r.x, r.y]])[0] * scale
                    x, y = int(u), int(v)
                    cv2.circle(img, (x, y), 6, c, 2, cv2.LINE_AA)
                label = str(int(r.id))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y), c, -1)
                cv2.putText(img, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        n = 0 if g is None else len(g)
        cv2.putText(img, f"frame {frame}  rows {n}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        enc.stdin.write(img.tobytes())
        frame += 1
        written += 1
        if written % 1000 == 0:
            print(f"{written} frames", flush=True)
    enc.stdin.close()
    dec.wait()
    if enc.wait():
        sys.exit("ffmpeg failed to encode the video")
    print(f"{a.out}: {written} frames")


if __name__ == "__main__":
    main()
