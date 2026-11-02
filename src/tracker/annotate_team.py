#!/usr/bin/env python
"""Draw team-aware court tracks on a video, and on training videos mark the cross-team errors.

  python src/tracker/annotate_team.py IMG_0107 TRACKS.txt OUT.mp4 [--video PATH] [--start F] [--frames N] [--no-gt]

Each box: colour per track, label "id team" where team is the track's belief (A, B, or ? when unsure, |log-odds| < 3),
and a small dot in the box's own colour call (white A, black B, none for no call). On training videos, a box on a GT player
of the other team from its track's main player gets a thick red outline and "X"; GT is used only to draw that.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from tracker.annotate import colour  # noqa: E402
from tracker.cross_team import load, rows_with_owner  # noqa: E402
from trackid import io  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("tracks")
    ap.add_argument("out", type=Path)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--registration", default=str(io.TRAIN / "court_registration.csv"))
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--no-gt", action="store_true")
    a = ap.parse_args()
    video = a.video or io.TRAIN / "vidoes" / f"{a.name}.MOV"

    t = load(a.tracks)
    if not a.no_gt:
        t = rows_with_owner(a.name, t)
    else:
        t["cross"] = False
    by_frame = {f: g for f, g in t.groupby("frame")}
    W, H = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                    "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout.strip().split(",")[:2])
    width, scale = round(W * a.height / H / 2) * 2, a.height / H
    _, c2i = io.load_homographies(a.registration)
    outline = (io.apply_h(c2i, [[0, 0], [io.COURT_W, 0], [io.COURT_W, io.COURT_H], [0, io.COURT_H]]) * scale).round().astype(np.int32)

    sel = f"select='gte(n,{a.start - 1})" + (f"*lt(n,{a.start - 1 + a.frames})'" if a.frames else "'")
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-hwaccel", "auto", "-i", str(video), "-vf", f"{sel},scale={width}:{a.height}:flags=area",
                            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{a.height}", "-r", "30000/1001",
                            "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(a.out)], stdin=subprocess.PIPE)
    size, frame, n_cross = width * a.height * 3, a.start, 0
    while len(buf := dec.stdout.read(size)) == size:
        img = np.frombuffer(buf, np.uint8).reshape(a.height, width, 3).copy()
        cv2.polylines(img, [outline.reshape(-1, 1, 2)], True, (0, 255, 255), 1, cv2.LINE_AA)
        g = by_frame.get(frame)
        crosses = 0
        for r in (() if g is None else g.itertuples()):
            c = colour(int(r.id))
            belief = getattr(r, "belief", 0.0)
            team = "A" if belief >= 3 else "B" if belief <= -3 else "?"
            if r.kind == "det":
                p1, p2 = (int(r.bx * scale), int(r.by * scale)), (int((r.bx + r.bw) * scale), int((r.by + r.bh) * scale))
                cv2.rectangle(img, p1, p2, c, 2, cv2.LINE_AA)
                call = getattr(r, "team", 0)
                if call:
                    cv2.circle(img, (p1[0] + 7, p1[1] + 7), 5, (255, 255, 255) if call > 0 else (0, 0, 0), -1, cv2.LINE_AA)
                if r.cross:
                    crosses += 1
                    cv2.rectangle(img, (p1[0] - 3, p1[1] - 3), (p2[0] + 3, p2[1] + 3), (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.putText(img, "X", (p2[0] - 18, p2[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                x, y = p1
            else:
                u, v = io.apply_h(c2i, [[r.x, r.y]])[0] * scale
                x, y = int(u), int(v)
                cv2.circle(img, (x, y), 6, c, 2, cv2.LINE_AA)
            label = f"{int(r.id)} {team}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y), c, -1)
            cv2.putText(img, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        n_cross += crosses
        cv2.putText(img, f"frame {frame}  cross-team boxes {crosses}  (total {n_cross})", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        enc.stdin.write(img.tobytes())
        frame += 1
    enc.stdin.close()
    dec.wait()
    if enc.wait():
        sys.exit("ffmpeg failed")
    print(f"{a.out}: frames {a.start}-{frame - 1}, {n_cross} cross-team boxes drawn")


if __name__ == "__main__":
    main()
