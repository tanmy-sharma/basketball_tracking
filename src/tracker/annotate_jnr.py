#!/usr/bin/env python
"""Draw JNR-stitched tracks on a video: "A JNR 10" above a box whose piece got an identity, "trk 105" when it got none.

  python src/tracker/annotate_jnr.py IMG_0104 J.txt J.scores.json OUT.mp4 [--video PATH] [--start F] [--frames N]

J.txt is jnr_tracks.py's output and J.scores.json its scores file. Ids 1-6 are identities, in the order of the scores file's
roster: team A's numbers, then team B's. Ids of 100 and up have no identity. Each identity keeps one box colour throughout.
No ground truth is used.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from tracker.annotate import colour  # noqa: E402
from tracker.cross_team import load  # noqa: E402
from trackid import io  # noqa: E402


def labels_of(scores: Path) -> dict[int, str]:
    teams = json.loads(scores.read_text())["teams"]
    names = [(t, n) for t in sorted(teams) for n in teams[t]]
    return {i: f"{t} JNR {n}" for i, (t, n) in enumerate(names, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("tracks")
    ap.add_argument("scores", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--registration", default=str(io.TRAIN / "court_registration.csv"))
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--frames", type=int)
    a = ap.parse_args()
    video = a.video or io.TRAIN / "vidoes" / f"{a.name}.MOV"
    names = labels_of(a.scores)

    t = load(a.tracks)
    by_frame = {f: g for f, g in t.groupby("frame")}
    W, H = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                    "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout.strip().split(",")[:2])
    width, scale = round(W * a.height / H / 2) * 2, a.height / H
    _, c2i = io.load_homographies(a.registration)

    sel = f"select='gte(n,{a.start - 1})" + (f"*lt(n,{a.start - 1 + a.frames})'" if a.frames else "'")
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-hwaccel", "auto", "-i", str(video), "-vf", f"{sel},scale={width}:{a.height}:flags=area",
                            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{a.height}", "-r", "30000/1001",
                            "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(a.out)], stdin=subprocess.PIPE)
    size, frame = width * a.height * 3, a.start
    while len(buf := dec.stdout.read(size)) == size:
        img = np.frombuffer(buf, np.uint8).reshape(a.height, width, 3).copy()
        g = by_frame.get(frame)
        for r in (() if g is None else g.itertuples()):
            tid = int(r.id)
            label = names.get(tid, f"trk {tid}")
            c = colour(tid)
            if r.kind == "det":
                p1, p2 = (int(r.bx * scale), int(r.by * scale)), (int((r.bx + r.bw) * scale), int((r.by + r.bh) * scale))
                cv2.rectangle(img, p1, p2, c, 2, cv2.LINE_AA)
                x, y = p1
            else:
                u, v = io.apply_h(c2i, [[r.x, r.y]])[0] * scale
                x, y = int(u), int(v)
                cv2.circle(img, (x, y), 6, c, 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y), c, -1)
            cv2.putText(img, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"frame {frame}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        enc.stdin.write(img.tobytes())
        frame += 1
    enc.stdin.close()
    dec.wait()
    if enc.wait():
        sys.exit("ffmpeg failed")
    print(f"{a.out}: frames {a.start}-{frame - 1}")


if __name__ == "__main__":
    main()
