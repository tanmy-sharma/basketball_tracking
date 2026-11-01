#!/usr/bin/env python
"""Draw projected court lines (and optional points) on a video frame.

  python viz_overlay.py VIDEO REGISTRATION.csv FRAME [--points file.txt] [--out out.png]
FRAME is 1-based (same numbering as annotations/submissions). Points file rows:
frame,court_x,court_y,team,jersey (GT court-position format == submission format).
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from trackid import io

HOOP = (752.5, 157.5)


def court_polylines() -> list[np.ndarray]:
    W, H = io.COURT_W, io.COURT_H
    rect = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]], float)
    kx0, kx1, ky = HOOP[0] - 245, HOOP[0] + 245, 580
    key = np.array([[kx0, 0], [kx0, ky], [kx1, ky], [kx1, 0]], float)
    th = np.linspace(0, np.pi, 200)
    arc = np.c_[HOOP[0] + 675 * np.cos(th), HOOP[1] + 675 * np.sin(th)]
    arc = arc[(arc[:, 0] >= 0) & (arc[:, 0] <= W) & (arc[:, 1] <= H)]
    mid = np.array([[0, H / 2], [W, H / 2]], float)
    return [rect, key, arc, mid]


def read_frame(video: str, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {frame} of {video}")
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("registration")
    ap.add_argument("frame", type=int)
    ap.add_argument("--points")
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--scale", type=float, default=1.0, help="resize output (e.g. 0.5 for 4K)")
    a = ap.parse_args()

    img = read_frame(a.video, a.frame)
    _, Hci = io.load_homographies(a.registration)
    th = max(1, round(img.shape[1] / 1000))
    for pl in court_polylines():
        pts = io.apply_h(Hci, pl).round().astype(np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), th, cv2.LINE_AA)
    if a.points:
        for line in open(a.points):
            p = line.strip().split(",")
            if len(p) >= 5 and int(p[0]) == a.frame:
                u, v = io.apply_h(Hci, [[float(p[1]), float(p[2])]])[0]
                col = (255, 80, 80) if p[3] == "A" else (80, 80, 255)
                cv2.circle(img, (int(u), int(v)), 4 * th, col, -1, cv2.LINE_AA)
                cv2.putText(img, p[3] + p[4], (int(u) + 6 * th, int(v) - 6 * th), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4 * th, col, th, cv2.LINE_AA)
    if a.scale != 1:
        img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(a.out, img):
        sys.exit(f"failed to write {a.out} (check the directory exists and the extension is an image type)")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
