#!/usr/bin/env python
"""Validate a TrackID3x3 submission (directory or zip) and optionally build the upload ZIP.

  python validate_submission.py path/to/dir_or.zip [--make-zip out.zip] [--frames IMG_0114=11536 ...]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2

from trackid import io

VIDEOS = ["IMG_0114", "IMG_0115"]
INT_RE = re.compile(r"^(0|[1-9]\d*)$")


def frame_count(video: str) -> int:
    cap = cv2.VideoCapture(str(io.FINAL / f"{video}.MP4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def check_file(name: str, text: str, n_frames: int, roster: io.Roster, err: list, warn: list) -> None:
    P = lambda m: err.append(f"{name}: {m}")
    W = lambda m: warn.append(f"{name}: {m}")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return P("empty file")
    if "﻿" in lines[0]:
        P("BOM at start of file")
    per_frame: dict[int, list] = {}
    xs, ys, bad = [], [], 0
    for ln, line in enumerate(lines, 1):
        line = line.rstrip("\r")
        p = line.split(",")
        if len(p) != 5:
            P(f"line {ln}: expected 5 fields, got {len(p)}: {line[:60]!r}")
            bad += 1
        elif any(f != f.strip() or " " in f for f in p):
            P(f"line {ln}: whitespace inside a field (breaks exact string match): {line[:60]!r}")
            bad += 1
        elif not INT_RE.match(p[0]):
            P(f"line {ln}: frame id not an int (header?): {p[0]!r}")
            bad += 1
        else:
            try:
                x, y = float(p[1]), float(p[2])
            except ValueError:
                P(f"line {ln}: non-numeric coordinates")
                bad += 1
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                P(f"line {ln}: non-finite coordinates (NaN/inf crash the evaluator): {line[:60]!r}")
                bad += 1
                continue
            if p[3] not in ("A", "B"):
                P(f"line {ln}: team must be A or B, got {p[3]!r}")
            elif not INT_RE.match(p[4]) or (p[3], p[4]) not in roster.identities:
                P(f"line {ln}: ({p[3]},{p[4]}) not in roster {sorted(roster.identities)}")
            xs.append(x), ys.append(y)
            per_frame.setdefault(int(p[0]), []).append((p[3], p[4]))
        if bad >= 20:
            return P("too many malformed lines, stopping")
    frames = set(per_frame)
    missing = sorted(set(range(1, n_frames + 1)) - frames)
    extra = sorted(frames - set(range(1, n_frames + 1)))
    if missing:
        P(f"{len(missing)} frames in 1..{n_frames} have no rows (first: {missing[:5]})")
    if extra:
        P(f"{len(extra)} frame ids outside 1..{n_frames} (first: {extra[:5]})")
    if xs and max(map(abs, xs + ys)) < 20:
        W("all |coords| < 20: looks like meters, expected cm (court is 1505 x 1105)")
    if xs and (min(xs) < -500 or max(xs) > 2000 or min(ys) < -500 or max(ys) > 1600):
        W("some coordinates are far outside the court (1505 x 1105 cm)")
    not6 = [f for f, r in per_frame.items() if len(r) != 6]
    if not6:
        W(f"{len(not6)} frames do not have exactly 6 rows (first: {sorted(not6)[:5]})")
    dups = [f for f, r in per_frame.items() if max(Counter(r).values()) > 1]
    if dups:
        W(f"{len(dups)} frames repeat an identity (first: {sorted(dups)[:5]})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", help="directory or .zip")
    ap.add_argument("--make-zip", metavar="OUT.ZIP", help="write a correct root-level ZIP (only if valid)")
    ap.add_argument("--frames", nargs="*", default=[], metavar="VIDEO=N", help="override frame counts")
    a = ap.parse_args()

    src = Path(a.submission)
    err: list[str] = []
    warn: list[str] = []
    texts: dict[str, str] = {}
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            for n, c in Counter(z.namelist()).items():
                if c > 1:
                    err.append(f"zip has {c} entries named {n} (ambiguous which one is evaluated)")
            for info in z.infolist():
                n = info.filename
                if n.endswith("/"):
                    continue
                if "/" in n:
                    err.append(f"zip entry not at root: {n}")
                # read by ZipInfo, not name, so every duplicate entry gets checked
                text = z.read(info).decode("utf-8", "replace")
                if n in texts:
                    n = f"{n} (duplicate)"
                texts[n] = text
    else:
        texts = {p.name: p.read_text() for p in src.iterdir() if p.is_file()}
    expected = {f"{v}.txt" for v in VIDEOS}
    for n in sorted(set(texts) - expected):
        warn.append(f"unexpected file: {n}")
    for n in sorted(expected - set(texts)):
        err.append(f"missing file: {n}")

    counts = {k: int(v) for k, v in (s.split("=") for s in a.frames)}
    rosters = io.load_roster(io.FINAL / "games_roster.csv")
    for v in VIDEOS:
        if f"{v}.txt" in texts:
            n = counts.get(v) or frame_count(v)
            print(f"{v}: {n} frames expected")
            check_file(f"{v}.txt", texts[f"{v}.txt"], n, rosters[v], err, warn)
    for w in warn:
        print("WARN ", w)
    for e in err:
        print("ERROR", e)
    print(f"{len(err)} error(s), {len(warn)} warning(s)")
    if a.make_zip and not err:
        with zipfile.ZipFile(a.make_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for n in sorted(expected):
                z.writestr(n, texts[n])
        print(f"wrote {a.make_zip}")
    elif a.make_zip:
        print("not writing zip: fix errors first")
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
