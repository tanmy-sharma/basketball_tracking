#!/usr/bin/env python
"""Write IMG_0114.txt and IMG_0115.txt from tracker output and validate the folder.

  python src/tracker/make_submission.py OUT_DIR --tracks IMG_0114=a.txt IMG_0115=b.txt [--frames IMG_0114=N ...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from tracker.identity import make_rows, read_tracks  # noqa: E402
from trackid import io  # noqa: E402
from validate_submission import frame_count  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tracks", nargs="+", required=True, metavar="VIDEO=FILE")
    ap.add_argument("--frames", nargs="*", default=[], metavar="VIDEO=N", help="override cv2 frame counts")
    ap.add_argument("--roster", type=Path, default=io.FINAL / "games_roster.csv")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--min-len", type=int, default=15)
    a = ap.parse_args()
    a.out_dir = a.out_dir.resolve()
    files = dict(s.split("=", 1) for s in a.tracks)
    counts = {k: int(v) for k, v in (s.split("=") for s in a.frames)}
    roster = io.load_roster(a.roster)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for video, f in files.items():
        n = counts.get(video) or frame_count(video)
        rows = make_rows(read_tracks(f), roster[video].teams, n, a.k, a.min_len)
        io.write_submission(a.out_dir / f"{video}.txt", rows)
        print(f"{video}: {len(rows)} rows, {n} frames")
    cmd = [sys.executable, str(ROOT / "tools" / "validate_submission.py"), str(a.out_dir)]
    if counts:
        cmd += ["--frames", *a.frames]
    sys.exit(subprocess.run(cmd, cwd=ROOT / "tools").returncode)


if __name__ == "__main__":
    main()
