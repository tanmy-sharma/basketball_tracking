"""ffmpeg frame iterator: RGB, downscaled, every Nth frame, 1-based frame numbers matching the labels."""
from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.core.constants import CHALLENGE_VIDEOS
from src.core.paths import add_tools_to_path

SUFFIXES = (".mov", ".mkv", ".mp4")


def video_path(video: str) -> Path:
    """<Training>/vidoes/<video>.MOV (the folder name is spelled that way on disk), case-insensitive."""
    if video in CHALLENGE_VIDEOS:
        raise ValueError(f"{video} is a challenge video")
    add_tools_to_path()
    from trackid import io
    folder = io.TRAIN / "vidoes"
    for p in sorted(folder.glob("*")) if folder.is_dir() else []:
        if p.stem.lower() == video.lower() and p.suffix.lower() in SUFFIXES:
            return p
    raise FileNotFoundError(f"no video {video} in {folder}")


def probe(path: Path) -> tuple[int, int]:
    """(width, height) of the first video stream."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


def frame_count(path: Path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return int(r.stdout.strip().strip(","))


def scaled_size(path: Path, height: int) -> tuple[int, int, float]:
    """(width, height, scale) where scale = source width / scaled width; asserts a 16:9 source."""
    w, h = probe(path)
    if abs(w * 9 - h * 16) > 16:
        raise ValueError(f"{path.name} is {w}x{h}, expected 16:9")
    width = round(w * height / h / 2) * 2
    return width, height, w / width


def _sync_flag() -> str:
    # -fps_mode replaced -vsync in ffmpeg 5.1, and Colab may ship an older build
    text = subprocess.run(["ffmpeg", "-h", "full"], capture_output=True, text=True).stdout
    return "-fps_mode" if "fps_mode" in text else "-vsync"


def _decode(path: Path, select: str, height: int, max_frames: int | None) -> Iterator[np.ndarray]:
    width, height, _ = scaled_size(path, height)
    filters = f"select='{select}',scale={width}:{height}:flags=area"
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "auto", "-i", str(path), "-vf", filters,
           _sync_flag(), "passthrough"]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    size = width * height * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while len(buf := p.stdout.read(size)) == size:
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        p.kill()
        p.wait()


def iter_frames(path: Path, every: int, height: int = 720, limit: int | None = None
                ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (1-based frame number, RGB image) for frames 1, 1+every, ...; `limit` caps the yielded frames."""
    if limit is not None and limit <= 0:
        return
    for n, img in enumerate(_decode(path, f"not(mod(n,{every}))", height, limit)):
        yield n * every + 1, img


def read_frames(path: Path, frame_numbers: list[int], height: int = 720) -> dict[int, np.ndarray]:
    """RGB images for the given 1-based frame numbers. Decodes up to the last one. Missing frames are absent."""
    want = sorted(set(frame_numbers))
    select = "+".join(f"eq(n,{f - 1})" for f in want)
    return {f: img for f, img in zip(want, _decode(path, select, height, len(want)))}
