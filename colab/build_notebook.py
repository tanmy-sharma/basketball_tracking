#!/usr/bin/env python
"""Build colab/detect_track_IMG_0106.ipynb, embedding the repo's code so Colab needs no git access.

  python colab/build_notebook.py

Only code and lock files are embedded. The notebook gets the video from the user's Drive and annotations from the official one.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["pyproject.toml", "uv.lock", "tools/run_detector.py", "tools/eval_detections.py", "tools/run_trackers.py",
         "tools/trackid/__init__.py", "tools/trackid/io.py", "tools/trackid/metric.py", "tools/trackid/eval_mot.py",
         "tools/trackid/stratified.py"]

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for name in FILES:
        tar.add(ROOT / name, arcname=name)
bundle = base64.b64encode(buf.getvalue()).decode()
# Results are kept per code version, so a rebuilt notebook never resumes from results made by different code
code_version = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in FILES)).hexdigest()[:8]

INTRO = """# RF-DETR Large + Roboflow trackers on IMG_0106

**Setup:** Runtime > Change runtime type > **A100 GPU**, then Runtime > **Run all**. Keep this tab open.

1. Allow the Google Drive prompt. The video is read from `My Drive/tracking_videos`, including any subfolder, and
   copied to the runtime's local disk.
2. Everything else runs unattended: detection on all 11,272 frames, detection scoring against the ground truth,
   5 trackers with 2 settings each, and the miss report.
3. Results are saved to `My Drive/tracking_results/IMG_0106/<code version>/` after every step, and after each tracker run.
   Nothing downloads to your computer.
   If the run stops, choose Run all again: finished steps are restored from Drive and skipped.
   Detection saves only when it finishes, so if it stops partway through the video it starts again.

**Model:** `rfdetr-large`: the largest Apache 2.0 RF-DETR, 704 x 704 input, COCO-pretrained by Roboflow.
It is not fine-tuned on any contest data, and no weights are updated here. Disclose it in the paper."""

SETUP = """import os, subprocess, sys
from pathlib import Path

VIDEO = "IMG_0106"
VIDEO_BYTES = 1354953016
MODELS = ["rfdetr-large"]
CODE_VERSION = "@CODE_VERSION@"
REPO = Path("/content/tracking_challenge")
PY = REPO / ".venv/bin/python"
OUTPUTS = REPO / "outputs"
REPORTS = OUTPUTS / "reports"

def sh(cmd, cwd=REPO, log=None):
    \"\"\"Run a command, stream its output, optionally keep a copy, and stop the notebook if it fails.\"\"\"
    print("$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(list(map(str, cmd)), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = open(log, "a") if log else None
    for line in p.stdout:
        print(line, end="", flush=True)
        if out:
            out.write(line)
    if out:
        out.close()
    if p.wait():
        raise RuntimeError(f"command failed with exit code {p.returncode}: {cmd}")

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
assert gpu.returncode == 0, "No GPU. Set Runtime > Change runtime type > A100 GPU, then run again."
print(gpu.stdout.strip())
if "A100" not in gpu.stdout:
    print("WARNING: this is not an A100. The run still works but will be slower.")"""

VIDEO_FROM_DRIVE = """# Copy the video from Drive to local disk, which decodes faster and avoids Drive read errors during the long run.
import shutil
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR = Path("/content/drive/MyDrive/tracking_videos")
assert DRIVE_DIR.is_dir(), f"{DRIVE_DIR} not found. Check the folder name and that it is in this account's My Drive."
RESULTS = Path(f"/content/drive/MyDrive/tracking_results/{VIDEO}/{CODE_VERSION}")
RESULTS.mkdir(parents=True, exist_ok=True)
print("Results will be saved to", RESULTS)

found = [p for p in DRIVE_DIR.rglob("*") if p.is_file() and p.name.lower() == f"{VIDEO}.mov".lower()]
listing = sorted(str(p.relative_to(DRIVE_DIR)) for p in DRIVE_DIR.rglob("*") if p.is_file())
assert found, f"{VIDEO}.MOV not found under {DRIVE_DIR}. Files there: {listing}"
matching = [p for p in found if p.stat().st_size == VIDEO_BYTES]
assert matching, f"No {VIDEO}.MOV of {VIDEO_BYTES} bytes. Found: {[(str(p), p.stat().st_size) for p in found]}. The upload may be incomplete."

target = Path(f"/content/{VIDEO}.MOV")
if not target.exists() or target.stat().st_size != VIDEO_BYTES:
    print(f"Copying {matching[0]} ...", flush=True)
    shutil.copyfile(matching[0], target)

size = target.stat().st_size
assert size == VIDEO_BYTES, f"{target} is {size} bytes after copying, expected {VIDEO_BYTES}."
print(f"Video OK: {target} ({size / 1e9:.2f} GB)")"""

UNPACK = f"""# Code bundled by colab/build_notebook.py
import base64, io, tarfile
BUNDLE = "{bundle}"
REPO.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE))) as tar:
    tar.extractall(REPO, filter="data")
print("Unpacked:", *sorted(str(p.relative_to(REPO)) for p in REPO.rglob("*.py")), sep="\\n  ")"""

INSTALL = """# Same locked versions as the local repo, in a separate env so Colab's own packages don't interfere
sh([sys.executable, "-m", "pip", "install", "-q", "uv", "gdown"])
sh([sys.executable, "-m", "uv", "sync", "--frozen", "--no-dev", "--python", "3.12"])  # the version the tools were tested on
sh([PY, "-c", "import torch, trackers, cv2; from importlib.metadata import version; print('torch', torch.__version__, 'trackers', version('trackers'), 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available(), 'torch cannot see the GPU'"])"""

DATA = """# Ground truth from the official challenge Drive, used for scoring only. The video goes where the tools expect it.
import gdown

TRAIN = REPO / "data/Training"
ANNOTATIONS = {
    "1o3uUDaj2MEAftF5W2B2zs2LUhaeBLgNL": (TRAIN / f"full player annotations/Image-space bounding boxes/{VIDEO}.txt", 67483),
    "1y_u3yAiZbMbmv9qDU6l9OtWNp_oX9uz7": (TRAIN / f"full player annotations/court position/{VIDEO}.txt", 67483),
    "1CDx1jVsYHs9CdvF3OC63gRk31WRvXEul": (TRAIN / "court_registration.csv", None),
    "1OL8kBVDOE3n6FK3swkc9Ic2DIzq9dzhH": (TRAIN / "games_roster.csv", None),
}
for file_id, (path, rows) in ANNOTATIONS.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() and not gdown.download(id=file_id, output=str(path), quiet=True):
        raise RuntimeError(f"Could not download {path.name} from Drive (https://drive.google.com/file/d/{file_id})")
    assert rows is None or sum(1 for _ in open(path)) == rows, f"{path} does not have {rows} rows"

video_path = TRAIN / "vidoes" / f"{VIDEO}.MOV"
video_path.parent.mkdir(parents=True, exist_ok=True)
if not video_path.exists():
    video_path.symlink_to(target)
print("Data ready")"""

RUN = """# Detect, score, track and report, saving to Drive after each step. Finished steps are restored and skipped.
import os, shutil, time
from google.colab import drive

def save(*items):
    \"\"\"Copy finished files to Drive. The rename means a partial copy never looks complete.\"\"\"
    for item in items:
        path = OUTPUTS / item
        for f in sorted(path.rglob("*")) if path.is_dir() else [path] if path.exists() else []:
            if f.is_file() and f.suffix != ".tmp":
                dest = RESULTS / f.relative_to(OUTPUTS)
                dest.parent.mkdir(parents=True, exist_ok=True)
                partial = dest.with_name(dest.name + ".partial")
                shutil.copyfile(f, partial)
                os.replace(partial, dest)

def restore():
    for f in RESULTS.rglob("*"):
        if f.is_file() and f.suffix != ".partial":
            dest = OUTPUTS / f.relative_to(RESULTS)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)

failures = []

def step(name, cmd, log, *items):
    \"\"\"Run one step unless Drive says it is done. Whatever exists is saved even if the step fails.\"\"\"
    marker = OUTPUTS / "done" / name
    if marker.exists():
        print(f"=== {name}: already done, skipping", flush=True)
        return True
    print(f"=== {name}", flush=True)
    start = time.time()
    try:
        sh(cmd, cwd=REPO / "tools", log=REPORTS / log)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return True
    except Exception as e:
        failures.append(name)
        print(f"{name} FAILED: {e}", flush=True)
        return False
    finally:
        print(f"{name}: {(time.time() - start) / 60:.1f} min", flush=True)
        save(*items, f"reports/{log}", f"done/{name}")

REPORTS.mkdir(parents=True, exist_ok=True)
restore()
dets = [OUTPUTS / f"detections/{VIDEO}__{m}__720p.txt" for m in MODELS]
try:
    if step("detection", [PY, "run_detector.py", video_path, "--models", *MODELS], "detector_log.txt", "detections"):
        step("detection_eval", [PY, "eval_detections.py", VIDEO, *dets], "detection_eval.txt")
        step("tracking", [PY, "run_trackers.py", VIDEO, *dets, "--sync-dir", RESULTS], "tracker_log.txt", "tracks", "reports")
finally:
    # Drive uploads in the background; this makes sure everything has reached it before the runtime can go away
    drive.flush_and_unmount()
    print("Results are in Drive:", RESULTS)

if failures:
    raise RuntimeError(f"Failed steps: {failures}. Finished work is in Drive. Choose Run all to retry only what is missing.")"""


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "machine_shape": "hm"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [{"cell_type": "markdown", "metadata": {}, "source": lines(INTRO)},
              *map(code, [SETUP.replace("@CODE_VERSION@", code_version), VIDEO_FROM_DRIVE, UNPACK, INSTALL, DATA, RUN])],
}
out = Path(__file__).with_name("detect_track_IMG_0106.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
