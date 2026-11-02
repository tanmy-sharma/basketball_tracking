#!/usr/bin/env python
"""Build colab/detect_all_videos.ipynb: RF-DETR Large detection on every video in the Drive folder, and nothing else.

  python colab/build_detect_all_notebook.py

Only code and lock files are embedded. The notebook gets the videos from the user's Drive.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["pyproject.toml", "uv.lock", "tools/run_detector.py"]
# Pulled in by trackers[detection], and inference_models only imports them if asked for those models. See docs/10 section 7.
EXCLUDE = ["python-doctr", "rf-groundingdino"]
# Only what is still missing. The other 7 training games finished under code version 8f5defcb, and the detector itself is
# unchanged since then: only the end-of-video frame check was relaxed, which had rejected IMG_0104.
VIDEOS = {"IMG_0114.MP4": 150358487, "IMG_0115.MP4": 154342609, "IMG_0104.MOV": 1421638793}

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for name in FILES:
        tar.add(ROOT / name, arcname=name)
bundle = base64.b64encode(buf.getvalue()).decode()
# Results are kept per code version, so a rebuilt notebook never resumes from results made by different code
code_version = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in FILES)).hexdigest()[:8]

INTRO = """# RF-DETR Large detection on all videos

**Setup:** Runtime > Change runtime type > **A100 GPU**, then Runtime > **Run all**. Keep this tab open.

1. Allow the Google Drive prompt. Videos are read from `My Drive/tracking_videos`, including any subfolder.
2. Each video is copied to the runtime's local disk, detected, saved to Drive, and removed from local disk.
   Detection is the only thing this notebook does.
3. Results are saved to `My Drive/tracking_results/<video>/<code version>/`, one video at a time:
   `detections/<video>__rfdetr-large__720p.txt`, `reports/detector_log.txt` and `done/detection`.
   If the run stops, choose Run all again: finished videos are skipped. A video that was partway through starts again.
4. A missing or incomplete video doesn't stop the others. The last cell lists any that need attention.

**Videos:** only the three still missing, IMG_0114, IMG_0115 and IMG_0104. The other training games are already done.
The first cell lists each video it can't find or whose size is wrong. Check it if the challenge videos are skipped again.

**Model:** `rfdetr-large`: the largest Apache 2.0 RF-DETR, 704 x 704 input, COCO-pretrained by Roboflow.
It is not fine-tuned on any contest data, and no weights are updated here. Disclose it in the paper."""

SETUP = """import os, shutil, subprocess, sys, time
from pathlib import Path

VIDEOS = @VIDEOS@
MODEL = "rfdetr-large"
CODE_VERSION = "@CODE_VERSION@"
REPO = Path("/content/tracking_challenge")
PY = REPO / ".venv/bin/python"
OUTPUTS = REPO / "outputs"

def sh(cmd, cwd=REPO, log=None):
    \"\"\"Run a command, stream its output, optionally keep a copy, and raise if it fails.\"\"\"
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

DRIVE = """# Find every video before spending any GPU time, and say plainly what is missing or incomplete.
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR = Path("/content/drive/MyDrive/tracking_videos")
RESULTS_ROOT = Path("/content/drive/MyDrive/tracking_results")
assert DRIVE_DIR.is_dir(), f"{DRIVE_DIR} not found. Check the folder name and that it is in this account's My Drive."

on_drive = [p for p in DRIVE_DIR.rglob("*") if p.is_file()]
SOURCES, problems = {}, {}
for name, size in VIDEOS.items():
    found = [p for p in on_drive if p.name.lower() == name.lower()]
    complete = [p for p in found if p.stat().st_size == size]
    if complete:
        SOURCES[name] = complete[0]
    elif found:
        problems[name] = f"expected {size} bytes, found {[p.stat().st_size for p in found]}. The upload may be incomplete."
    else:
        problems[name] = "not in the folder."
print(f"Ready: {len(SOURCES)} of {len(VIDEOS)} videos")
for name, why in problems.items():
    print(f"  SKIPPING {name}: {why}")
assert SOURCES, f"No usable videos under {DRIVE_DIR}. Files there: {sorted(p.name for p in on_drive)}\""""

UNPACK = """# Code bundled by colab/build_detect_all_notebook.py
import base64, io, tarfile
BUNDLE = "@BUNDLE@"
REPO.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE))) as tar:
    tar.extractall(REPO, filter="data")
print("Unpacked:", *sorted(str(p.relative_to(REPO)) for p in REPO.rglob("*.py")), sep="\\n  ")"""

INSTALL = """# Same locked versions as the local repo, in a separate env so Colab's own packages don't interfere
EXCLUDE = @EXCLUDE@
sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
sh([sys.executable, "-m", "uv", "sync", "--frozen", "--no-dev", "--python", "3.12",
    *[arg for package in EXCLUDE for arg in ("--no-install-package", package)]])
sh([PY, "-c", "import importlib.util as u, torch; from inference_models import AutoModel; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available(), 'torch cannot see the GPU'; bad = [m for m in ('doctr', 'groundingdino') if u.find_spec(m)]; assert not bad, f'excluded packages are installed: {bad}'"])"""

RUN = """# One video at a time, saved to Drive as soon as it finishes. A failure is recorded and the next video still runs.
def save(src, dest):
    \"\"\"Copy to Drive. The rename means a partial copy never looks complete.\"\"\"
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    shutil.copyfile(src, partial)
    os.replace(partial, dest)

failures = {}
try:
    for name, source in SOURCES.items():
        stem = Path(name).stem
        results = RESULTS_ROOT / stem / CODE_VERSION
        output = f"detections/{stem}__{MODEL}__720p.txt"
        marker = results / "done/detection"
        if marker.exists() and (results / output).exists():
            print(f"=== {stem}: already done, skipping", flush=True)
            continue
        print(f"=== {stem}", flush=True)
        start = time.time()
        local = Path("/content") / name
        log = OUTPUTS / "reports" / stem / "detector_log.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(source, local)
            assert local.stat().st_size == VIDEOS[name], f"{local} has the wrong size after copying"
            sh([PY, "run_detector.py", local, "--models", MODEL], cwd=REPO / "tools", log=log)
            save(OUTPUTS / output, results / output)
            save(log, results / "reports/detector_log.txt")
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            (OUTPUTS / output).unlink()
        except Exception as e:
            failures[name] = str(e)
            print(f"{stem} FAILED: {e}", flush=True)
            if log.exists():
                save(log, results / "reports/detector_log.txt")
        finally:
            local.unlink(missing_ok=True)
            print(f"{stem}: {(time.time() - start) / 60:.1f} min", flush=True)
finally:
    # Drive uploads in the background; this makes sure everything has reached it before the runtime can go away
    drive.flush_and_unmount()
    print("Results are in Drive:", RESULTS_ROOT)

unfinished = {**problems, **failures}
if unfinished:
    raise RuntimeError("Not finished, everything else is in Drive. Fix these, then choose Run all to process only what is missing:\\n"
                       + "\\n".join(f"  {name}: {why}" for name, why in unfinished.items()))
print(f"All {len(VIDEOS)} videos are done.")"""


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


fills = {"@VIDEOS@": repr(VIDEOS), "@CODE_VERSION@": code_version, "@BUNDLE@": bundle, "@EXCLUDE@": repr(EXCLUDE)}


def fill(text: str) -> str:
    for key, value in fills.items():
        text = text.replace(key, value)
    return text


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
              *(code(fill(cell)) for cell in [SETUP, DRIVE, UNPACK, INSTALL, RUN])],
}
out = Path(__file__).with_name("detect_all_videos.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB), code version {code_version}")
