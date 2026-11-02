#!/usr/bin/env python
"""Build colab/detect_finetuned_videos.ipynb: the fine-tuned RF-DETR Small on every frame of all 10 videos.

  python colab/build_detect_finetuned_notebook.py

Same layout as colab/build_detect_all_notebook.py (one video at a time, results per video in Drive), but with the
weights from colab/train_detector.ipynb, found in Drive by their sha256. Only code and lock files are embedded.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

from build_src_notebooks import EXCLUDE_PACKAGES, ROOT, bundle_files, code, fill, lines

OUT = Path(__file__).resolve().parent / "detect_finetuned_videos.ipynb"
# best.pth of run rfdetr_small__2a3fda10 (batch 32, 30 epochs, val mAP 0.8185), code version e6a5aded
WEIGHTS_SHA256 = "471b3fff3f77d320f9c84cdc1d8978d8a1b6162d6813915c1d8e0c700955f80a"
WEIGHTS_HINT = "tracking_results/src_train_detector/e6a5aded/runs/rfdetr_small__2a3fda10/best.pth"
# Challenge videos and the held-out game first: they are the ones whose results mean something.
# The other 7 games are training games for this detector, so their detections are in-sample.
VIDEOS = {"IMG_0114.MP4": 150358487, "IMG_0115.MP4": 154342609, "IMG_0108.MOV": 1454621442,
          "IMG_0104.MOV": 1421638793, "IMG_0105.MOV": 1590207934, "IMG_0106.MOV": 1354953016,
          "IMG_0107.MOV": 1433298468, "IMG_0109.MOV": 1293838444, "IMG_0110.MOV": 1287035963,
          "IMG_0111.MOV": 1469316500}
PREVIEWS = 8


def infer_bundle_files() -> list[str]:
    return sorted(set(bundle_files()) | {p.relative_to(ROOT).as_posix() for p in (ROOT / "src/infer").glob("*.py")})


def make_bundle() -> tuple[str, str]:
    names = infer_bundle_files()
    h = hashlib.sha256()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for n in names:
            h.update(n.encode() + b"\0" + (ROOT / n).read_bytes() + b"\0")
            tar.add(ROOT / n, arcname=n)
    return base64.b64encode(buf.getvalue()).decode(), h.hexdigest()[:8]


INTRO = f"""# Fine-tuned RF-DETR Small detection on all 10 videos

**Setup:** Runtime > Change runtime type > **A100 GPU**, then Runtime > **Run all**. Keep this tab open.

1. Allow the Google Drive prompt. Videos are read from `My Drive/tracking_videos` (any subfolder), including the
   challenge videos `IMG_0114.MP4` and `IMG_0115.MP4`. The weights are found in Drive by their hash, normally at
   `My Drive/{WEIGHTS_HINT}`. A copy anywhere under `My Drive/tracking_weights` also works.
2. Each video is copied to the runtime's local disk, run on every frame, saved to Drive, and removed from local disk.
3. Results go to `My Drive/tracking_results/<video>/<code version>/`: `detections/<video>__rfdetr-small-ft-471b3fff__720p.txt`
   (same format as the RF-DETR Large detections, so the trackers read it unchanged), {PREVIEWS} preview frames with the
   boxes drawn, `reports/detector_log.txt` and `done/detection_ft`. If the run stops, choose Run all again: finished
   videos are skipped.
4. A missing video doesn't stop the others. At the end one zip with every detection file and preview is downloaded.

**Order:** IMG_0114, IMG_0115 and IMG_0108 first. IMG_0108 was held out of training. The other 7 games trained this
detector, so their detections are in-sample: fine for tracker plumbing, not for scoring the detector.

**Look at the challenge previews.** The detector was fine-tuned on the end-line camera only. On the challenge camera,
check that it still boxes every player and does not box referees or people off court.

**Model:** RF-DETR Small (Apache 2.0, COCO-pretrained by Roboflow), fine-tuned on the 7 training games only. No
weights are updated here, and nothing from the challenge videos is used for training. Disclose it in the paper."""

SETUP = '''import os, shutil, subprocess, sys, time, hashlib, zipfile
from pathlib import Path

VIDEOS = @VIDEOS@
WEIGHTS_SHA256 = "@SHA@"
PREVIEWS = @PREVIEWS@
MODEL = "rfdetr-small-ft-" + WEIGHTS_SHA256[:8]
CODE_VERSION = "@CODE_VERSION@"
REPO = Path("/content/tracking_challenge")
PY = REPO / ".venv/bin/python"
OUTPUTS = REPO / "outputs"

def sh(cmd, cwd=REPO, log=None):
    """Run a command, stream its output, optionally keep a copy, and raise if it fails."""
    print("$", " ".join(map(str, cmd)), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "agg"}  # Colab's inline backend does not exist in our env
    p = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
    print("WARNING: this is not an A100. The run still works but will be slower.")'''

DRIVE = '''# Find the weights and every video before spending any GPU time, and say plainly what is missing.
from google.colab import drive, files

drive.mount("/content/drive")
MY = Path("/content/drive/MyDrive")
DRIVE_DIR = MY / "tracking_videos"
RESULTS_ROOT = MY / "tracking_results"
assert DRIVE_DIR.is_dir(), f"{DRIVE_DIR} not found. Check the folder name and that it is in this account's My Drive."

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

WEIGHTS = Path("/content/best.pth")
if not (WEIGHTS.exists() and sha256(WEIGHTS) == WEIGHTS_SHA256):
    places = [MY / "@HINT@", *sorted((RESULTS_ROOT / "src_train_detector").rglob("best.pth")),
              *sorted((MY / "tracking_weights").rglob("*.pth"))]
    src = next((p for p in places if p.is_file() and p.stat().st_size > 1e6 and sha256(p) == WEIGHTS_SHA256), None)
    assert src, (f"No weights with sha256 {WEIGHTS_SHA256[:8]}... in Drive. Run colab/train_detector.ipynb first, or put "
                 f"best.pth from weights_e6a5aded.zip into My Drive/tracking_weights.")
    shutil.copyfile(src, WEIGHTS)
    assert sha256(WEIGHTS) == WEIGHTS_SHA256, "weights changed while copying"
    print("Weights:", src)

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
assert SOURCES, f"No usable videos under {DRIVE_DIR}. Files there: {sorted(p.name for p in on_drive)}"'''

UNPACK = '''# Code bundled by colab/build_detect_finetuned_notebook.py
import base64, io, tarfile
BUNDLE = "@BUNDLE@"
REPO.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE))) as tar:
    tar.extractall(REPO, filter="data")
print("Unpacked; code version", CODE_VERSION)'''

INSTALL = '''# Same locked versions as the local repo, with the rfdetr group, in a separate env so Colab's packages don't interfere.
EXCLUDE = @EXCLUDE@
sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
sh([sys.executable, "-m", "uv", "sync", "--frozen", "--no-dev", "--python", "3.12", "--group", "rfdetr",
    *[arg for package in EXCLUDE for arg in ("--no-install-package", package)]])
sh([PY, "-c", "import torch, rfdetr, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); "
    "assert torch.cuda.is_available(), 'torch cannot see the GPU'"])'''

RUN = '''# One video at a time, saved to Drive as soon as it finishes. A failure is recorded and the next video still runs.
def save(src, dest):
    """Copy to Drive. The rename means a partial copy never looks complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    shutil.copyfile(src, partial)
    os.replace(partial, dest)

def package():
    """One zip of every finished detection file, preview and log, copied to Drive and downloaded."""
    z_path = Path(f"/content/finetuned_detections_{CODE_VERSION}.zip")
    with zipfile.ZipFile(z_path, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as z:
        for name in VIDEOS:
            d = RESULTS_ROOT / Path(name).stem / CODE_VERSION
            for f in sorted(d.rglob("*")) if d.is_dir() else []:
                if f.is_file() and f.suffix != ".partial":
                    z.write(f, f"{Path(name).stem}/{f.relative_to(d)}")
    print(f"built {z_path.name} ({z_path.stat().st_size / 1e6:.1f} MB)")
    try:
        save(z_path, RESULTS_ROOT / f"finetuned_detections_{CODE_VERSION}.zip")
        drive.flush_and_unmount()  # Drive uploads in the background; make sure everything has reached it
        print("Results are in Drive:", RESULTS_ROOT)
    except Exception as e:
        print("Copy to Drive failed:", repr(e), "- the zip is at", z_path)
    try:
        files.download(str(z_path))
    except Exception as e:
        print(f"Download failed ({e!r}). Get it from Drive: {RESULTS_ROOT / z_path.name}")

failures = {}
try:
    for name, source in SOURCES.items():
        stem = Path(name).stem
        results = RESULTS_ROOT / stem / CODE_VERSION
        output = f"detections/{stem}__{MODEL}__720p.txt"
        marker = results / "done/detection_ft"
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
            sh([PY, "-m", "src.infer.detect_video", "--video", local, "--weights", WEIGHTS, "--previews", PREVIEWS], log=log)
            save(OUTPUTS / output, results / output)
            for p in sorted((OUTPUTS / "previews").glob(f"{stem}__*.jpg")):
                save(p, results / "previews" / p.name)
                p.unlink()
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
    package()

unfinished = {**problems, **failures}
if unfinished:
    raise RuntimeError("Not finished, everything else is in Drive. Fix these, then choose Run all to process only what is missing:\\n"
                       + "\\n".join(f"  {name}: {why}" for name, why in unfinished.items()))
print(f"All {len(VIDEOS)} videos are done.")'''


def build() -> dict:
    bundle, version = make_bundle()
    kw = dict(VIDEOS=repr(VIDEOS), SHA=WEIGHTS_SHA256, PREVIEWS=str(PREVIEWS), CODE_VERSION=version, BUNDLE=bundle,
              EXCLUDE=repr(EXCLUDE_PACKAGES), HINT=WEIGHTS_HINT)
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": lines(INTRO)},
                  *(code(fill(c, **kw)) for c in (SETUP, DRIVE, UNPACK, INSTALL, RUN))]}


def main() -> None:
    nb = build()
    OUT.write_text(json.dumps(nb, indent=1))
    version = make_bundle()[1]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB), code version {version}")


if __name__ == "__main__":
    main()
