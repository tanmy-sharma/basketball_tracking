#!/usr/bin/env python
"""Build colab/jnr_all_videos.ipynb: jersey-number evidence for every fine-tuned detection of all 10 videos.

  python colab/build_jnr_notebook.py

Per video: reuse the fine-tuned detections already in Drive (colab/detect_finetuned_videos.ipynb), or run the
detector first when they are missing (the challenge videos), then src/infer/jnr_video.py. Weights are found in Drive
by sha256. Only code and lock files are embedded.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_detect_finetuned_notebook import INSTALL, SETUP, VIDEOS, WEIGHTS_SHA256, make_bundle
from build_src_notebooks import EXCLUDE_PACKAGES, code, fill, lines

OUT = Path(__file__).resolve().parent / "jnr_all_videos.ipynb"
ELIG_SHA256 = "57a986ea4d332ca334a0d90c7b56f7a7e663ee31eed40655aaca810abe0f1f80"  # eligibility__6fd02d77 (e0966686)
VITSTR_SHA256 = "7319ecec70eb2d5b188c4cc161611ee9faf65d666d6182a39e3378a7387734d6"  # vitstr_all__04e8fdd7 (e0966686)
EVERY = 2
MIN_CONF = 0.3

INTRO = f"""# Jersey numbers for every detection, all 10 videos

**Setup:** Runtime > Change runtime type > **A100 GPU**, then Runtime > **Run all**. Keep this tab open.

1. Allow the Google Drive prompt. Videos come from `My Drive/tracking_videos` (any subfolder), **including
   `IMG_0114.MP4` and `IMG_0115.MP4`**. Weights are found in Drive by their hash: the detector from
   `tracking_results/src_train_detector/`, the eligibility head and ViTSTR from `tracking_results/src_train_classifier/`
   (or any copy under `My Drive/tracking_weights`).
2. Per video: the fine-tuned detections are taken from Drive if `detect_finetuned_videos.ipynb` already made them.
   Otherwise (the challenge videos) the detector runs first, on every frame, with preview images.
3. Then every detection with confidence ≥ {MIN_CONF} on every {EVERY}nd frame gets a ViTPose torso crop, a readability
   score (DINOv2 eligibility head) and a score for each of the 9 numbers (fine-tuned ViTSTR), plus a free read.
4. Results go to `My Drive/tracking_results/<video>/<code version>/jnr/<video>__rfdetr-small-ft-471b3fff__720p__jnr.csv.gz`,
   one video at a time. Run all again skips finished videos. A missing video doesn't stop the others.
   At the end one zip with every result is downloaded.

**Using the file:** join on `det_row` (the 0-based line in the detection file). For a track, renormalise `p_*` over
the team's 3 roster numbers, vote over crops with `elig >= 0.5`, and fall back to all crops when a window has none.

**Order:** IMG_0114, IMG_0115 and IMG_0108 first. Nothing here trains: all models were fine-tuned on the 7 training
games only, and the challenge videos are only read for inference."""

DRIVE = '''# Find the three weights and every video before spending any GPU time.
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

WEIGHTS_USED = {}

def fetch_weights(name, digest, folders, pattern, fallback=None):
    """The file with the expected sha256; if none is in Drive, the run's own best.pt (`fallback`), with a warning."""
    local = Path("/content/weights") / name
    if local.exists() and sha256(local) == digest:
        WEIGHTS_USED[name] = {"sha256": digest, "expected": True}
        return local
    places = [p for d in folders if d.is_dir() for p in sorted(d.rglob(pattern))]
    src = next((p for p in places if p.is_file() and p.stat().st_size > 1e6 and sha256(p) == digest), None)
    if src is None and fallback is not None and fallback.is_file():
        src = fallback
        print(f"WARNING: no {name} with sha256 {digest[:8]}... in Drive. Using {fallback} instead.\\n"
              f"  The training notebook may have left an earlier epoch there. For the exact final weights, upload them to\\n"
              f"  My Drive/tracking_weights/ and run this cell again.")
    assert src, f"No {name} with sha256 {digest[:8]}... in {[str(d) for d in folders]}. Put a copy in My Drive/tracking_weights."
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, local)
    got = sha256(local)
    WEIGHTS_USED[name] = {"source": str(src), "sha256": got, "expected": got == digest}
    print(f"{name}: {src} (sha256 {got[:8]}{'' if got == digest else ', NOT the expected ' + digest[:8]})")
    return local

CLS_RUNS = RESULTS_ROOT / "src_train_classifier/e0966686/runs"
WEIGHTS = fetch_weights("detector.pth", WEIGHTS_SHA256, [MY / "tracking_weights", RESULTS_ROOT / "src_train_detector"], "*.pth")
ELIG = fetch_weights("eligibility.pt", "@ELIG@", [MY / "tracking_weights", RESULTS_ROOT / "src_train_classifier"], "*.pt",
                     CLS_RUNS / "eligibility__6fd02d77/best.pt")
VITSTR = fetch_weights("vitstr.pt", "@VITSTR@", [MY / "tracking_weights", RESULTS_ROOT / "src_train_classifier"], "*.pt",
                       CLS_RUNS / "vitstr_all__04e8fdd7/best.pt")

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

UNPACK = '''# Code bundled by colab/build_jnr_notebook.py
import base64, io, tarfile
BUNDLE = "@BUNDLE@"
REPO.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE))) as tar:
    tar.extractall(REPO, filter="data")
print("Unpacked; code version", CODE_VERSION)'''

RUN = '''# One video at a time: detections (reused or made), then jersey numbers; saved to Drive as soon as each finishes.
import json
EVERY, MIN_CONF = @EVERY@, @MIN_CONF@

def save(src, dest):
    """Copy to Drive. The rename means a partial copy never looks complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    shutil.copyfile(src, partial)
    os.replace(partial, dest)

def existing_dets(stem):
    """Fine-tuned detections for this video from any earlier run in Drive, newest first."""
    d = RESULTS_ROOT / stem
    hits = sorted(d.glob(f"*/detections/{stem}__{MODEL}__720p.txt"), key=lambda p: p.stat().st_mtime, reverse=True) if d.is_dir() else []
    return hits[0] if hits else None

def package():
    """One zip of every jnr file, new detection file, preview and log from this code version; to Drive and downloaded."""
    z_path = Path(f"/content/jnr_{CODE_VERSION}.zip")
    with zipfile.ZipFile(z_path, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as z:
        for name in VIDEOS:
            d = RESULTS_ROOT / Path(name).stem / CODE_VERSION
            for f in sorted(d.rglob("*")) if d.is_dir() else []:
                if f.is_file() and f.suffix != ".partial":
                    z.write(f, f"{Path(name).stem}/{f.relative_to(d)}")
    print(f"built {z_path.name} ({z_path.stat().st_size / 1e6:.1f} MB)")
    try:
        save(z_path, RESULTS_ROOT / z_path.name)
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
        dets_name = f"{stem}__{MODEL}__720p.txt"
        jnr_name = f"jnr/{stem}__{MODEL}__720p__jnr.csv.gz"
        if (results / "done/jnr").exists() and (results / jnr_name).exists():
            print(f"=== {stem}: already done, skipping", flush=True)
            continue
        print(f"=== {stem}", flush=True)
        start = time.time()
        local = Path("/content") / name
        log = OUTPUTS / "reports" / stem / "jnr_log.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        dets = OUTPUTS / "detections" / dets_name
        try:
            shutil.copyfile(source, local)
            assert local.stat().st_size == VIDEOS[name], f"{local} has the wrong size after copying"
            found = existing_dets(stem)
            if found is not None:
                print(f"{stem}: detections from {found}", flush=True)
                dets.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(found, dets)
            else:
                print(f"{stem}: no fine-tuned detections in Drive yet, running the detector", flush=True)
                sh([PY, "-m", "src.infer.detect_video", "--video", local, "--weights", WEIGHTS, "--previews", 8], log=log)
                save(dets, results / "detections" / dets_name)
                for p in sorted((OUTPUTS / "previews").glob(f"{stem}__*.jpg")):
                    save(p, results / "previews" / p.name)
            sh([PY, "-m", "src.infer.jnr_video", "--video", local, "--dets", dets, "--eligibility", ELIG, "--vitstr", VITSTR,
                "--every", EVERY, "--min-conf", MIN_CONF], log=log)
            save(OUTPUTS / "jnr" / Path(jnr_name).name, results / jnr_name)
            used = OUTPUTS / "reports" / stem / "weights_used.json"
            used.write_text(json.dumps(WEIGHTS_USED, indent=1))
            save(used, results / "reports/weights_used.json")
            save(log, results / "reports/jnr_log.txt")
            (results / "done").mkdir(parents=True, exist_ok=True)
            (results / "done/jnr").touch()
        except Exception as e:
            failures[name] = str(e)
            print(f"{stem} FAILED: {e}", flush=True)
            if log.exists():
                save(log, results / "reports/jnr_log.txt")
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
    kw = dict(VIDEOS=repr(VIDEOS), SHA=WEIGHTS_SHA256, PREVIEWS="8", CODE_VERSION=version, BUNDLE=bundle,
              EXCLUDE=repr(EXCLUDE_PACKAGES), ELIG=ELIG_SHA256, VITSTR=VITSTR_SHA256, EVERY=str(EVERY),
              MIN_CONF=str(MIN_CONF))
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": lines(INTRO)},
                  *(code(fill(c, **kw)) for c in (SETUP, DRIVE, UNPACK, INSTALL, RUN))]}


def main() -> None:
    OUT.write_text(json.dumps(build(), indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB), code version {make_bundle()[1]}")


if __name__ == "__main__":
    main()
