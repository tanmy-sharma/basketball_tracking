#!/usr/bin/env python
"""Build colab/jnr_test_videos.ipynb: fine-tuned detections and jersey-number evidence for the two challenge videos.

  python colab/build_jnr_test_notebook.py

Same cells as colab/jnr_all_videos.ipynb (build_jnr_notebook.py), with three differences:
- videos come from My Drive/test_videos;
- only IMG_0114.MP4 and IMG_0115.MP4 are processed;
- the final zip is jnr_test_<code version>.zip, so it never overwrites the training-video zip.

The detector runs on every frame, since no fine-tuned detections exist for these videos yet, then src/infer/jnr_video.py.
Nothing trains: the challenge videos are only read for inference. Colour labels, tracking, JNR stitching and the submission
run on the Mac afterwards (docs/19_challenge_run.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import build_jnr_notebook as jnr
from build_detect_finetuned_notebook import INSTALL, SETUP, VIDEOS, WEIGHTS_SHA256, make_bundle
from build_src_notebooks import EXCLUDE_PACKAGES, code, fill, lines

OUT = Path(__file__).resolve().parent / "jnr_test_videos.ipynb"
TEST_VIDEOS = {k: v for k, v in VIDEOS.items() if k in ("IMG_0114.MP4", "IMG_0115.MP4")}
FOLDER = "test_videos"

INTRO = f"""# Challenge videos: detections and jersey numbers

**Setup:** Runtime > Change runtime type > **A100 GPU**, then Runtime > **Run all**. Keep this tab open.

1. Allow the Google Drive prompt. The videos come from `My Drive/{FOLDER}` (any subfolder): `IMG_0114.MP4`
   ({TEST_VIDEOS['IMG_0114.MP4']:,} bytes) and `IMG_0115.MP4` ({TEST_VIDEOS['IMG_0115.MP4']:,} bytes). A file of
   another size is skipped as an incomplete upload.
2. The weights are found in Drive by their hash, as in `jnr_all_videos.ipynb`:
   - the detector from `tracking_results/src_train_detector/`;
   - the eligibility head and ViTSTR from `tracking_results/src_train_classifier/`, or any copy under
     `My Drive/tracking_weights`.
   They are pinned by sha256 to the checkpoints the training-game results were made with (readability head
   `{jnr.ELIG_SHA256[:8]}`). If one is missing, the notebook stops rather than use other weights.
3. Per video:
   - the fine-tuned RF-DETR Small runs on every frame, with 8 preview images;
   - every detection with confidence ≥ {jnr.MIN_CONF} on every {jnr.EVERY}nd frame gets a ViTPose torso crop, a
     readability score and a score for each of the 9 numbers.
4. Results go to `My Drive/tracking_results/<video>/<code version>/`, under `detections/` and `jnr/`, one video at a
   time. Run all again skips finished videos. At the end `jnr_test_<code version>.zip` is downloaded and kept in
   `My Drive/tracking_results/`.

**Rules:** nothing here trains. All models were fine-tuned on the training games only, and the challenge videos are
only read for inference. Bring the zip back to the Mac for colour labels, tracking, stitching and the submission."""


def build() -> dict:
    bundle, version = make_bundle()
    drive = jnr.DRIVE.replace('MY / "tracking_videos"', f'MY / "{FOLDER}"')
    run = jnr.RUN.replace('f"/content/jnr_{CODE_VERSION}.zip"', 'f"/content/jnr_test_{CODE_VERSION}.zip"')
    assert drive != jnr.DRIVE and run != jnr.RUN, "build_jnr_notebook.py changed: update the replacements here"
    kw = dict(VIDEOS=repr(TEST_VIDEOS), SHA=WEIGHTS_SHA256, PREVIEWS="8", CODE_VERSION=version, BUNDLE=bundle,
              EXCLUDE=repr(EXCLUDE_PACKAGES), ELIG=jnr.ELIG_SHA256, VITSTR=jnr.VITSTR_SHA256, EVERY=str(jnr.EVERY),
              MIN_CONF=str(jnr.MIN_CONF))
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": lines(INTRO)},
                  *(code(fill(c, **kw)) for c in (SETUP, drive, jnr.UNPACK, INSTALL, run))]}


def main() -> None:
    OUT.write_text(json.dumps(build(), indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB), code version {make_bundle()[1]}")


if __name__ == "__main__":
    main()
