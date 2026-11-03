# ABC 2027 TrackID3x3 Challenge: inference

Team- and jersey-aware tracking of the 6 players in a 3x3 basketball video, written as court positions per frame in the
challenge format (`frame,x_cm,y_cm,team,jersey`). This README covers reproducing the submitted results from a video.
Nothing here trains; the model weights are fixed (see Weights).

Pipeline (each stage is an existing module). `run_pipeline.py` passes the settings in `configs/submission.yaml`
explicitly; every other setting is the module's source default at the tagged commit, so the configuration is that YAML
plus the code version:

| Stage | Module | Hardware | Output |
|---|---|---|---|
| 1. Detection: fine-tuned RF-DETR Small, every frame at 720p | `src/infer/detect_video.py` | GPU (CPU works, slowly) | `detections/` |
| 2. Jersey evidence: ViTPose torso crop, DINOv2 eligibility head, fine-tuned ViTSTR, every 2nd frame | `src/infer/jnr_video.py` | GPU (CPU works, slowly) | `jnr/` |
| 3. Team colour labels per box | `src/team/run.py` | CPU | `labels/` |
| 4. Team-aware UCMC tracker on the court plane (setting `ft_guard_lock30`) | `src/tracker/run_ucmc.py` | CPU | `tracks/` |
| 5. Jersey stitching, export, depth smoothing (σ = 4 frames) | `src/tracker/jnr_tracks.py` | CPU | `submission/` |
| 6. Format check and upload ZIP | `tools/validate_submission.py` | CPU | `submission.zip` |

## Setup

Requirements: Linux or macOS, Python 3.12, [uv](https://docs.astral.sh/uv/), and `ffmpeg`/`ffprobe` on the PATH. For the
GPU stages, an NVIDIA GPU with about 12 GB of memory and a driver recent enough for the CUDA build of torch in `uv.lock`.

```
git clone <repo> && cd tracking_challenge
uv sync --frozen --no-dev --python 3.12 --group rfdetr \
    --no-install-package python-doctr --no-install-package rf-groundingdino
```

This is the install used on Colab for the submitted run. Afterwards, always run with `uv run --no-sync` (or
`.venv/bin/python`): a plain `uv run` re-syncs the environment and would add the dev group and the two excluded packages. `--frozen` installs exactly the versions in `uv.lock`. The two
excluded packages come in only through `trackers[detection]`; nothing in this repo imports them. ViTPose
(`usyd-community/vitpose-base-simple`, revision pinned in `src/core/config.py`) and the DINOv2 / ViT backbone configs are
downloaded from the Hugging Face Hub on first use.

### Weights

Put the three files in `weights/` under these names. `run_pipeline.py` checks each sha256 and stops on a mismatch.

| File | sha256 | What |
|---|---|---|
| `detector_471b3fff.pth` | `471b3fff3f77d320f9c84cdc1d8978d8a1b6162d6813915c1d8e0c700955f80a` | RF-DETR Small fine-tuned on the 7 training games (run `rfdetr_small__2a3fda10`) |
| `eligibility_4d403c88.pt` | `4d403c88ad87819e86a4e7b3a79dfdaeae09369d8342ec93725df6dd74dd43f2` | DINOv2 eligibility head (run `eligibility__6fd02d77`, the checkpoint kept in Drive) |
| `vitstr_7319ecec.pt` | `7319ecec70eb2d5b188c4cc161611ee9faf65d666d6182a39e3378a7387734d6` | ViTSTR fine-tuned on all jersey crops (run `vitstr_all__04e8fdd7`) |

Download: TODO (fixed-version links). The eligibility file is not the epoch-13 `57a986ea` in the weights zip of
that run. The submission used `4d403c88`, the earlier checkpoint of the same run kept in Drive, which every scored
training-game result also used (docs/18). The Colab records in `results/challenge/colab_reports/*/weights_used.json`
show `"expected": false` for it because the notebook pinned `57a986ea` at the time; the hash recorded there is
`4d403c88`. Those records are kept as written.

### Data

`data/Final challenge/games_roster.csv` and `court_registration.csv` (the organisers' files) are in the repo. The videos
are not: pass their paths with `--video-path`.

## Run

```
uv run --no-sync python run_pipeline.py --config configs/submission.yaml \
    --video-path /path/to/IMG_0114.MP4 /path/to/IMG_0115.MP4 --output-dir runs/fresh
```

With both challenge videos, the run ends with `runs/fresh/submission/IMG_0114.txt`, `IMG_0115.txt`, the validated upload
ZIP `runs/fresh/submission.zip`, `logs/` and `manifest.json`. The manifest records the environment (Python, platform,
commit), the weights loaded and hash-checked in this run, the origin of every stage output (`computed`,
`copied_from_cache` or `reused_from_previous_run`, with the cache's own provenance), the sha256 of every stage output, whether each is byte-identical to the submitted run
(`same` / `DIFFERENT`), and `ti_hota_vs_submitted`. A finished run exits with 0 even if outputs differ; add `--verify` for
a strict replay that exits with 1 unless every output is byte-identical.

The output folder is guarded:
- it must not overlap `results/`, `data/`, `outputs/`, the cache, the weights or the videos' folders, and inside the
  repo it must be under `runs/`;
- rerunning into the same folder resumes the stages that finished, but only if the videos, config, cache, device,
  `--limit` and commit are the same (`run.json`). Otherwise, or if the folder is non-empty without `run.json`, the run
  is refused before anything is written;
- a finished stage is reused only if its inputs and outputs still have the hashes recorded in `done/<stage>.json`. A
  missing or changed file makes the stage run again.

Options:
- `--cache-dir results/challenge/intermediates`: take detections, jersey evidence and colour labels from the submitted
  run instead of computing them. Only stages 4 to 6 run, on CPU, in about a minute. With `--verify` this is the exact
  replay check. A partial cache works too (e.g. detections and jersey evidence only, so the colour labels are recomputed),
  but jersey evidence and labels point into the detection file row by row: a cached one is accepted only if the
  detections also came from the cache, and every one, cached or computed, is checked against the detection file's rows.
- `--device cpu`: the jersey reader on CPU. The detector uses CUDA when available, otherwise MPS on a Mac, otherwise CPU.
- `--limit N`: smoke test on the first N frames of each video (no reference comparison).

Time for both videos on a single GPU: TODO. On a 2026 MacBook (MPS), 60 frames take about 10 s to detect and 30 s to
read; the colour stage takes about 100 s per video regardless of `--limit`.

## Reproducibility: what matches exactly

`results/challenge/` holds the submitted files and everything they were made from:
- `submission/`: the uploaded `IMG_0114.txt` and `IMG_0115.txt`;
- `intermediates/`: detections and jersey evidence from the Colab run (A100, code version 0ddd8626), and the colour labels;
- `colab_reports/`: the Colab logs and the weights each video used;
- `SHA256SUMS`.

Verified on the development Mac (macOS, Apple silicon):
- `--cache-dir results/challenge/intermediates` (stages 4 to 6 run): tracks and both submission files are
  byte-identical to the submitted ones.
- The same with the colour labels left out of the cache, so stage 3 reruns: still byte-identical, because the labels
  were made on this Mac.
- `--limit 60` on both videos with `--device cpu` runs every stage without error.

`ti_hota_vs_submitted` is the new submission scored with TI-HOTA against the submitted one as if that were the ground
truth. It measures how close the result is, not identity: small coordinate differences can still score 1.0. Only the
hashes (`same`) show byte identity.

Known sources of difference in a fresh run:
- **GPU inference (stages 1 and 2).** Different GPUs, drivers and kernels give slightly different floats, so detection
  and jersey files from a fresh run are not byte-identical to the Colab ones. On the first 60 frames, the Mac (MPS) gave
  820 boxes on IMG_0114 against 818 on the A100, with no line identical.
- **Video decoding (stages 1 to 3).** Frames are decoded with `ffmpeg -hwaccel auto`, which uses the platform's hardware
  decoder (VideoToolbox on a Mac, NVDEC or software elsewhere). Pixel values differ slightly between decoders: on
  IMG_0114, software decoding changes 4.8% of the bytes of a frame, by up to 28 levels. Measured on the Mac by forcing
  software decoding for stage 3 with the cached detections and jersey evidence: 0.93% / 0.75% of the colour team calls
  change, and `ti_hota_vs_submitted` is 0.9962 (IMG_0114) and 0.9992 (IMG_0115).

So a fresh run on another machine is not expected to be byte-identical. How close a fresh run with the production
weights gets is not measured yet: the 0.9962 / 0.9992 above isolate decoding in stage 3 only, with stored detections and
jersey evidence, and fresh GPU inference adds its own differences. The exact reproduction is
`--cache-dir results/challenge/intermediates --verify`, from the stored model outputs.

Stages 4 and 5 are CPU code with no randomness; on the Mac, repeated runs give identical files. That they are also
identical on Linux is what `--verify` on another machine checks.

Not yet verified: a fresh run with the production eligibility weights (`4d403c88`) on a GPU. The `--limit 60` smoke test
above used the local `57a986ea` file, so it shows that the code runs, not that the results match.
