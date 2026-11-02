# src/: RF-DETR fine-tune, jersey readers, weak eligibility labels

Run everything from the repo root with `uv run python -m ...`. `src` is not an installed package. Plan and design
decisions are in `docs/10_src_implementation_plan.md` (section 8 overrides the earlier sections).

Pipeline: RF-DETR player box -> ViTPose shoulders and hips -> torso crop -> DINOv2 eligibility head -> fine-tuned
ViTSTR-small digit reader -> score the team's 3 roster numbers -> vote across the tracklet. The DINOv2 9-class head and
ResNet-34 are baselines for the paper, not the production reader.

## Folder map

| Path | What it holds |
|---|---|
| `core/` | Config schema (pydantic, `_base` inheritance, `--set`), run directories, atomic checkpoints, the generic phase loop, plugin registry |
| `dataio/` | Label loading (drops stray tracks, dedupes), ffmpeg frame iterator, crop store (`crops.bin` + `manifest.csv`), train/val split with guard band, row selection |
| `crops/` | Torso geometry, ViTPose wrapper, `extract` (one game to crops, resumable) |
| `classifier/` | DINOv2 and ResNet-34 backbones, 9-class and eligibility heads, augmentations (no flips) |
| `recognizer/` | ViTSTR-small: digit tokens, weight loading with sha256 check, roster-constrained scoring, free decoding, stock baseline |
| `weak/` | 3-fold weak eligibility labels (`folds`), final eligible list (`finalize`), contamination checks and contact sheets (`qa`) |
| `evaluate/` | Held-out report on the test game: accuracy, per-track variance, tracklet votes, coverage curves, free reads |
| `detector/` | RF-DETR Small fine-tune wrapper, COCO export, stock/fine-tuned prediction, detector comparison |
| `tracker/` | UCMCTrack tracker code (separate work, not used by anything below) |
| `team/` | Team colour without a model: calibrate on 10 random frames, classify any crop, write labels, score against GT, draw a team video. `python -m src.team.run VIDEO --boxes gt\|DETECTIONS [--every N] [--seeds N] [--video]`. See `docs/13` |
| `configs/` | `classifier/`, `recognizer/`, `detector/`, and `smoke/` (Mac smoke tests) |
| `tests/` | `python -m pytest src/tests -q`: no GPU, downloads or real video |
| `train.py` | General entry point: `python -m src.train --config PATH [--set a.b=c]` |

`colab/build_src_notebooks.py` builds `colab/train_classifier.ipynb` and `colab/train_detector.ipynb`.

## Commands

```bash
uv run python -m pytest src/tests -q                      # all tests, about 1-2 minutes
uv run python colab/build_src_notebooks.py                # rebuild both notebooks after any change to src/

# crops (ViTPose; skips games that are done)
uv run python -m src.crops.extract --config src/configs/classifier/base.yaml --games IMG_0104 IMG_0105
uv run python -m src.crops.extract --overlay --games IMG_0104      # alignment check, no pose, no crops

# weak labels
uv run python -m src.weak.folds --config src/configs/classifier/weak_folds3.yaml
uv run python -m src.train --config src/configs/classifier/eligibility.yaml
uv run python -m src.weak.finalize --config src/configs/classifier/weak_folds3.yaml --eligibility src/configs/classifier/eligibility.yaml
uv run python -m src.weak.qa --config src/configs/classifier/weak_folds3.yaml

# digit reader and baselines
uv run python -m src.recognizer.predict --config src/configs/recognizer/vitstr_all.yaml --stock --game IMG_0108 --out outputs/reports/stock_vitstr
uv run python -m src.train --config src/configs/recognizer/vitstr_all.yaml        # also vitstr_eligible, vitstr_xcolor_no_yellow, vitstr_xcolor_no_green
uv run python -m src.train --config src/configs/classifier/number_all.yaml        # DINOv2 baselines: number_eligible, xcolor_*, number_eligible_resnet34

# held-out report on the test game (free reads: src.recognizer.predict --best ... --mode free)
uv run python -m src.evaluate.classifier --game IMG_0108 --out outputs/reports/eval \
    --number vitstr_all=src/configs/recognizer/vitstr_all.yaml --number number_all=src/configs/classifier/number_all.yaml \
    --eligibility src/configs/classifier/eligibility.yaml [--free-reads vitstr_all=free.csv]

# detector (needs the rfdetr group)
uv run --group rfdetr python -m src.train --config src/configs/detector/rfdetr_small.yaml
uv run --group rfdetr python -m src.detector.predict --weights stock --game IMG_0108
uv run --group rfdetr python -m src.detector.predict --weights outputs/runs/<run>/best.pth --game IMG_0108
uv run python -m src.detector.evaluate --game IMG_0108 --stock STOCK.txt --finetuned FT.txt
```

Every run lives in `outputs/runs/<run_name>__<config hash>/` with `config.json`, `manifest.json` (git hash, versions,
weight sources and hashes), `metrics.csv`, `best.pt` and `done.json`. A finished run is skipped when its command is
repeated; an unfinished one resumes from `last.pt`.

### Smoke tests on the Mac (tier 2, real weights, IMG_0106)

```bash
uv run python -m src.crops.extract --config src/configs/smoke/classifier_mac.yaml --games IMG_0106
uv run python -m src.train --config src/configs/smoke/classifier_mac.yaml
uv run python -m src.train --config src/configs/smoke/recognizer_mac.yaml
uv run python -m src.evaluate.classifier --game IMG_0106 --allow-nontest --number smoke=src/configs/smoke/classifier_mac.yaml --out outputs/reports/smoke
uv run --isolated --group rfdetr python -m src.train --config src/configs/smoke/detector_mac.yaml
uv run --isolated --group rfdetr python -m src.detector.predict --weights stock --game IMG_0106 --every 30 --limit 5
```

The detector smoke run is the first real execution of the RF-DETR wrapper. The unit tests use a fake `rfdetr` module.

## Public checkpoints and licences (disclosures)

| Component | Source | Licence | Used for |
|---|---|---|---|
| RF-DETR Small, COCO-pretrained | Roboflow, package `rfdetr` 1.11.2 | Apache 2.0 | Player detector (stock and fine-tuned) |
| DINOv2 small, `facebook/dinov2-small` | Meta | Apache 2.0 | Eligibility head and 9-class baseline backbone |
| ViTPose base, `usyd-community/vitpose-base-simple` | Hugging Face transformers port | Apache 2.0 | Shoulder and hip keypoints for torso crops. Upstream training used the `mmpose` toolbox; that is provenance only, nothing from it is installed or imported here |
| ViTSTR-small, `vitstr_small_patch16_224_aug.pth` | `roatienza/deep-text-recognition-benchmark`, sha256 `3ff5c049...e42aa` | Repo Apache 2.0; the weight files have no separate licence, trained on synthetic text datasets | Digit reader, fine-tuned here on the official crops (head remapped to digits) |
| ResNet-34 ImageNet weights | torchvision | Code BSD-3; ImageNet data terms not checked | Classifier baseline only |

Everything is fine-tuned on the official training videos and annotations only. No other data, no synthetic digits in
this build. Banned tooling (MMDetection family, docTR, STRHub, Ultralytics, GroundingDINO, the excluded jersey
pipeline repo) is never imported; `tests/test_compliance.py` checks this. In the Colab install, `python-doctr` and
`rf-groundingdino` are skipped (`--no-install-package`) because only `trackers[detection]` pulls them in and `src/`
never imports it. `timm`, `transformers` and `torch` still install through the other dependencies.

## Reproduction

1. `uv sync` (add `--group rfdetr` for the detector). Put the 8 training videos in `data/Training/vidoes/` and the
   annotations in `data/Training/` (not in git).
2. Crops for all 8 games (20-30 min on a GPU), then the weak folds. Held-out groups are `[0104, 0109]`, `[0105, 0110]`
   and `[0106, 0107, 0111]`; each fold's ViTSTR trains on the other groups and reads the held-out one, so a crop's label
   says whether a model that never saw that game read the number. Every number is covered in every fold.
3. Eligibility head, then `weak.finalize`, then `weak.qa`. Look at the contact sheets before trusting the labels.
4. ViTSTR fine-tunes (`vitstr_all`, `vitstr_eligible`, two cross-colour runs), then the optional DINOv2 and ResNet-34
   baselines, then `evaluate.classifier` on IMG_0108, the only test game.
5. Detector: fine-tune, predict stock and fine-tuned on IMG_0108, compare.

On Colab, run `colab/train_classifier.ipynb` and `colab/train_detector.ipynb` on an A100 (about 3-4 h and 45-60 min,
both unmeasured). They save to `My Drive/tracking_results/src_<notebook>/<code version>/` after each step, copy training
checkpoints there every 10 minutes, resume after a disconnect, continue without any video they cannot get or read
(in a separate `<code version>_without_<videos>` folder, using `src/core/drop_games.py`), and download `reports_<ver>.zip` and `weights_<ver>.zip` at the end, also after a failure. Allow multiple
downloads in the browser.

## Stock ViTSTR baseline (untrained)

Measured on 239 crops (unfiltered, IMG_0106, 720p, fixed torso fraction of the box, stretched to 224): free read 0.4%
exact, digits-only read 8.4%, roster-limited 3-way 31.4%, against chance of 33.3%. The stock model has no usable
signal on these crops, so fine-tuning has to supply it. Report this as the untrained baseline. The notebook repeats
the measurement on IMG_0108 (`outputs/reports/stock_vitstr/stock_metrics.json`).

## Limits to state alongside any number

- **Leakage.** The test game shares the competition's teams, bibs and venue with the training games. Models may partly
  remember people and bibs, because a number only ever appears on one bib and only `1` repeats across numbers.
  Weak labels carry the same risk. Mitigations: 9-way unmasked scoring, greyscale ViTSTR input, `fold_saw_bib` and facing
  diagnostics in `weak.qa`, contact sheets, and the cross-colour runs, which are the only result that isolates digit
  reading from person memory. Validation is a late block of each game with a guard band, used for early stopping only.
- **Camera shift.** Training crops come from the end-line camera, simulated at 720p. The challenge videos use an oblique
  camera with H.264 compression, and the fine-tuned detector also treats officials and waiting players as negatives
  because only 6 players are labelled. Held-out IMG_0108 scores use the training camera and cannot show this effect.
  Keep stock RF-DETR as the fallback detector. Nothing here measures the classifier shift either.
- **Correlated crops.** Neighbouring frames of one player are almost identical, so per-crop accuracy overstates
  certainty. The per-track table (6 players) is the honest variance estimate.
- **IMG_0104** has 11,400 frames but labels stop at 11,376; the 24-frame tail is treated as unlabelled. Overlays at the
  first and last labelled frames showed boxes tight on the players.
- **Not verified.** RF-DETR training end to end, real DINOv2 weight loading, `gdown.download_folder` against the
  official folders, all A100 timings and the Colab ffmpeg decode of this HEVC.
