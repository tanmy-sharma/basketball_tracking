# Models and training runs (detector and jersey numbers)

Written 2026-11-02. Covers everything built and run for the detector and jersey-number recognition (JNR): the code in
`src/`, the Colab notebooks, every trained model and its results, the bugs found on Colab, and the decisions taken.
The tracker side (`src/tracker`, `src/team`) is documented in docs 12–15.

## 1. Status at a glance

| Item | State |
|---|---|
| RF-DETR Small fine-tuned on player boxes | Trained, scored on the held-out game, run on all 8 training videos |
| Fine-tuned detector on the challenge videos | **Not run yet**: IMG_0114 and IMG_0115 were missing from the Drive folder in that run |
| Torso crops (ViTPose) for all 8 training videos | Done |
| Weak readable/not-readable labels (3 folds) | Done |
| DINOv2 eligibility head, ViTSTR recogniser, baselines | Trained and scored on IMG_0108 |
| **Submission JNR pipeline** | **Chosen: DINOv2 eligibility head → fine-tuned ViTSTR-small → roster-limited track vote** |
| Wiring the JNR models into the tracker | Not built |
| Anything tested on the challenge camera | Only one visual check of 16 warm-up frames of IMG_0114 (section 6.4) |

Nothing from this work is committed yet. The code, docs and notebooks are untracked or modified files on branch
`setup/uv-env-and-docs`.

## 2. Pipeline

```
video frame ──► RF-DETR Small (fine-tuned) ──► player box (also gives the court position)
                                                 │
                                                 ▼
                            ViTPose base: shoulders (5,6) + hips (11,12) ──► torso crop, ~20×33 px at 720p
                                                 │
                                                 ▼
                            DINOv2-small eligibility head: is the number readable? (score ≥ 0.5)
                                                 │
                                                 ▼
                            ViTSTR-small (fine-tuned): reads the digits; scored over the team's 3 roster numbers
                                                 │
                                                 ▼
                            vote over the track (fall back to all crops when a window has no readable crop)
```

**Closed set.** Only 9 jersey numbers exist across all rosters and ground truth: 1, 3, 4, 5, 6, 9, 10, 11, 12. Bibs:
green {6, 9, 11}, yellow {6, 9, 10}, pink {1, 3, 5}, purple {4, 10, 12}. Challenge games: IMG_0114 has yellow
{6, 9, 10} vs green {6, 9, 11}; IMG_0115 has purple {4, 10, 12} vs pink {1, 3, 5}. All of these bib and number pairs
appear in the training games.

**Why this design.** The user chose a two-stage design, a readability filter and then real OCR, because it makes a clearer
paper than a closed-set classifier. DINOv2 `number_all` scored higher on IMG_0108 (section 7) and stays as a baseline.

## 3. Data and splits

- **Training games:** IMG_0104, 0105, 0106, 0107, 0109, 0110, 0111.
- **Test game:** IMG_0108, a training-set game held out from every model. There is no official test set, and the
  challenge videos have no labels.
- **Validation:** the last 20% of each training game, with a 150-frame guard band on each side that is used by neither.
  Validation shares players, bibs and lighting with training, so it is only for early stopping.
- **Weak-label folds:** [0104, 0109], [0105, 0110], [0106, 0107, 0111]. Each fold covers all 9 numbers.
- **Data fixes, logged at load:** IMG_0105 has 25 duplicate (frame, track_id) rows, which are dropped. IMG_0110 has 1
  box on an off-roster track 0 at frame 6416, which is dropped.
- **IMG_0104:** the container counts 11,400 frames, but only about 11,380 decode, and labels end at 11,376. The
  overlays at frames 1 and 11,376 are tight, so the labels are aligned and the extra frames are an unlabelled tail
  that does not even decode.
- **Rules followed:** only the official training data is used. Nothing is trained on challenge frames, and no labels
  were made for them. Inference on challenge videos is allowed.

## 4. Code

```
src/
  core/        config schema (pydantic, YAML `_base` inheritance, --set overrides, config hash), training loop,
               plugins, run directories, checkpoints, paths (outputs_dir(), SRC_OUTPUTS), drop_games.py
  train.py     one entry point for every model: python -m src.train --config <yaml>
  dataio/      label loading and cleaning, splits, row selection, crop store, video decoding (refuses challenge videos)
  crops/       ViTPose torso crops: geometry, pose, extract (+ --overlay alignment check)
  classifier/  DINOv2-small and ResNet-34 backbones, number and eligibility heads, predict
  recognizer/  ViTSTR-small: tokens, model, transforms, dataset, roster scoring, predict (--stock, free decoding)
  weak/        3-fold weak eligibility labels: folds, assemble, finalize, qa
  evaluate/    metrics and the held-out report (python -m src.evaluate.classifier)
  detector/    RF-DETR: COCO export, training plugin, predict, evaluate
  infer/       detect_video.py: fine-tuned RF-DETR on any video, including challenge videos (inference only)
  configs/     classifier/, recognizer/, detector/, smoke/
  tests/       255 tests, all passing on 2026-11-02
```

The package is called `dataio` because `.gitignore`'s `data/` would match a `src/data/` folder. Run outputs go to
`outputs/` (gitignored). Each run gets `outputs/runs/<run_name>__<config hash>/` with `config.json`, `manifest.json`
(git hash, bundle hash, package versions, weight sources, input-file hashes), `log.txt`, `metrics`, `best.pt`, `last.pt`
and `done.json`.

## 5. Models

All weights are in `outputs/colab/<code version>/runs/<run>/` locally (gitignored) and in
`My Drive/tracking_results/src_<notebook>/<code version>/runs/` on Drive.

### 5.1 Pretrained checkpoints used (disclose in the paper)

| Component | Source | Licence | Notes |
|---|---|---|---|
| RF-DETR Small | Roboflow, COCO-pretrained, `rf-detr-small.pth` (md5 checked by the library) | Apache 2.0 | Fine-tuned here |
| DINOv2 small, `facebook/dinov2-small` | Meta | Apache 2.0 | Backbone of the eligibility head and the 9-class baseline |
| ViTPose base, `usyd-community/vitpose-base-simple` | Hugging Face transformers port | Apache 2.0 | Not fine-tuned. Upstream was trained with the OpenMMLab stack: flagged against the NDA list, the user approved using it |
| ViTSTR-small, `vitstr_small_patch16_224_aug.pth` | `roatienza/deep-text-recognition-benchmark` release v0.1.0, sha256 `3ff5c049…` | Repo Apache 2.0; weights carry no separate licence, trained on synthetic text | Fine-tuned here |
| ResNet-34 ImageNet | torchvision | Code BSD-3; ImageNet terms apply to the weights | Baseline only |

Considered and rejected: TrOCR (its tokenizer would not load under transformers 5.15.1, and it treats whole numbers as
single tokens), PaddleOCR (too hard to set up in this environment), a hockey legibility ResNet-34 (CC BY-NC), and stock
EasyOCR and MGP-STR (11.7% and 0.4% exact on 239 IMG_0106 crops). The NDA excludes two named libraries and anything
built on them; see the memory note, not repeated here.

### 5.2 Trained models

Batch size 128 for all crop models. Each phase unfreezes the last N blocks with layer-wise LR decay 0.75; early stopping
patience is 4 epochs, counted only in phases with unfrozen blocks. "Val" is balanced accuracy on the validation block,
which shares people with training, so it is optimistic.

| Run | Role | Architecture | Phases (epochs, blocks unfrozen) | Best epoch / ran | Val | sha256 (first 16) |
|---|---|---|---|---|---|---|
| `rfdetr_small__2a3fda10` | **Detector (used)** | RF-DETR Small, all layers trained, 1 class | 30 epochs, batch 32, lr 1e-4 (encoder 1.5e-4) | 28 / 30 | mAP50:95 0.8185 | `471b3fff3f77d320` |
| `eligibility__6fd02d77` | **Readability (used)** | DINOv2-small + 2-class head | head 3 (0), finetune 10 (3) | 13 / 13 | 0.900 vs weak labels | `57a986ea4d332ca3` |
| `vitstr_all__04e8fdd7` | **OCR (used)** | ViTSTR-small, 12-token head | top2 3 (2), top6 8 (6) | 11 / 11 | 0.941 | `7319ecec70eb2d5b` |
| `vitstr_eligible__06cf5f62` | Ablation | ViTSTR-small, readable crops only | as above | 6 / 10, stopped early | 0.998 (inflated) | `107f4e2e12d01ff6` |
| `vitstr_xcolor_no_yellow__07ab3bae` | Cross-colour test | ViTSTR-small, no yellow 6/9/10 | as above | 11 / 11 | 0.936 | `debfeb7fbe1bb6a8` |
| `vitstr_xcolor_no_green__6e77c83b` | Cross-colour test | ViTSTR-small, no green 6/9 | as above | 11 / 11 | 0.952 | `01527ea966d5969f` |
| `number_all__52fd0c98` | Baseline | DINOv2-small + 9-class head | head 3 (0), finetune 10 (3) | 12 / 13 | 0.999 | `b40b640fdcd4cd3b` |
| `number_eligible__3ba624d5` | Baseline | same, readable crops only | as above | 10 / 13 | 0.999 | `b53aefa932cf9a6a` |
| `number_eligible_resnet34__f261ecf0` | Baseline | ResNet-34 + 9-class head | head 3 (0), finetune 10 (2) | 6 / 10, stopped early | 0.999 | `62d5377a5ad2de2d` |
| `xcolor_no_yellow__6b8d7ce4` | Cross-colour test | DINOv2 9-class, no yellow 6/9/10 | as number_all | 13 / 13 | 0.998 | `c3e846cd83bb1d73` |
| `xcolor_no_green__cdf8bacf` | Cross-colour test | DINOv2 9-class, no green 6/9 | as number_all | 11 / 13 | 0.998 | `c1ce8baace595ca1` |
| `fold_folds3_*` (3 runs) | Weak-label makers | ViTSTR-small, every 2nd frame | head 2 (0), finetune 4 (2) | 6 / 6 | 0.709, 0.695, 0.661 on the held-out fold | in Drive only |

ViTSTR details: greyscale 224×224 input (letterbox), a 12-token head ([GO], [s], 0–9) initialised by copying the
pretrained rows, max length 4, no label smoothing. Roster scoring sums log-probabilities of each allowed number.

## 6. Detector

### 6.1 Training

- **Data:** every 15th labelled frame of the 7 training games at 720p. Last 20% of each game is validation, with a
  150-frame guard band. IMG_0108 and the challenge videos are refused by the exporter.
- **Run:** RF-DETR Small, resolution 512, batch 32 (16 in the config, raised with a one-cell patch because batch 16 used
  only 17 of 40 GB), 30 epochs, cosine LR, EMA, bf16. About 1 minute per epoch on an A100, 29.8 min in total. The
  dataset export took about 16 minutes on top.
- **Validation:** mAP50:95 rose from 0.721 at epoch 1 to 0.8185 at epoch 28, and mAP@50 was 0.987–0.989 from epoch 1.

### 6.2 Held-out result on IMG_0108 (every 5th frame, on-court only)

| Detector | Conf | Recall | Precision | F1 | False boxes/frame | Foot error median | p90 | Feet within 30 cm | Within 100 cm |
|---|---|---|---|---|---|---|---|---|---|
| Stock RF-DETR Small | 0.40 | 0.936 | 0.731 | 0.821 | 2.07 | 15.9 cm | 38.1 cm | 83.4% | 98.0% |
| **Fine-tuned** | 0.50 | **0.985** | **0.996** | **0.990** | **0.02** | **5.6 cm** | **18.4 cm** | **95.8%** | **99.8%** |

Recall where players overlap heavily (IoU > 0.5) went from 0.58 to 0.93, and the box-bottom bias went from −7.0 px to
−0.3 px. TI-HOTA matches at 100 cm, so the fine-tuned boxes land within tolerance 99.8% of the time on this camera.

### 6.3 Full-video detections

`colab/detect_finetuned_videos.ipynb` (code version `2fb1eb84`) ran the fine-tuned detector on every frame of all 8
training videos: on average 5.97–6.00 boxes per frame at conf ≥ 0.5, as expected for 6 players. Files:
`outputs/colab/2fb1eb84/<video>/detections/<video>__rfdetr-small-ft-471b3fff__720p.txt`, in the same format as the
RF-DETR Large detections. The 7 training games are in-sample for this detector; only IMG_0108 is a fair test.
**IMG_0114 and IMG_0115 were not processed:** they were not found in `My Drive/tracking_videos` in that run.

### 6.4 Challenge camera (one visual check)

16 pre-game frames of IMG_0114, run locally with the real weights: all 6 bib-wearing players on court were boxed
(conf 0.79–0.93); people without bibs (the ball handler, a man waving, spectators) were ignored; one purple-bib player
waiting off court was boxed and must be removed by the on-court filter. The detector learned "player in a bib", which
matches the labels. In-game frames on this camera are still unchecked.

## 7. Jersey-number results on IMG_0108

### 7.1 Crops and weak labels

- **Crops:** 112,000 torso crops across 8 videos, about 4 minutes per video. Pose usable on 99.6–99.8% of crops;
  median crop 19–20 × 33–34 px. IMG_0108 has 14,520 crops, 14,358 usable.
- **Weak labels:** a crop is "readable" if the fold model that never saw its game reads it correctly
  (thresholds 0.7 / 0.3). 46,616 crops labelled (every 2nd sampled frame); 54% readable. By facing: back 67%, front 62%,
  side 4%.
- **Finalize:** all 25,358 readable crops came from the weak labels; the eligibility head filled nothing, because every
  fold saw every bib.

### 7.2 Per-model scores (all 14,358 crops)

"3-way" picks among the player's team roster. Votes are sums of probabilities, roster-limited.

| Model | 3-way | 9-way | Track vote, whole game | 10 s windows | 2 s windows | Free read, exact |
|---|---|---|---|---|---|---|
| Stock ViTSTR (untrained) | 0.385 | 0.063 | – | – | – | 0.037 |
| **ViTSTR `vitstr_all`** | 0.941 | 0.884 | 6/6 | 1.000 | 0.990 | 0.879 |
| ViTSTR, readable only | 0.801 | 0.634 | 6/6 | 0.996 | 0.948 | 0.633 |
| DINOv2 `number_all` | **0.981** | **0.977** | 6/6 | 1.000 | 0.996 | – |
| DINOv2, readable only | 0.954 | 0.925 | 6/6 | 1.000 | 0.989 | – |
| ResNet-34, readable only | 0.908 | 0.897 | 6/6 | 1.000 | 0.981 | – |

On the report's "clean" subset (n = 10,036), `vitstr_all` reaches 0.966 and `number_all` 0.999.
ViTSTR's main confusion is reading 11 as 1 (51 crops), which the roster limit removes. Side views are hardest (0.896).

### 7.3 The chosen pipeline: eligibility head + `vitstr_all`

| Setting | Result |
|---|---|
| Crops with eligibility ≥ 0.5 | 49% of crops, 97.6% correct (3-way) |
| Track vote over readable crops: whole game / 60 s / 10 s | 1.000 / 1.000 / 0.996 |
| Track vote over readable crops, 2 s windows | 0.980 when it votes, no vote on 11.1% of windows |
| Fallback rule | When a window has no readable crop, vote over all crops (`vitstr_all` all-crop 2 s: 0.990) |

The eligibility head is a weaker predictor of a correct read than ViTSTR's own top probability (AUC 0.72 vs 0.86). The
paper should say so; it may still help on the challenge camera, where crops are harder.

### 7.4 Cross-colour test: digit reading or memory?

Each model was trained without one colour's numbers, then scored on those players (3-way, per crop):

| Held-out player | ViTSTR | DINOv2 9-class |
|---|---|---|
| Yellow 10 | 0.21 | 0.00 |
| Yellow 6 | 0.54 | 0.47 |
| Yellow 9 | 0.70 | 0.83 |
| Green 6 | 0.26 | 0.12 |
| Green 9 | 0.84 | 0.66 |

Both models partly memorise bib colour and appearance with the number: each got 1 of 6 tracks wrong in every
cross-colour run. ViTSTR did better on 4 of 5 held-out players, so it reads digits somewhat better. On the challenge
videos every bib and number pair was seen in training, so memory helps rather than hurts there, but the camera is
different.

### 7.5 Jersey-number evidence for every detection (input to the tracker)

`src/infer/jnr_video.py` runs the chosen pipeline on detections rather than ground-truth boxes. For every detection with
conf ≥ 0.3 on every 2nd frame it writes one row to `<video>__rfdetr-small-ft-471b3fff__720p__jnr.csv.gz`:
`det_row` (0-based line in the detection file, for joining), the box, `max_iou`, `pose_conf`, `facing`, `status`,
crop size, `elig`, `p_1 … p_12` (the 9 number probabilities, before any roster limit) and `read` (free read; load it with
`dtype={"read": str}`). Renormalise `p_*` over the team's 3 roster numbers before voting.

Smoke test with the real weights on the first 100 frames of IMG_0108 (300 detections, pre-game, Mac): roster-limited
accuracy 97.7% on all crops and 100% on the 125 crops with `elig ≥ 0.5`, against boxes matched to ground truth at
IoU ≥ 0.5. The full run is `colab/jnr_all_videos.ipynb`; it also detects IMG_0114 and IMG_0115, which had no
fine-tuned detections yet.

## 8. Colab notebooks

All are generated by builder scripts that embed the code, so Colab needs no git access. Rebuild after changing code.

| Notebook | Builder | What it does | Time on A100 |
|---|---|---|---|
| `colab/train_detector.ipynb` | `colab/build_src_notebooks.py` | RF-DETR fine-tune, stock vs fine-tuned predictions on IMG_0108, evaluation | ~50 min (export 16, train 30, predict and eval 5) |
| `colab/train_classifier.ipynb` | `colab/build_src_notebooks.py` | Crops, weak folds, eligibility, finalize, QA, stock ViTSTR, 4 ViTSTR runs, 5 DINOv2/ResNet runs, evaluation | ~1 h 35 min (crops 32 min, the rest 63 min) |
| `colab/detect_finetuned_videos.ipynb` | `colab/build_detect_finetuned_notebook.py` | Fine-tuned detector on every frame of all 10 videos, 8 previews each | ~3–6 min per training video |
| `colab/jnr_all_videos.ipynb` | `colab/build_jnr_notebook.py` | Per video: reuse fine-tuned detections from Drive (or detect first), then eligibility + ViTSTR scores for every detection | not measured yet |

How they behave:
- **Run:** A100 runtime, then Runtime > Run all. Videos come from `My Drive/tracking_videos` (any subfolder, exact file
  names), annotations from the official Drive folders.
- **Saving:** results go to `My Drive/tracking_results/src_<notebook>/<code version>/` after every step. Training
  checkpoints (`last.pt`, RF-DETR `last.ckpt`) are copied there every 10 minutes and removed when the run finishes.
- **Resuming:** Run all again skips finished steps and continues an unfinished run from its last copied checkpoint.
  A rebuilt notebook whose code changed gets a new code version, hence a new folder and a fresh start.
- **Missing videos:** a video that is missing, incomplete or undecodable is skipped with a warning; the run continues
  in a folder named `<code version>_without_<videos>` with the games removed from every config
  (`src/core/drop_games.py`). Without IMG_0108 the models train but nothing is scored.
- **Downloads:** at the end, even after a failure, `reports_<ver>.zip` and `weights_<ver>.zip` are copied to Drive and
  downloaded. Allow multiple downloads in the browser.
- **Code versions used:** detector `e6a5aded`, classifier `e0966686`, fine-tuned detection `2fb1eb84`. The training
  bundle only includes the packages training needs, so edits elsewhere in `src/` do not change these versions.

### 8.1 Problems found on Colab, all fixed

| Problem | Cause | Fix |
|---|---|---|
| Detector training crashed on import | Colab sets `MPLBACKEND` to its inline backend, which the venv's matplotlib rejects | Subprocesses get `MPLBACKEND=agg` |
| Packaging crashed, so nothing downloaded | `CODE_VERSION.json` in the bundle has a 1970 timestamp, which zip rejects | Zips use `strict_timestamps=False` |
| Checkpoints never reached Drive | `last.pt` was only on the runtime disk, and steps saved only when they finished | 10-minute background copy of resume files |
| IMG_0104 crop step failed | The decode check wanted all 11,400 counted frames | Requires only every labelled frame to decode; warns about the tail |
| Each reports zip contained the previous zips | `restore()` copied old zips back from Drive | Root-level zips are not restored |
| Code version kept changing | Another session's `src/team` edits were bundled | Bundle limited to training packages |

## 9. Decisions and constraints

- The NDA excludes two named libraries and anything built on them; one of the two names must never appear in the
  repo, and `src/tests/test_compliance.py` checks for it by hash. A transitive install of `python-doctr` and
  `rf-groundingdino` (via `trackers[detection]`) is tolerated only while unused; the notebooks skip installing them.
- ViTSTR-small was chosen as the OCR model after TrOCR and PaddleOCR were rejected (section 5.1).
- The detector's batch size was raised to 32 for the run; the config still says 16.
- The submission uses the eligibility head and ViTSTR (section 7.3), with DINOv2 `number_all` and ResNet-34 as paper
  baselines.
- Training runs on Colab A100, not on the Mac.

## 10. Open items and risks

1. **Run the fine-tuned detector on IMG_0114 and IMG_0115** (upload both to `My Drive/tracking_videos`, then Run all on
   `detect_finetuned_videos.ipynb`; finished videos are skipped) and look at their previews during play.
2. **Wire the JNR models into the tracker** with the fallback rule from section 7.3. The per-detection evidence file is
   built (section 7.5). The tracker side that votes over it is built too ([17_jnr_tracker_extension.md](17_jnr_tracker_extension.md)),
   but it has only been tested on synthetic data.
3. **Camera shift:** every score here comes from the training camera. The challenge camera is angled, so crops will be
   smaller and more often side-on.
4. **Memory over reading:** the cross-colour results show the models lean on bib colour and person appearance.
5. **Weak labels cover every 2nd frame only,** which is why the readable-only models trained on about 25,000 crops
   instead of about 97,000 and scored worse.
6. **RF-DETR resume from `last.ckpt`** has not been exercised on Colab; plain retraining took 30 minutes.
7. **Nothing is committed.** Commit 223aa8f contains the excluded name in five docs; rewriting that history needs the
   user's approval and affects the branch the other session uses.
