# Reuse from `object_tracking_paper`: progress notes

Survey finished 2026-11-01. No code has been ported yet.
Source repo: `/Users/tanmaysharma/code/research/object_tracking_paper`.
Its checkout moved from `feat/cache_creation` (`ac6eebf`) to `audit` (`9ee418e`) during the survey, so
re-check paths and line numbers before copying anything.

## Survey status

All three read-only Haiku surveys finished (verifier inventory, overall map, failure diagnosis). Their findings
come from reading the code; nothing was run.

## Where the misses come from: tools to reuse

Most of the failure-study pipeline works from tracker MOT output, MOT GT and a per-frame detection cache, so it
can score Roboflow `trackers` output. The exception is `track_deaths.py`, which needs per-frame cost matrices
and gate thresholds that only this repo's `SortTracker` records.

| Rank | Item (under `scripts/failure_study/` unless noted) | What it answers | Port effort |
|---|---|---|---|
| 1 | `build_ledger.py`, `src/eval/stratified.py` `clear_ledger` | Per-row link state and ID switches, tied back to `eval_clear` totals | Easy to adapter |
| 2 | `make_records.py` `label_rows` | Per-GT-box label. `lost_no_det` = detector miss, `lost` = association miss | Easy |
| 3 | `build_gtcontext.py` `det_available`, `best_det_iou` | Whether a usable detection existed (IoU >= 0.5) | Adapter: cache layout, replace S7 check |
| 4 | `idsw_taxonomy.py` | ID switch causes: detector, direct swap, re-emergence, fragment, drift | Adapter |
| 5 | `fragmentation.py` | Gaps resumed by the same ID vs a new one; whether a longer `max_age` would help | Easy |
| 6 | `episode_failures.py`, `survival.py` | Identity during and after occlusion | Easy; `survival.py` is adapter level |
| 7 | `scripts/viz/error_video.py` | Visual check of each event | Adapter: frame layout, hard-coded ffmpeg path |
| 8 | `track_deaths.py` | Contention vs gate failure for each lost track | Entangled |

Also `src/eval/eval_mot.py` (metrics, numpy and scipy only) and `src/eval/analyzer.py` (tracker-agnostic analyzer;
levels 0-2 need no decision records).

Porting cautions:
- Do not use `run_all.sh`: it hard-codes DanceTrack paths and calls `ablation.py`, which is missing.
- Re-derive frame-based constants, which assume 20 fps (`PHASE_WINDOW`, `MAX_REEMERGE_GAP`, `SURVIVAL_HORIZON`).
  Training video is 29.97 fps.
- GT here has no visibility column, so use `occ_source=depth_proxy`, and check that proxy on this camera.
- All matching is 2D image IoU. This explains box-level misses but does not measure TI-HOTA, which needs court
  positions and team+jersey identity.
- No tests were found by filename for `track_deaths.py`, `fragmentation.py`, `survival.py` or
  `episode_failures.py`.
- `eval_mot.py` follows TrackEval conventions. Check for line-level ports before copying (TrackEval's licence is
  believed to be MIT; not verified).

## Verifiers most worth reusing

About 21 substantive verifiers exist. None of them checks court-plane or homography coordinates, so this
project needs new checks for that. Thresholds and constants are specific to DanceTrack and need replacing.

| Item | What it catches | Reuse |
|---|---|---|
| `src/eval/eval_mot.py` `build_sequence_data` | Same track ID twice in one frame, which inflates IDF1 and AssA | Direct, for MOT-format output |
| `src/eval/stratified.py` `assert_recomposes`, `_check_strata` | Per-stratum metrics that don't sum back to the overall HOTA | Direct, for any stratified miss report |
| `src/tcd/decode.py` `assert_same_space` | Mixing coordinate spaces before computing IoU | Pattern, for image vs court coordinates |
| `scripts/failure_study/sanity.py` | Cross-script mismatches in IDSW, Frag and taxonomy totals | Pattern only: IDENTITY / CONTROL / REPORTED checks. Constants are hard-coded |
| `scripts/failure_study/build_ledger.py`, `build_gtcontext.py` `_check_s7` | Ledger totals vs metric totals; detection counts vs tracker trace | Likely useful with the failure study |
| `src/track/cache_benchmarks.py` `verify_sequence`, `check_sha256sums`, `check_job_params` | Corrupted or mismatched detection caches | Pattern, for a frozen detection cache |
| `scripts/m4_sanity.py` | Custom detector decode or rescale drifting from a reference | Only with a custom decode path |
| `src/tracker/sweep.py` `_check_identity`, `src/tracker/compare.py` `_check_comparable` | Regressions in control arms; comparing runs on different data | Pattern, for comparing trackers |
| `tests/test_imm.py` `_assert_spd` | Kalman covariance that isn't symmetric positive definite | Direct, test helper |

## Other components

Likely reusable:
- `src/eval/eval_mot.py`: HOTA, CLEAR and Identity scoring. This is standard box-level scoring and does not
  replace TI-HOTA in `tools/trackid/metric.py`.
- `src/common/device.py`: CUDA, MPS or CPU selection.
- `src/track/kalman.py`, `matching.py`, `matchers.py`, `sort.py`: tracking building blocks.
- `src/track/cache_detections.py`: detection cache.
- `src/tcd/gt.py`: MOT ground-truth loader.

Not in that repo: ReID, jersey identification, court-plane or homography code.

Offline tracklet restitching exists in `scripts/global_track_restitch/`. The map agent ranked it low for being
"not real-time", but this competition has no real-time requirement, so it deserves another look.
The organizers have confirmed that automatic offline processing is allowed (`competition/organizer_emails.md`, Email 5).

## Compliance points to settle before reuse

- `models/` holds ByteTrack YOLOX-x checkpoints fine-tuned on DanceTrack, MOT17, MOT20 and CrowdHuman. Whether
  these count as "general-purpose" is unconfirmed. They must be public and disclosed, and they are not the
  challenge data. Decide per checkpoint.
- The repo has no `LICENSE` file. `third_party/yolox_bytetrack/` is vendored, and its licence was not verified.
- Any reused code ships in the competition code ZIP.

## Progress

Ported so far: `eval_mot.py` and `stratified.py` into `tools/trackid/`, with a simplified per-box miss label in
`tools/run_trackers.py`. The rest of the failure study is not ported.

Next run: `colab/detect_track_IMG_0106.ipynb` (rebuild with `colab/build_notebook.py` after changing `tools/`).
It runs `rfdetr-large` on all of IMG_0106, then SORT, ByteTrack, OC-SORT, BoT-SORT and CBIoU, and saves each
step to `My Drive/tracking_results/IMG_0106/<code version>/`. Rerunning resumes from what is saved.

Checkpoint to disclose: `rfdetr-large` through `inference-models`: Apache 2.0, COCO-pretrained, 704 x 704, 33.9M params.
