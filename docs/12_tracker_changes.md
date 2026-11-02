# Changes made to UCMCTrack

Written 2026-11-01. The code is in `src/tracker/`, adapted from [UCMCTrack](https://github.com/corfyi/UCMCTrack) (MIT, commit
7851a49). Numbers are from IMG_0106 with RF-DETR Large detections. The reasoning and the post-processing decision are in
[11_post_processing_decision.md](11_post_processing_decision.md).

## Adapting it to a fixed camera and a known court

| Change | File |
|---|---|
| The camera model is the homography from `court_registration.csv`, not a camera parameter file | `mapper.py` |
| Camera motion compensation is removed, since the camera doesn't move | `ucmc.py` |
| Foot-point noise scales with box height, with no fixed pixel limits, so it carries over between resolutions | `mapper.py` |
| Detections up to 100 cm outside the court are tracked, but only rows inside the court are written | `run_ucmc.py` |

## Fixes

- **Coasting bug from upstream:** an unmatched track stayed Confirmed when there were no low-score detections for it to compete
  for. It is now marked as coasted (`ucmc.py`).
- **Bug in our ambiguity check:** it stopped nearby new tracks from ever confirming. New tracks now skip the check, since they
  have no identity to protect yet (`associate_tentative` in `ucmc.py`).

## Changes that are on

| Change | Detail | Setting | File |
|---|---|---|---|
| Clearer track age | `death_count` now means frames since the last observation. The old `cdt=5` equals the new `cdt=3` | | `ucmc.py` |
| Longer retention | Tracks stay alive for 8 unobserved frames | `cdt=8` | `run_ucmc.py` |
| Retention separate from output | Tracks are kept while unobserved, but predicted-only rows aren't written | `emit_coast=0` | `run_ucmc.py` |
| Noise by detection confidence | Lower-confidence boxes get more foot-point noise (`CONF_NOISE`). The homography still carries that noise into court space | `conf_noise_scale=3.0` | `mapper.py` |
| Motion noise | Set by testing values | `wx=wy=25` | `run_ucmc.py` |
| Ambiguous matches not committed | If a match beats a rival pair by less than the margin, the track coasts and the box is dropped, so it can't corrupt the track's motion state | `ambiguity=1.0` | `ucmc.py` |

Unchanged: the ranking cost (Mahalanobis² + log det) and its limit of 20, and 3 hits to confirm a new track.

## Built and tested, but off

| Switch | Off value | Why it's off |
|---|---|---|
| Independent Mahalanobis gate | `gate=0` | No effect at 30 or above, worse when tighter |
| Recently seen tracks match first | `recency=0` | No meaningful change |
| Confirm new tracks after 2 hits | `birth_hits=3` | No clear gain |

## New tools

- `src/tracker/diagnose.py` classifies each identity break as a takeover, handover, restart or duplicate. It also counts
  recoveries and wrong-player frames.
- `src/tracker/evaluate.py` scores court tracks against the training ground truth.
- `src/tracker/annotate.py` draws track boxes and IDs on the video.
- `tools/tests/test_ucmc.py` has 10 tests covering track lifecycle, retention vs output, ambiguity, the gate and confidence noise.
  All pass.
- `kalman.py` gains `distance_terms`, which splits the cost into its Mahalanobis and log-det parts so the gate can use the first
  on its own.

With every new switch off and `cdt=3`, the code reproduces the first version's output byte for byte.

## Result

| Metric | First version | Now |
|---|---|---|
| AssA | 5.0 | 12.1 |
| IDF1 | 12.2 | 22.7 |
| Box HOTA | 18.2 | 28.5 |
| DetA | 67.2 | 67.2 |
| Track IDs | 253 | 67 |
| Restarts | 129 | 25 |
| Duplicates | 81 | 3 |
| Swaps (takeover + handover) | 205 | 247 |
| Wrong-player frames | 1,369 | 8,467 |
| Coverage of GT boxes | 0.926 | 0.931 |
| Purity | 0.978 | 0.865 |

Swaps, wrong-player frames and purity got worse, because a wrong link now lasts longer. The settings were tuned on this one
game. With ground-truth boxes on all 8 games, old and new settings are about level (AssA 22.0 vs 20.9).

## Output files

- `outputs/tracks_ucmc/IMG_0106__rfdetr-large.txt`: current tracks, 73,790 rows and 67 track IDs.
- `outputs/tracks_ucmc/IMG_0106__rfdetr-large__v1.txt`: first-version tracks, kept for comparison.
- `outputs/videos/IMG_0106__ucmc.mp4`: annotated video of the current tracks.
