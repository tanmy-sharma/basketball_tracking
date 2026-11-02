# End-to-end plan: detection to submission

Written 2026-11-02. Results are due Nov 3, 23:59 AoE. This is a plan: nothing in the team-aware tracker or `identity.py` is built
yet. The model side is detailed in [10_src_implementation_plan.md](10_src_implementation_plan.md), and earlier tracker work is in
[11_post_processing_decision.md](11_post_processing_decision.md) and [12_tracker_changes.md](12_tracker_changes.md).

## Pipeline

```
RF-DETR Large boxes
  -> ViTPose shoulders + hips -> torso crop
       -> bib colour: green, yellow, pink, purple, none -> roster -> team A, team B, not a player, unsure
       -> jersey: eligibility head -> ViTSTR-small -> scores for the roster numbers
  -> court-space UCMCTrack with a soft team penalty
  -> offline: split tracks where the team changes, remove non-player pieces
  -> identity.py: track pieces -> 3 identities per team, fill gaps
  -> submission, checked with tools/validate_submission.py
```

| Stage | Owner | Status |
|---|---|---|
| Detection | Tracker side, run by the user on Colab | `colab/detect_all_videos.ipynb` is ready, not yet run on Colab. IMG_0106 boxes already exist |
| ViTPose and torso crops | Model side | `src/crops/` passes its tests, not yet run on real video. Reads ground-truth boxes only |
| Colour labels on real detections | Built in `src/team/` | Tested on 8 games with ground-truth boxes and 7 with RF-DETR. Challenge videos need detections first. See "Team colours" below |
| Jersey models | Model side | See `docs/10`, section 8 |
| Team-aware tracker | Tracker side | Not built. Only throwaway tests exist |
| Identity assignment and submission writer | Recommended: tracker side, not yet decided | Not built. `docs/10` leaves connecting the models to a tracker out of scope |

## What the tests showed

All team labels here come from ground truth, so these are best cases. The scripts and raw results are in
`outputs/team_label_tests/`, which git ignores.

**IMG_0106 with RF-DETR, one game:**

| Setup | AssA | IDF1 | Box HOTA | DetA | Wrong-player frames |
|---|---:|---:|---:|---:|---:|
| Current tracker (`cdt=8`, ambiguity 1) | 12.1 | 22.7 | 28.5 | 67.2 | 8,467 |
| Non-players removed only | 10.9 | 22.5 | 29.0 | 77.4 | 10,915 |
| Perfect team, same settings | 22.4 | 37.8 | 41.9 | 78.7 | 2,988 |
| Non-players removed, no team, `cdt=20`, ambiguity 0 | 19.3 | 39.7 | 38.8 | 77.9 | 20,394 |
| Perfect team, `cdt=20`, ambiguity 1 | 43.9 | 62.5 | 58.8 | 78.9 | 2,116 |
| Perfect team, `cdt=20`, ambiguity 0 | 45.5 | 64.4 | 59.9 | 79.1 | 88 |

**Ground-truth boxes, 7 of 8 games:** team labels raise average AssA from 21.9 to 35.8, IDF1 from 32.8 to 46.9 and box HOTA from 45.9
to 59.0. Every game improved. IMG_0110 failed on an unmapped track ID in the test script, so it isn't included. `cdt=20` barely changes
these results, so its large gain has been seen on only one game, with RF-DETR.

**Two trackers vs one:** a hard team block in one tracker gives identical output to two separate trackers, even with noisy labels. A
track only ever receives its own team's boxes, so it can never be outvoted. The benefit of one tracker comes only from a soft penalty
and from unsure labels.

**Noisy labels, IMG_0106, `cdt=20`, ambiguity 0.** Labels were flipped at the given rates for clear and overlapped boxes. "Burst" means
errors last 15 frames per player. This is one seed, and the run stopped at 24 jobs, so treat it as a direction only. For comparison,
using no labels gives AssA 19.3.

| Wrong rate, clear / overlap | Noise | Hard block | Penalty 5 | Penalty 10 | Penalty 20 |
|---|---|---:|---:|---:|---:|
| 0.5% / 5% | Random | 38.6 | 36.8 | 39.6 | 39.6 |
| 0.5% / 5% | Burst | 30.0 | 44.0 | 37.2 | 37.7 |
| 2% / 15% | Random | 16.7 | 36.7 | 31.7 | 19.0 |
| 2% / 15% | Burst | 11.9 | 38.1 | 20.3 | 21.4 |
| 2% / 30% | Random | 10.5 | 26.8 | 21.5 | Not run |
| 2% / 30% | Burst | 6.5 | 15.5 | 14.0 | Not run |

A hard block falls below no labels at about 2% / 15%. Penalty 5 beat it in five of these six rows.

**Unsure in overlaps:** marking overlapped boxes unsure gave AssA 11.8 and 2,557 wrong-player frames at `cdt=8`, ambiguity 1. Colour
opinions seem to matter most during overlap, though this wasn't tested at the newer settings.

## Team colours

`src/team/` replaces the ViTPose and colour-classifier route for team labels. It uses no model or weights, so team labels don't depend
on the ViTPose NDA decision. Jersey reading still might.

1. **Calibrate:** draw 10 random frames from the middle 90% of the video. Their median is an empty-court background image, because the
   camera is fixed.
2. **Find each box's colour:** in the chest region of the box (25-75% of width, 15-50% of height), keep pixels that differ from the
   background and aren't court blue. Take the median Lab a and b of the vivid ones.
3. **Decide the colours:** group clear, confident boxes by the nearest bib hue, using the two colours from the roster, then measure both
   colours on this video.
4. **Run:** `TeamColours.run(crop, background=...)` returns a colour, team and scores, or `none` or `unsure`.

Precision is the share of team calls on matched players that name the right team. Calibration frames aren't scored.

| Boxes | Precision | Coverage | Clear | Same-team overlap | Cross-team overlap | Non-players rejected |
|---|---:|---:|---:|---:|---:|---:|
| Ground truth, mean of 8 games, every 5th frame | 98.2% | 98.8% | 99.96% | 99.8% | 89.9% | n/a |
| RF-DETR, mean of 7 games, every frame | 98.8% | 94.0-99.8% | 99.9% | 96.0-97.6% | 86.2% | 97.6% |

Full results, advantages and disadvantages: [14_team_colours.md](14_team_colours.md).

- **Random draws:** over 20 draws per run, precision stayed at 97.7% or above and no calibration raised a warning. Coverage fell to
  89.5% on IMG_0108 (yellow and green, the closest pair) in the worst draw.
- **All boxes, not only matched ones:** for RF-DETR at confidence 0.3 and above, team calls are 94.9% right team, 0.5% wrong team, 3.6%
  loose boxes around a real player, and 1.0% non-players.
- **Fallback table below:** these numbers point to the full soft penalty (clear error under 5%, cross-team overlap error under 15%).
  Measured on 7 training games with RF-DETR. IMG_0104 and the challenge videos have no detections yet.
- **Not measured:** the challenge videos have no ground truth, and they're the camera change the training videos can't test. Their
  cameras moved by at most about 0.3 pixels in the 10 frames checked per video, so background removal should work there.
- **Possible leakage:** the bib references and thresholds were set on the same training games used for scoring. The references only
  group boxes by colour, and each video's colours are measured on that video.
- **Disclose in the paper:** the per-video background image and colours are computed automatically from each challenge video, and team
  names come from the roster and bib colours learned from training.

## Tracker design

**Built and tested 2026-11-02, see [15_team_tracker.md](15_team_tracker.md).** Outcome: soft penalty x6 plus NMS 0.7 at current
settings; veto, birth block, overlap-weighted trust and the looser settings all lost. Oracle TI 0.596 -> 0.758 over 7 games.

- **One tracker, soft penalty.** A team mismatch adds to the matching cost instead of blocking the match. The cost is close to -2 log
  likelihood, so the penalty is `2 logit(q)`, where q is the calibrated chance that box and track share a team. Unsure adds nothing.
  Penalty 5 corresponds to about 92% and penalty 10 to about 99.3%.
- **Team belief.** Each track holds capped log-odds that fade over time, so a swapped track can flip back.
- **Veto, to be tested.** A penalty mostly changes ranking, so a lone wrong-team box may still join a track. Test a veto on clear,
  confident mismatches and keep it only if it wins.
- **Births.** No new track when all 3 of a team's confirmed tracks matched this frame. The full limit is applied offline.
- **Non-players.** Drop a box only when it is clear and confident, and otherwise treat it as unsure. Remove what remains by a vote over
  the whole track piece.
- **Starting settings.** `ambiguity=0`, `cdt=20`, `conf_noise_scale=3`, with one test of `gate=30`.
- **Offline repair.** Split tracks at team change-points. Organizers allow automatic whole-video processing
  ([Email 5](competition/organizer_emails.md)), but no training or manual identity on challenge videos.

### Review of the current tracker

| Item | Finding | Action |
|---|---|---|
| Ambiguity check | Flags valid matchings, throws boxes away, and the track can still take a low-confidence box | Turn off, don't fix. Confirm across games in step 2 |
| Swap and restart diagnostics | Loose definitions. One clean run counted 134 "swaps" with 88 wrong rows, and duplicates 81→3 is partly caused by no longer writing coast rows | Stop using swap counts. Decide on AssA, IDF1, purity and simulated TI-HOTA, and add a cross-team row count |
| Confidence noise ×3 | Variance is 9× the measured table, so it is inflation rather than calibration | Keep for now, one test in the sweep |
| Tuned on one game | Still true of `cdt=20` | Check on IMG_0108 and IMG_0109 once detections exist |

## Interface with the model side

One file per video, `outputs/labels/<video>__<detfile>.csv.gz`, plus the detection file's hash.

| Field | Meaning |
|---|---|
| `frame`, `det_row` | Join key. `det_row` is the 0-based line in the detection file |
| `x`, `y`, `w`, `h` | Copied box, checked on load |
| `p_green`, `p_yellow`, `p_pink`, `p_purple`, `p_none` | Raw scores, NaN if no opinion |
| `status`, `pose_min_conf` | `ok`, `no_colour` or `degenerate` from `src/team/`, which leaves `pose_min_conf` empty. `low_pose` is unused |
| `colour`, `team` | The decision from `src/team/`: a bib colour and A or B, or `none` or `unsure` with no team |
| `p_num_<n>` | 9 jersey scores, in a second file with the same keys |

A missing row means unsure. The tracker maps colour to team through the roster (`BIB_OF_ROSTER`) and applies thresholds, so purple in
IMG_0114 becomes "not a player". Until real labels arrive, `synth_labels` writes this same format from ground truth.

**The model side needs to know:** `extract_game` reads only ground-truth boxes and refuses challenge videos, so jersey scores on
detections need a new path. Team labels no longer depend on it.

## Steps

1. **Detections, now.** Run `colab/detect_all_videos.ipynb` after checking that the videos are in `My Drive/tracking_videos`.
2. **Finish the noise evidence, Mac, under 30 minutes.** Only what can change a decision: IMG_0106, 3 seeds, `cdt` 8 and 20, ambiguity
   0 and 1, penalty 2, 5 and 10, with burst 2% / 15%, occluder takes the label 15%, and overlap unsure. Also ambiguity 0 and 1 with
   perfect labels on ground-truth boxes across games. Keep `ambiguity=0` and `cdt=20` unless an alternative wins mean AssA by at least
   2. Copy the old results first, because the test script deletes them.
3. **`src/tracker/identity.py` and the submission writer.** Split track pieces, sum capped jersey evidence, assign identities with no
   overlapping pieces sharing one and 3 per team, then fill gaps to all 6 identities on every frame. This comes first because TI-HOTA is
   0 without it, and it works on today's tracks. Must pass `validate_submission.py`.
4. **Label loading.** Add `det_row` to `Det` and `COLUMNS`, and a `labels.py`. With no labels, output must stay byte-identical.
5. **Team penalty in `ucmc.py`.** Change `cost`, `commit`, `associate_tentative` and `initial_tentative`. Contrary evidence must flip a
   belief, perfect labels must reach AssA of at least 43 on IMG_0106, and synthetic noise must do at least as well as penalty 5 above.
6. **Diagnostics.** Add `cross_team_rows` to `diagnose.py`.
7. **Real labels.** Measure errors on IMG_0108 for clear, same-team overlap and cross-team overlap boxes, and build the calibration
   table. Rerun IMG_0108 and IMG_0109, then the challenge videos.
8. **Submit and freeze.** A valid file by the end of Nov 2, overwritten as things improve. Only the last submission counts. Freeze code
   12 hours before the deadline.

Identity comes before the team penalty on purpose, so a valid submission exists early. The first draft built it afterwards.

## If colour is weak

| Measured on IMG_0108 | Action |
|---|---|
| Cross-team overlap error up to 15% | Full soft penalty |
| 15–30% | Overlapped boxes unsure, rely on offline repair |
| Clear-box error over 5%, or challenge videos disagree over 2× IMG_0108 | No team in matching, colour only per track piece |

These cut-offs come from simulation only.

**Minimum path:** challenge detections, the tracker at the best settings from step 2, `identity.py` with whatever labels exist, and a
valid submission. With no jersey model the file is valid but scores poorly.

## Known weak points

- `cdt=20` rests on one game with RF-DETR, turning ambiguity off on perfect labels, and the noise results on one incomplete seed.
- The noise model is simple. It flips same-team overlaps, where colour shouldn't fail, never copies the occluder's colour, and assumes
  perfect non-player removal.
- Validation can't reveal camera shift. IMG_0108 is 4K end-line and the challenge is 720p oblique, so step 7 can only use indirect checks:
  colour disagreement within tracks and team counts per frame.
- AssA penalises clean fragments, while TI-HOTA scores identity per row, so higher purity may be worth more than AssA suggests.
- Team labels don't fix same-team swaps. Those depend on jersey reading.

## Open decisions

| Decision | Recommendation |
|---|---|
| Who owns identity assignment and the submission writer? | Tracker side, in `src/tracker/` |
| Which detector checkpoint? | The public RF-DETR Large weights, because `docs/10` warns that fine-tuned models may not carry over to the challenge camera |
| Is colour calibration held out? | Yes, calibrate on a model that hasn't seen IMG_0108, so the error rates behind the fallback table are honest |
| Is ViTPose acceptable under the NDA? | The user's call. This is the model the user chose, and it runs through `transformers` with no `mmpose` installed, but the checkpoint came from an mmpose-based repo. `docs/10` still lists this as open |
