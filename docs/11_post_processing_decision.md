# Post-processing findings, and why we use none for now

Decided 2026-11-01. All numbers are from IMG_0106 with RF-DETR Large unless stated.

**Decision:** no track linking, smoothing or interpolation for now. `src/tracker/` runs causally and writes only what the tracker
observed. We fix the online tracker first, since any post-processing would inherit its errors.

**This doesn't cover identity.** A submission needs team and jersey on every row, or TI-HOTA is 0. Assigning those is still
required before the Nov 3 deadline, and it may need whole-track voting, which is itself a form of post-processing. If so,
this decision has to be revisited for that step.

## Why not now

1. **Half of the identity breaks were real swaps, not gaps.** Of 415 breaks in the first version: 156 takeovers and 49
   handovers (205 swaps), 129 restarts, 81 duplicates. Stitching pieces can repair restarts and duplicates, but a swap inside a
   piece gets carried into whatever it is linked to.
2. **Motion can't settle who is who after an overlap.** 97% of breaks involved overlap with another player, and teammates
   leave an overlap with no motion evidence of who is who.
3. **The online fixes changed the input a linker would see.** Track IDs fell from 253 to 67 and restarts from 129 to 25,
   so a linker tuned on the old tracks would have been tuned on the wrong problem.
4. **The Kalman filter's positions gained nothing, so smoothing is unproven.** Filtered and raw positions scored 0.787
   and 0.789. Backward smoothing might do better, but that is untested.
5. **The rules question was open when this was decided.** The organizers have since confirmed that automatic offline
   processing is allowed (`competition/organizer_emails.md`, Email 5), so this reason no longer applies.
6. **We were measuring with a biased score.** TI-HOTA with one perfect label per track rises as tracks get shorter, so it
   can't judge linking alone. Changes are now judged on AssA, IDF1, wrong-player frames, restarts, fragmentation and coverage.

## Methods reviewed, none run on our data

| Method | Uses | Finding |
|---|---|---|
| AFLink (StrongSORT) | Motion only | Stock checkpoint isn't trained on court coordinates. Code is GPL-3.0, which matters because our code ships publicly |
| Two-sided Kalman merge (`object_tracking_paper/scripts/global_track_restitch/mahalanobis_merge.py`) | Motion only | Cheapest to try, no training. Built for image boxes, no recorded results found |
| C-TWiX | Coordinates only | Transformer, reported on DanceTrack. Too heavy for the time left |
| OCAFI | Motion only | Combines the same parts, aimed at settings like surveillance |
| GTA (gta-link, MIT) | OSNet appearance | Same bibs within a team are likely its weak spot. No imports of the excluded libraries found. Checkpoint training data not checked |
| Context-conditioned motion models (CVPR 2013) | Mostly motion | Basketball and hockey, older hand-built work |

No sports-specific linker based purely on motion turned up in our searches. Those searches were not exhaustive.

## When to come back to this

- Fixed-roster linking on court coordinates is still the likely first choice. Training pieces can be cut from the court positions of
  all 8 training games, which needs no videos.
- Reconsider once online changes stop helping and the identity step exists. The organizers have answered: it is allowed.
- Measure wrong links as well as score. A linker that joins two different players makes things worse while appearing to help.

## What changed in the online tracker

Each change was tested in the order below, starting from the first version. With every switch off and `cdt=3`, the new
code reproduced the first version's output byte for byte.

| Change | Result | Kept? |
|---|---|---|
| Counter now means frames since last observation (old `cdt=5` is new `cdt=3`) | Same behaviour, clearer meaning | Yes |
| Retain tracks longer | Restarts fall, but swaps rise: 205 at `cdt=3`, 276 at 8, 429 at 20, 783 at 60 | Yes, `cdt=8` |
| Stop writing coast rows | Box metrics unchanged, TI-HOTA equal or slightly better | Yes |
| Independent Mahalanobis gate | No effect at 30 or above with calibrated noise, tighter is worse (at `cn=3`, gate 13.8 gives 87 IDs vs 50) | No, off |
| Noise by detection confidence, times 3 | IDs 189 to 50, IDF1 15.4 to 24.6, but swaps 276 to 403 | Yes |
| Motion noise `wx = wy` | 25 balanced best. GT acceleration variance is about 24 m/s^2 squared at x, 158 at y, inflated by box jitter | Yes, 25 |
| Don't commit ambiguous matches (margin 1.0) | Swaps 403 to 247, but coverage 0.953 to 0.931 | Yes |
| Recently-seen tracks match first | No meaningful change | No, off |
| Confirm tracks after 2 hits, not 3 | No clear gain | No, stays 3 |

The review's measured detail differed in one place: it gave 27.2 cm median error for low-confidence boxes, while we measured
32.2 cm below confidence 0.7 and 12.1 cm at or above it. The conclusion holds. Its other code claims checked out.

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

Not every number improved. Swaps rose, and wrong-player frames rose about 6x, since one wrong link now lasts longer. Results
also move sharply between neighbouring settings, because one swap in a long track can shift thousands of frames.

## Limits

- Settings were tuned on IMG_0106, the only game with detections. No other game has confirmed the gain.
- With ground-truth boxes on all 8 games (no detector errors), old and new settings are about level: AssA 22.0 vs 20.9, IDF1 33.3 vs
  31.6, swaps 46 vs 33, restarts 14 vs 44. So the gain seems tied to handling real detector noise, which still needs checking on
  other games.
- Even with perfect boxes, motion alone averages only AssA of about 21 and box HOTA of about 45. Colour and jersey are needed.
