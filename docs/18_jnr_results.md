# Jersey numbers on real tracks: results

Written 2026-11-02. This is the first real-data run of [17_jnr_tracker_extension.md](17_jnr_tracker_extension.md): attach
the jersey-reader (JNR) reads to the tracks, split them, stitch them. The tracker is the one from
[15_team_tracker.md](15_team_tracker.md), and the setbacks are in [setbacks/team_tracker.md](setbacks/team_tracker.md).

## Summary

- **JNR stitching works.** Real TI-HOTA goes from **0.184 to 0.797** on average over the 7 in-sample games, and from
  **0.005 to 0.892** on IMG_0108, the held-out game. AssA goes from 30.4 to 73.7, and from 38.4 to 81.4 on IMG_0108.
- **The reader is not why some games are weak.** Its raw reads are 96–97% right on every game, IMG_0109 and IMG_0111
  included.
- **The weak games come from one route in the stitcher.** Almost every wrong label was assigned through `vote_2nd`, a piece
  taking its second choice. On those pieces, the piece's own vote was usually correct.
- **Fixing that route (now applied, see "After the fix")** raises the in-sample mean from 0.797 to **0.902** and
  IMG_0108 from 0.892 to **0.907**. IMG_0109 goes from 0.472 to 0.914, IMG_0111 from 0.621 to 0.872. AssA rises from
  73.7 to **85.6** in-sample and from 81.4 to 83.1 on IMG_0108.
- The tables below up to "After the fix" describe the run **before** the fix.

## Setup

| | |
|---|---|
| **JNR run** | Colab run `0ddd8626` on all 8 training games. Copied to `outputs/colab_jnr/jnr_0ddd8626/`, with the read files in `outputs/jnr/<video>__rfdetr-small-ft-471b3fff__720p__jnr.csv.gz` |
| **Detections** | The run's detection files are byte-identical to `outputs/detections_ft/`, so every tracker row joins its read by box |
| **Tracker** | The kept setting, sweep variant `s6_qb_back.4_lock30`: ×6 team penalty, plus the colour safeguard (overlap measured on all boxes, trust by overlap, no call on back or unclear boxes at IoU ≥ 0.4), plus a team lock after 30 frames |
| **Stitching** | `src/tracker/jnr_tracks.py` with its defaults at the time, including `--max-overlap 10` (now 30) |
| **Scoring** | "Before" is `identity.py` with guessed jerseys, "after" is `jnr_tracks.py --submission`. Both are scored against GT with `--score`. Box metrics come from `tracker.diagnose`. GT is used only for grading |

**Caveats:**
- **Held-out vs in-sample.** IMG_0108 is held out from both the detector and the JNR models. The other 7 games are
  in-sample for both, so their numbers are optimistic.
- **Readability checkpoint.** All 8 games used `eligibility.pt` sha256 `4d403c88…`. `weights_used.json` flags it as
  unexpected only because the notebook then asked for `57a986ea…`, the run's epoch-13 file. `4d403c88` is an earlier
  copy of the same run's `best.pt` that Drive kept (docs/16, §5.2). The notebooks now pin `4d403c88`, so these scores and
  the challenge run use the same weights. The detector (`471b3fff`) and ViTSTR (`7319ecec`) hashes are as expected.

## IMG_0104 first

| | Tracker only | + JNR stitch |
|---|---|---|
| AssA | 32.2 | **80.5** |
| IDF1 | 49.5 | **94.9** |
| HOTA (box) | 52.5 | **83.0** |
| DetA | 85.6 | 85.6 |
| Coverage | 0.968 | 0.968 |
| Track ids | 41 | **17** (6 named + 11 unnamed) |
| Real TI-HOTA | 0.158 | **0.908** |

Of the 30 named pieces, 29 carry the right number. The one wrong piece is 18 rows long; see track 103 below.

The annotated video is `outputs/overlays/IMG_0104__ft_guard_lock30__jnr.mp4`, drawn by the new
`src/tracker/annotate_jnr.py`. Each box is labelled "A JNR 10" when its piece has an identity, or "trk 105" when it has
none.

## All 8 games

From `outputs/reports/jnr_all.csv`:
- "Wrong rows" are rows in named pieces whose number differs from the GT player that the original track mostly follows.
  This slightly overcounts on tracks the stitcher split.
- "Unnamed rows" are rows in pieces left as `trk`.

| Game | Real TI-HOTA before → after | AssA before → after | IDF1 before → after | Ids before → after | Named pieces (wrong) | Wrong rows | Unnamed rows |
|---|---|---|---|---|---|---|---|
| IMG_0104 | 0.158 → 0.908 | 32.2 → 80.5 | 49.5 → 94.9 | 41 → 17 | 30 (1) | 18 | 2,261 |
| IMG_0105 | 0.184 → 0.933 | 37.8 → 88.1 | 51.2 → 99.0 | 33 → 10 | 29 (2) | 16 | 99 |
| IMG_0106 | 0.043 → 0.919 | 31.9 → 82.2 | 47.6 → 96.1 | 24 → 12 | 20 (1) | 5,132 | 1,941 |
| IMG_0107 | 0.142 → 0.909 | 34.6 → 85.4 | 48.1 → 98.0 | 33 → 11 | 30 (1) | 4 | 377 |
| IMG_0109 | 0.146 → **0.472** | 28.2 → 52.6 | 45.1 → 75.9 | 43 → 14 | 38 (13) | 12,629 | 2,157 |
| IMG_0110 | 0.334 → 0.820 | 23.4 → 76.3 | 40.9 → 92.7 | 55 → 17 | 53 (7) | 2,113 | 1,429 |
| IMG_0111 | 0.281 → **0.621** | 25.1 → 50.9 | 38.0 → 72.4 | 42 → 14 | 42 (19) | 19,745 | 1,668 |
| **Mean, 7 in-sample** | **0.184 → 0.797** | **30.4 → 73.7** | **45.8 → 89.9** | 38.7 → 13.6 | 34.6 (6.3) | 5,665 | 1,419 |
| **IMG_0108 (held out)** | **0.005 → 0.892** | **38.4 → 81.4** | **57.1 → 95.6** | 36 → 14 | 29 (6) | 848 | 1,351 |

Coverage after stitching is 0.962–0.984 on every game, unchanged from the tracker, so the gains are not from dropping rows.

## The reader is right; the weak games are not its fault

These are raw JNR reads with status `ok` and `elig` ≥ 0.5, matched straight to GT boxes at IoU ≥ 0.5, with no tracker
involved. The share whose top number equals the GT jersey:

| IMG_0104 | IMG_0105 | IMG_0106 | IMG_0107 | IMG_0108 | IMG_0109 | IMG_0110 | IMG_0111 |
|---|---|---|---|---|---|---|---|
| 0.966 | 0.974 | 0.963 | 0.968 | 0.967 | 0.968 | 0.959 | 0.967 |

IMG_0109 and IMG_0111 read as well as the good games, so their low scores come from later in the pipeline.

## Track 103 on IMG_0104

"trk 103" is tracker track 42: frames 3478–4268, 779 rows, really B9. Its reads say B9 with a score of 0.9992 (next best
0.00076), yet it was left unnamed as a `conflict`. What happened:

1. In its last frames, track 42 swaps onto A12: 9 of the 11 frames from 4250 to 4268 are on A12.
2. The real B9 is picked up by track 47 from frame 4248. B9 therefore appears twice in **11 frames**, one more than
   `--max-overlap 10` allows.
3. Track 47 is much stronger (1,792 eligible reads vs 178), so it keeps B9, and all 779 rows of track 42 stay unnamed.
4. That leaves B9 free in frames 4003–4020. Piece 44, which is really A12 and has 18 rows and 5 eligible reads, takes B9
   as its `vote_2nd`. This is IMG_0104's only wrong label.

Allowing more overlap:

| `--max-overlap` | Real TI-HOTA | Track 42 | Piece 44 (really A12) | Track 65 (really A4, 1,340 rows) |
|---|---|---|---|---|
| 10 (default) | 0.9084 | trk (conflict) | B9 (wrong team) | trk (conflict) |
| 20 | 0.9088 | **B9** ✔ | A10 (right team, wrong number) | trk (conflict) |
| 30 | 0.9088 | B9 ✔ | A10 | trk (conflict) |

20 is the smallest value that names track 42. The stitcher already drops the weaker piece's rows in the shared frames, and
those frames were on the wrong player anyway. Track 65 is still a conflict at every value tried.

## Where the wrong labels come from

Rows in named pieces, split by how the piece got its label and whether that label is right:

| Game | `vote`: right / wrong | `vote_2nd`: right / wrong | `bridge`: right / wrong |
|---|---|---|---|
| IMG_0104 | 63,865 / 0 | 0 / 18 | — |
| IMG_0109 | 48,331 / **0** | 113 / **12,595** | 54 / 34 |
| IMG_0111 | 50,284 / 2,249 | 419 / **17,457** | 0 / 39 |

**When a piece takes its own best vote, it is almost always right. Nearly all the wrong rows come from `vote_2nd`.**

In almost every wrong `vote_2nd` piece, the piece's own best vote was the **true** identity. It wasn't given that identity
because another piece held it at the same time, so it took its second choice, however small. For example, IMG_0109 track 1:
- 343 of its 353 eligible reads say 10;
- its best is A10 at 0.979;
- it was assigned A4 through `vote_2nd`, with a score of **0.012**.

Once one piece is wrongly blocked, `vote_2nd` gives it another player's number. That blocks that player's piece, which then
takes a third player's number, and so on. The result is a **rotation within a team**. On IMG_0109: true A10 → A4,
A12 → A10, A4 → A12, exactly the cycle seen in the wrong pieces.

### Root cause and fix

This was traced on IMG_0109 by logging the fit checks in `assign()` (`src/tracker/jnr_tracks.py:257-273`).

1. **Trigger: an end-of-track swap overlaps the next piece of the same player.**
   - Piece 32 (frames 2071–4418) is really B5, and its best vote is B5 at 0.9935.
   - It shares **19 frames** with the next B5 piece, 54 (frames 4400–9082), because of a swap at piece 32's end. That is
     more than `--max-overlap 10`, so piece 32 is blocked from B5.
   - This is the same mechanism as track 103 on IMG_0104, which had 11 shared frames.
2. **Bug: `assign()` falls through to any identity with a score above zero.** Line 266 reads `if sc <= 0: break`, so piece
   32 takes **A4, on the wrong team, with a score of about 0.004**.
3. **Cascade.** Pieces are placed strongest first, so each wrong pick blocks the next piece:

   | Piece | Really | Blocked from it by | Takes instead |
   |---|---|---|---|
   | 18 | A4 | piece 32 | A12 |
   | 4 | A12 | piece 18 | A10 |
   | 1 | A10 | piece 4 | A4 |
   | 6 | A4 | piece 1 | A12 |

   The chain continues the same way. That is the A10 → A4, A12 → A10, A4 → A12 rotation, about 12,600 wrong rows,
   although the reader was right.

**Fixes tested on all 8 games:**
- (a) A fallback choice must itself score at least `--min-score` (0.6). Otherwise the piece is a conflict and stays
  unnamed.
- (b) `--max-overlap 30`. The stitcher already drops the weaker piece's rows in shared frames when it rewrites.

They were tested with a runtime patch (`scratchpad/jnr_fix_test.py`), so the repo code is unchanged. The summary is in
`scratchpad/jnrfix/summary.csv`. Scores are real TI-HOTA from `jnr_tracks.py --score`:

| Game | Current | (a) fallback ≥ 0.6 | (a) + (b) max-overlap 30 |
|---|---|---|---|
| IMG_0104 | 0.9084 | 0.9088 | 0.9088 |
| IMG_0105 | 0.9328 | 0.9326 | 0.9326 |
| IMG_0106 | 0.9186 | 0.9186 | 0.9186 |
| IMG_0107 | 0.9092 | 0.9092 | 0.9092 |
| IMG_0109 | **0.4720** | **0.9134** | **0.9137** |
| IMG_0110 | 0.8197 | 0.8588 | 0.8589 |
| IMG_0111 | **0.6210** | **0.8718** | **0.8717** |
| **Mean, 7 in-sample** | **0.7974** | **0.9019** | **0.9019** |
| **IMG_0108 (held out)** | **0.8920** | **0.9069** | **0.9069** |

How the pieces change, per game, as current / (a) / (a) + (b):

| Game | `vote_2nd` pieces | Conflicts | Output ids |
|---|---|---|---|
| IMG_0104 | 1 / 0 / 0 | 2 / 3 / 1 | 17 / 18 / 15 |
| IMG_0105 | 1 / 0 / 0 | 1 / 2 / 1 | 10 / 11 / 10 |
| IMG_0106 | 0 / 0 / 0 | 2 / 2 / 0 | 12 / 12 / 10 |
| IMG_0107 | 1 / 0 / 0 | 1 / 1 / 0 | 11 / 11 / 10 |
| IMG_0108 | 4 / 0 / 0 | 1 / 2 / 1 | 14 / 14 / 13 |
| IMG_0109 | 11 / 0 / 0 | 4 / 5 / 1 | 14 / 15 / 11 |
| IMG_0110 | 9 / 0 / 0 | 4 / 5 / 2 | 17 / 18 / 14 |
| IMG_0111 | 14 / 0 / 0 | 5 / 4 / 0 | 14 / 13 / 9 |

- **Fix (a) is the one that moves the score.** A piece's identity scores sum to 1, and a piece needs a best of at least
  0.6 to be labelled, so a fallback scoring 0.6 or more is impossible. The fix therefore removes the second-choice route:
  a blocked piece stays unnamed (`trk`) instead of taking a wrong number.
- **Fix (b) cleans up without changing the score.** It resolves most conflicts, so there are fewer unnamed pieces and the
  id count gets closer to 6 plus a few.

**Recommended:** apply both.
- A one-line change at `jnr_tracks.py:266`: `if sc[pos[sid], k] < p.min_score: break`.
- Make `max_overlap` default to 30.

**Testing pitfall:** a module function patched inside multiprocessing workers stays patched for later jobs in the same
worker. The first comparison was contaminated by this and was rerun.

## After the fix

Both fixes are now in `src/tracker/jnr_tracks.py`:
- A fallback identity must score at least `--min-score`, otherwise the piece is a `conflict`.
- `--max-overlap` defaults to 30.

Two tests in `src/tests/test_jnr_tracks.py` cover them. Both fail on the old code and pass on the new:
- `test_blocked_piece_stays_unnamed`
- `test_tail_overlap_still_stitches`

All 8 games were regenerated with the fixed code (`outputs/reports/jnr_all.csv`; the pre-fix table is kept as
`jnr_all_before_fix.csv`). The results match the patched test exactly.

| Game | Real TI-HOTA before → after fix | AssA before → after fix | Ids | Unnamed rows | Conflicts |
|---|---|---|---|---|---|
| IMG_0104 | 0.9084 → 0.9088 | 80.5 → 82.4 | 15 | 1,454 | 1 |
| IMG_0105 | 0.9328 → 0.9326 | 88.1 → 88.0 | 10 | 99 | 1 |
| IMG_0106 | 0.9186 → 0.9186 | 82.2 → 86.6 | 10 | 25 | 0 |
| IMG_0107 | 0.9092 → 0.9092 | 85.4 → 86.1 | 10 | 46 | 0 |
| IMG_0109 | 0.4720 → **0.9137** | 52.6 → **84.7** | 11 | 724 | 1 |
| IMG_0110 | 0.8197 → **0.8589** | 76.3 → **84.9** | 14 | 131 | 2 |
| IMG_0111 | 0.6210 → **0.8717** | 50.9 → **86.8** | 9 | 87 | 0 |
| **Mean, 7 in-sample** | **0.7974 → 0.9019** | **73.7 → 85.6** | 11.3 | 367 | 0.7 |
| **IMG_0108 (held out)** | **0.8920 → 0.9069** | **81.4 → 83.1** | 13 | 1,287 | 1 |

**What's left:** remaining wrong labels are now in-sample IMG_0106 (one long B1/B5 piece) and IMG_0111. Their
"wrong rows" counts are overstated for split tracks, since each piece is credited to its whole original track's main
player. IMG_0108's leftover unnamed rows are mostly one conflict.

## Next steps

1. **Trace the remaining conflicts** (IMG_0104 track 65, really A4; IMG_0108's one conflict), and the IMG_0106 B1/B5
   piece. IMG_0108 is the fair check; tune only on the other 7.
2. ~~Resolve the readability checkpoint mismatch~~: done. `4d403c88` is pinned (see Caveats).
3. **Run the challenge videos** through the same pipeline for the submission.

## Files

- `outputs/reports/jnr_all.csv`: the all-games table.
- `outputs/team_tracker/<video>__ft_guard_lock30*`, per game:
  - `.txt`: tracks;
  - `__jnr.txt`, `__jnr.segments.csv`, `__jnr.scores.json`: the stitched output;
  - `__before_sub.txt`, `__jnr_sub.txt`: the submissions before and after.
- `outputs/overlays/IMG_0104__ft_guard_lock30__jnr.mp4`: the IMG_0104 video.
- `src/tracker/annotate_jnr.py`: the new drawer.
