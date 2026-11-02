# Team colours in the tracker: what was built, what was tested, what won

Written 2026-11-02. Follows [13_end_to_end_plan.md](13_end_to_end_plan.md) (the plan) and
[14_team_colours.md](14_team_colours.md) (the colour labels). Code is in `src/tracker/`, tests in `tools/tests/`.

## Result in one paragraph

Feeding the colour labels into UCMCTrack as a soft cost penalty, with NMS before tracking, takes the oracle TI-HOTA from 0.596 to
0.758 averaged over the 7 training games (IMG_0105-0111), and improves every game by at least 0.107. Cross-team wrong rows fall 98%
(7,472 to 182 per game) and purity rises from 0.844 to 0.979. The **real** TI-HOTA (no GT identity at all) stays near 0.18-0.21,
because jersey numbers are still guessed: the tracks are clean, the numbers on them are not. A jersey reader is the remaining lever.

**Settings to use:** current tracker settings (cdt 8, ambiguity 1) plus `--labels <file> --team-scale 6 --nms-iou 0.7`.

## What was built

| File | What it does |
|---|---|
| `src/tracker/run_ucmc.py` | `--labels` loads a `src/team` label file (checked against the detection file's sha256), joins on `det_row`, gives each box a team call (+1 A, -1 B, 0 none) and evidence (logit of the call being right). New params: `team_scale`, `q_clear`, `q_overlap`, `team_cap`, `team_decay`, `team_veto`, `team_birth`, `team_learn_overlap`, `nms_iou`, `q_table`. With labels, output gains columns `team, belief, det_row` |
| `src/tracker/ucmc.py` | Team penalty in `cost()`, per-track belief (`add_team`), optional veto, optional birth block. All off by default |
| `src/tracker/team_rates.py` | Measures how often colour calls are right, clear vs overlapped, with leave-one-video-out |
| `src/tracker/overlap_rates.py` | Same, by IoU band and the neighbour's call (tested, lost, see below) |
| `src/tracker/sweep_team.py` | Runs variants x 2 settings x 7 games in parallel and summarises. `--max-frame` for a smoke test |
| `src/tracker/cross_team.py` | Finds cross-team error episodes on a training video and says how they began (GT used only to describe) |
| `src/tracker/annotate_team.py` | Video with track team, each box's own call, and red X on cross-team errors |
| `src/tracker/identity.py` | Tracks to submission: split at team change-points, team per piece by vote, 3 roster numbers per team with no same-time clashes, every frame filled. `jersey_scores` hook for a reader. `--score` grades a training video |
| `src/tracker/make_submission.py` | Writes IMG_0114.txt and IMG_0115.txt and runs `tools/validate_submission.py` |
| `tools/tests/test_ucmc.py` | 16 tests, 6 new for the team logic. `tools/tests/test_identity.py`: 5 tests |

**Guarantee:** with no labels, IMG_0106's output is byte-identical to `outputs/tracks_ucmc/IMG_0106__rfdetr-large.txt` (73,790
rows), and identical again with labels but `team_scale 0`.

## How the team penalty works

The association cost is Mahalanobis² + logdet, roughly -2 log-likelihood, so team is added on the same scale:

```
q      = chance this box's call is right (from measured error rates)
pi     = sigmoid(call x track belief)          # how much the track agrees with the call
penalty = team_scale x max(0, -2 log(2 (q pi + (1-q)(1-pi))))
```

No call or a track with no belief adds 0. Agreeing adds 0 (no reward, so team alone never pulls a box in). A sure track of the other
team adds about `team_scale x 2 logit(q)`: about 75 at x6 for a clear box, far above the cost limit of 20, so in practice a block.
For an overlapped box (q 0.939) it is about 24 at x6.

Track belief = log-odds of team A, adding `call x evidence` for each matched box, capped at +-10. GT is never an input.

## Label error rates (src/tracker/team_rates.py)

On RF-DETR Large boxes, boxes matching a GT player, pooled over IMG_0105-0111:

| Box | Calls right | Leave-one-video-out |
|---|---:|---:|
| Clear (no other box at IoU >= 0.2) | 99.93% | 99.92-99.93% |
| Overlapped | 93.9% | 93.7-94.2% |
| of which same-team overlap | 99.8% | |
| of which cross-team overlap | 86.2% | |

These are the defaults `q_clear 0.9993`, `q_overlap 0.939`. 14.9% of calls are on boxes matching no GT player (loose boxes and
non-players) and aren't in these rates.

## Experiments, in order

All runs use identical RF-DETR Large detections. Oracle TI gives each whole track its true identity, so it measures track quality
and is a ceiling, not a score. "Wrong rows" are box rows on a player other than the track's main one, split by whether that player
is on the other team.

**1. Penalty strength, veto, birth block** (`outputs/reports/team_sweep.jsonl`, `team_sweep_s34.jsonl`). Current settings, mean of 7:

| Variant | Oracle TI | Purity | AssA | Cross-team rows | Same-team rows |
|---|---:|---:|---:|---:|---:|
| off | 0.596 | 0.844 | 10.2 | 7,472 | 2,564 |
| x1 | 0.661 | 0.899 | 12.7 | 5,287 | 1,215 |
| x2 | 0.690 | 0.924 | 12.9 | 3,536 | 1,346 |
| x3 | 0.713 | 0.943 | 13.6 | 2,684 | 974 |
| x4 | 0.727 | 0.954 | 13.6 | 1,707 | 1,189 |
| x1 + veto | 0.662 | 0.900 | 12.5 | 5,379 | 1,044 |
| x1 + birth block | 0.637 | 0.889 | 13.3 | 5,645 | 1,408 |

Veto and birth block add nothing or cost coverage. Looser settings (cdt 20, ambiguity 0) were worse on oracle TI throughout,
although the earlier perfect-label test had favoured them.

**2. Where the remaining cross-team errors came from** (`cross_team.py` on IMG_0107 at x4, 4,099 rows, 81 episodes). 12 long
episodes held 95% of the rows. During them, 96% of colour calls named the player actually in the box, so labels were right and the
track was wrong. The mechanism, seen in track 185 at frames 9089-9098: two players of different teams overlap, the box on the
track's own player gets the other team's colour a few frames running (overlapped calls carry 2.7 log-odds each, cap 10), four such
calls flip the track's belief, and the penalty then helps it slide onto the opponent. Mostly in the near half of the court.

**3. Fixes for that** (`team_sweep_v2.jsonl`), current settings, mean of 7:

| Variant | Oracle TI | Purity | AssA | Cross-team rows | Same-team rows |
|---|---:|---:|---:|---:|---:|
| x4 | 0.727 | 0.954 | 13.6 | 1,707 | 1,189 |
| x4, cap 20 | 0.732 | 0.958 | 14.4 | 1,183 | 1,504 |
| x4, overlapped boxes count 0 in belief | 0.741 | 0.966 | 14.6 | 211 | 2,042 |
| x4, overlapped boxes count 1/4 | 0.743 | 0.967 | 15.0 | 320 | 1,817 |
| x6 | 0.749 | 0.972 | 13.6 | 180 | 1,606 |

A stronger penalty alone does as well as the belief changes.

**4. Overlap-weighted trust** (`overlap_rates.py`, the idea of trusting a call less the more it overlaps and depending on the
neighbour). Measured accuracy, pooled over 7 games, stable leave-one-out:

| Neighbour's call | IoU 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8+ |
|---|---:|---:|---:|---:|
| opposite | 99.6% | 93.8% | 74.5% | 57% (n=12) |
| same | 97.9% | 92.6% | 87.3% | 90.1% |
| none | 99.2% | 93.1% | 76.9% | 83% (n=52) |

Used as `q_table`, it **lost**: cross-team rows rose to 655 (x4) and 1,813 (x6, loose), because it weakens the penalty exactly
where the guard is needed. Not used.

**5. NMS before tracking.** Evidence first, on IMG_0107 within +-3 frames of its 199 cross-team switch moments vs all frames:
duplicate boxes (IoU >= 0.5 with a higher-confidence box) 1.12 vs 0.52 per frame, frames with a duplicate 64% vs 34%, boxes matching
no player 3.18 vs 2.36, and the box switched onto had confidence < 0.7 in 41% of switches vs 6% of rows. Then the sweep
(`team_sweep_v3.jsonl`), mean of 7:

| Variant (current settings) | Oracle TI | Purity | AssA | IDF1 | Cross-team rows | Same-team rows |
|---|---:|---:|---:|---:|---:|---:|
| off | 0.596 | 0.844 | 10.2 | 20.7 | 7,472 | 2,564 |
| NMS 0.5 only, no team | 0.585 | 0.835 | 9.5 | 19.0 | 8,518 | 2,079 |
| NMS 0.7 only, no team | 0.605 | 0.852 | 10.5 | 20.3 | 8,031 | 1,464 |
| x6 | 0.749 | 0.972 | 13.6 | 24.0 | 180 | 1,606 |
| x6 + NMS 0.5 | 0.759 | 0.980 | 14.2 | 24.8 | 203 | 1,065 |
| x6 + NMS 0.6 | 0.755 | 0.977 | 13.8 | 23.5 | 247 | 1,247 |
| **x6 + NMS 0.7** | **0.758** | **0.979** | **14.2** | 24.5 | 182 | 1,194 |
| loose best (x6 + NMS 0.6) | 0.729 | 0.958 | 17.0 | 28.4 | 265 | 2,505 |

NMS alone barely moves anything (0.5 loses 0.011, 0.7 gains 0.009), and without team labels it raises cross-team rows. With team
labels it cuts same-team rows by 26%. NMS 0.7 is chosen over 0.5 because it never lowers oracle TI
against x6 on any game (0 to +0.037), while 0.5 loses on 3 games.

## The chosen tracker, in counts

TrackEval box metrics (IoU 0.5), totals over IMG_0105-0111:

| | Off | x6 + NMS 0.7 |
|---|---:|---:|
| Tracks | 601 | 572 |
| ID switches | 1,306 | 1,019 |
| Fragmentations | 8,017 | 7,110 |

About 82 tracks per game for 6 players: tracks are pure but still short. `identity.py` gives each piece a number. ID switches here
include a new track taking over a player, so they overlap with fragmentation; cross-team swaps are a small part of what remains.
Fragmentations are high because coasting tracks write no rows, so every short miss counts.

## Real TI-HOTA (full pipeline, nothing from GT)

RF-DETR boxes -> tracker -> colour labels -> `identity.py` with guessed numbers:

| Game | Labels, no penalty | x2 | x6 + NMS 0.7 | Random number shuffles, mean (x6 + NMS) |
|---|---:|---:|---:|---:|
| IMG_0105 | 0.208 | 0.146 | 0.087 | 0.186 |
| IMG_0106 | 0.239 | 0.244 | 0.260 | 0.160 |
| IMG_0107 | 0.233 | 0.211 | 0.215 | 0.170 |
| IMG_0108 | 0.281 | 0.236 | 0.135 | 0.153 |
| IMG_0109 | 0.197 | 0.366 | 0.216 | 0.152 |
| IMG_0110 | 0.158 | 0.095 | 0.194 | 0.181 |
| IMG_0111 | 0.139 | 0.092 | 0.149 | 0.156 |
| **Mean** | **0.208** | **0.199** | **0.179** | |

These move by +-0.1 per game with the number guess, so the differences are noise. With team right and numbers ignored, IMG_0106
(first 3,000 frames) scores about 0.79; the best of all 36 number permutations there is 0.794. **The gap is jersey numbers.**

## Commands

```bash
# track one video with team labels (chosen settings)
uv run python src/tracker/run_ucmc.py DETS.txt REGISTRATION.csv OUT.txt --labels LABELS.csv.gz --team-scale 6 --nms-iou 0.7
# identity and score on a training video
uv run python src/tracker/identity.py OUT.txt --video IMG_0106 --roster data/Training/games_roster.csv --n-frames N --out SUB.txt --score
# submission for the challenge videos
uv run python src/tracker/make_submission.py OUT_DIR --tracks IMG_0114=T14.txt IMG_0115=T15.txt
# sweeps and analysis
uv run python src/tracker/sweep_team.py [--videos ...] [--variants ...] [--max-frame 3000]
uv run python src/tracker/cross_team.py IMG_0107 TRACKS.txt --detections DETS.txt
uv run python src/tracker/annotate_team.py IMG_0107 TRACKS.txt OUT.mp4 [--start F --frames N]
```

Outputs: tracks and submissions in `outputs/team_tracker/` (`*__s6_nms7*.txt` is the chosen setting), sweep results in
`outputs/reports/team_sweep*.jsonl`, error rates in `outputs/reports/team_rates.json` and `overlap_rates.json`, video
`outputs/overlays/IMG_0107__team_s4.mp4` (x4, before NMS).

## Still open

- **Jersey numbers.** The biggest gain. Plug a reader's per-track scores into `identity.py`'s `jersey_scores`. The other session's
  reader uses ViTPose crops, which needs the OpenMMLab/NDA question settled.
- **Challenge detections.** IMG_0114 and IMG_0115 need the detection notebook run, then `python -m src.team.run` for labels.
- **The challenge camera is new** (720p, different angle). Error rates and settings come from training only.
- **IMG_0104** has no detections (decoded 11,376 of 11,400 frames; the detector now warns instead of failing).
- The video should be redrawn with the chosen setting (x6 + NMS 0.7).
- The perfect-label diagnostic in `sweep_team.py` skips IMG_0110's stray GT track 0.
- Nothing is committed.
