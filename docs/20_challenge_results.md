# Challenge videos: first full run and what it showed

Written 2026-11-03. This is the first end-to-end run on the two challenge videos, IMG_0114 and IMG_0115, using the
pipeline and kept settings scored in [18_jnr_results.md](18_jnr_results.md) and the steps in
[19_challenge_run.md](19_challenge_run.md). There is no ground truth for these videos, so every check here compares
against the training game with the same matchup:

| Video | Same matchup in training | Teams |
|---|---|---|
| IMG_0114 | IMG_0108 (held out) | A yellow 6, 9, 10 vs B green 6, 9, 11 |
| IMG_0115 | IMG_0109 | A purple 4, 10, 12 vs B pink 1, 3, 5 |

**Summary.** Weights, colours, jersey reads and the submission format are all fine. Two problems remain:

1. **A non-player on the left sideline is given a player identity.** On IMG_0115 this is serious: a bib-less person in a
   black shirt was submitted as B1 for 7,904 rows, and the real B1 lost its number. The cause is the order in which
   `jnr_tracks.assign` places pieces, not jersey reading (see "Root cause").
2. **Boxes merge, and tracks fragment much more than in training,** because the challenge players are smaller.

## 1. What was run

### Colab (`colab/jnr_test_videos.ipynb`, code version 0ddd8626)

The detector ran on every frame of both videos, then ViTPose crops and jersey reads on every 2nd frame.

| | IMG_0114 | IMG_0115 |
|---|---|---|
| Frames decoded | 11,536 | 11,823 |
| Detections given jersey reads | 47,830 | (log not pasted) |
| Jersey reading time | 1,069 s, about 20 min per video in total | similar |

`reports/weights_used.json` shows the same weights on both videos, matching the scored training results:

| Model | sha256 |
|---|---|
| Detector | `471b3fff` (as expected) |
| Readability head | `4d403c88`, the checkpoint pinned in docs/16 §5.2 |
| ViTSTR | `7319ecec` (as expected) |

The zip was unpacked into `outputs/colab_jnr/jnr_test_0ddd8626/`, and the detection and JNR files were copied to
`outputs/detections_ft/` and `outputs/jnr/`.

### Mac (exactly as in doc 19, section 2)

1. Colour labels: `python -m src.team.run <V> --boxes outputs/detections_ft/<V>__rfdetr-small-ft-471b3fff__720p.txt`.
2. Tracker: `src/tracker/run_ucmc.py` with the kept settings (`--team-scale 6`, guard IoU 0.4, front 20 px, team lock 30).
3. Stitching and submission: `src/tracker/jnr_tracks.py ... --submission outputs/submission/<V>.txt --n-frames 11536`
   for IMG_0114, and `11823` for IMG_0115.
4. Validation: `tools/validate_submission.py outputs/submission`, without `--frames`. Result: **0 errors**. The only
   warnings are frames without exactly 6 rows: 3,254 on IMG_0114 and 1,835 on IMG_0115.
5. Overlays: `src/tracker/annotate_jnr.py`.

**Workaround.** `annotate_jnr.py` defaults to the training video (`.MOV`) and the training court file. For the challenge
videos it needs `--video "data/Final challenge/<V>.MP4" --registration "data/Final challenge/court_registration.csv"`.
Every other step found the challenge roster and court file by itself.

## 2. Numbers against the same matchup in training

### IMG_0114 against IMG_0108

| Metric | IMG_0114 | IMG_0108 |
|---|---|---|
| Detections per frame (all / conf ≥ 0.5) | 11.89 / 7.55 | 8.13 / 6.00 |
| In-court detections per frame: <6 / =6 / >6 | 35.4 / **58.7** / 5.9% | 12.4 / 86.9 / 0.7% |
| Median confidence; share in [0.3, 0.7) | 0.775; 15.1% | 0.881; 2.6% |
| Detections with conf ≥ 0.5 dropped as off court | 24.4% | 1.3% |
| Colour calibration: warnings, gap, gap/sigma | none, **29.1**, 4.39 | none, 16.3, 3.66 |
| Detections with no colour label | 10.6% | 0.3% |
| Reads with elig ≥ 0.5 | 47.8% | 52.6% |
| Mean top probability (readable reads); share ≥ 0.9 | 0.990; 97.4% | 0.992; 97.9% |
| Yellow reads 6 / 9 / 10 | 35 / 33 / 30% | 34 / 34 / 32% |
| Green reads 6 / 9 / 11 | 28 / 33 / 39% | 27 / 32 / 41% |
| Tracker tracks; median length; tracks under 30 frames | **105**; 134; 35 | 36; 727; 6 |
| Tracker rows with a team label (detection / belief) | 77.9 / 95.8% | 92.1 / 99.9% |
| Team flips | 0 | 0 |
| Stitching pieces: vote / bridge / conflict / none | 75 / 8 / 4 / 28 | 27 / 3 / 1 / 6 |
| Unnamed rows; output ids | 5,851; **38** | 146; 13 |
| Submission rows per frame | 5.66 | 5.83 |
| Coverage per identity | 91.6–96.4% | 95.7–98.7% |
| Coverage dropouts | 1,123 | 347 |
| Duplicate identities in a frame | 0 | 0 |
| Jumps over 100 cm in one frame (A6 / A9 / A10 / B6 / B9 / B11) | 7 / **76** / 5 / 1 / 12 / 30 | 0 / 19 / 0 / 1 / 4 / 4 |

### IMG_0115 against IMG_0109

| Metric | IMG_0115 | IMG_0109 |
|---|---|---|
| Detections per frame, conf ≥ 0.5 | 6.35 | 5.98 |
| Frames with <6 / =6 / >6 detections (conf ≥ 0.5) | 3.9 / **59.2** / 36.9% | 2.7 / 96.4 / 0.9% |
| Median confidence, all detections | **0.36** | 0.89 |
| Median box height | 94 px (720p) | 265 px (4K) |
| Colour calibration: warnings, gap | none, **33.0** | none, 20.7 |
| Detections with no colour label | 14.3% | 1.9% |
| Reads with elig ≥ 0.5 | 55.1% | 54.5% |
| Mean top probability (readable reads) | 0.986 | 0.991 |
| Reads whose top number is on the roster | 96.1% | 99.7% |
| Number's team agrees with colour team | 98.6% | 99.1% |
| Tracker tracks; median length | 58; 831 | 43; 871 |
| Tracker rows with a team label | 73.5% | 92.5% |
| Frames with more than 6 tracker rows | **59.7%** | 0.5% |
| Stitching pieces: vote / bridge / conflict / none | 43 / 4 / **10** / 8 | 36 / 5 / 1 / 4 |
| Unnamed rows; output ids | **14,140 (18.3%)**; 24 | 724 (1.1%); 11 |
| Submission rows per frame; frames with <6 rows | 5.83; 15.5% | 5.87; 11.6% |
| Coverage per identity; longest gap | 96–98%; 3 s | 96–99%; 4 s |
| Jumps over 1 m in one frame (B1 / B3 / B5) | **98 / 71 / 65** | 1 / 0 / 4 |

### Track counts

| Stage | IMG_0114 | IMG_0115 | Total |
|---|---|---|---|
| Tracker tracks | 105 | 58 | 163 |
| After stitching (current) | 38 | 24 | 62 |
| Named player ids | 6 | 6 | 12 |
| Unnamed pieces (ids 100+) | 32 | 18 | 50 |
| Identities in the submission | 6 | 6 | 12 |

Training games end with 9–15 ids after stitching.

## 3. What worked

- **Correct weights everywhere.** The readability head is the pinned `4d403c88`, the one every scored training result used.
- **Team colour.** The colour gap is wider than in training (29.1 and 33.0, against 16.3 and 20.7), with no calibration
  warnings, and there are no team flips on IMG_0114.
- **Jersey reading carried over to the new camera.**
  - Mean top probability is 0.986–0.990, and the read distributions match IMG_0108 and IMG_0109 almost exactly.
  - Cross-team reads of 10 and 11 on IMG_0114 are under 1.3%.
  - Number crops are the same size as in training (median about 32 px), so the smaller players don't hurt reading.
  - The worry that unseen numbers might not read didn't come true: all 9 roster numbers were in training.
- **Shared numbers 6 and 9 on IMG_0114 don't swap.** A6 and B6 each have 7 or fewer large jumps.
- **The submissions are well formed.**
  - Frame counts are right (11,536 and 11,823).
  - No identity appears twice in a frame.
  - The validator reports 0 errors.
  - Each identity covers 92–98% of frames.
- **Most people off court are handled.**
  - On IMG_0114, three purple-bibbed people on the grass (the next game's players) are detected at 0.81–0.91 in 91% of
    frames. Their feet are 1.1–2.4 m outside the court, beyond the tracker's 1 m margin, so none of them reach a track.
    The reader does see their bibs, with 1,796 readable reads of "4" or "12", but these never feed an identity.
  - On IMG_0115, the yellow-bibbed bench people are not detected at all.

## 4. What did not work

### 4.1 A non-player gets a player identity (serious on IMG_0115)

- **IMG_0115: a bib-less person in a black shirt on the left sideline.**
  - Where: image x ≈ 250–290. Their feet project to court position (1412, 883) cm, about 0.9 m inside the court edge,
    so the tracker's court filter keeps them.
  - How often: 5,653 detections at conf ≥ 0.5 in that spot, spread over tracks 13, 38, 50, 57, 65, 79, 117, 136 and 170
    (about 8,461 rows).
  - **Effect: 7,904 submission rows of B1 are this person.** B1 alternates between them and the real player 70 times,
    which causes the B1, B3 and B5 jumps. The real B1's pieces became the 10 conflicts, and piece 19 went out as B5 for
    1,856 rows.
  - In the overlay at 66.7 s the black shirt is labelled "B JNR 1" while the real B1 is "trk 104".
- **IMG_0114: a referee in the same spot** (black shirt, red shorts).
  - Low-confidence detections make about 10 long unnamed tracks. Track 7 leans slightly yellow, so it went out as
    **A9 for 235 rows**.
  - That causes A9's 10–11 m one-frame jumps between frames 348 and 2663.

### 4.2 Root cause: the order `assign` places pieces in

`assign` places pieces strongest first, with strength = (reads used) × (top score − second score).

1. **Unreadable reads count fully.** A piece with no readable reads (elig ≥ 0.5) falls back to voting on all its reads,
   and they count as much as readable reads toward strength.
2. **Unreadable reads still give a confident score.** The score is renormalised over only the 3 roster numbers of the
   piece's team, so garbage reads still look certain.

The result on IMG_0115, B1 pieces in the order `assign` handles them:

| Piece | Who | Readable reads | Reads used | B1 score | Strength | Rank | Result |
|---|---|---|---|---|---|---|---|
| 79 | sideline person | 0 | all 1,186 | 0.996 | 1177 | 2nd | takes B1 |
| 110 | real B1 | 1,013 | readable | 0.996 | 1006 | 5th | conflict |
| 13 | sideline person | 0 | all 1,432 | 0.786 | 986 | 6th | takes B1 |
| 42 / 19 / 94 / 1 | real B1 | 745 / 561 / 340 / 134 | readable | ≥ 0.99 | 743 … | later | conflict |

This never happens in training, which is why it got through: no training piece is named from unreadable reads alone.

### 4.3 Merged boxes and fragmentation

- **Smaller players.** They are about 94 px tall here, against about 265 px in training. Overlapping players are boxed
  as one, especially when four crowd under the hoop.
- **IMG_0114:** only 58.7% of frames have exactly 6 in-court detections, against 86.9% on IMG_0108. It has 105 tracks
  with a median length of 134 frames, against 36 and 727.
- **Conflict pieces on IMG_0114** cause jumps:
  - pieces 115, 128 and 129 produce B11 jumps at frames 8304–8344;
  - an A10 jump of 500 cm at 10933;
  - A6 jumps of up to 458 cm at 11145–11190.
- **Lower detector confidence.** The median on IMG_0115 is 0.36, against 0.89 on IMG_0109. There is no blur or
  camera-angle augmentation in detector training (RF-DETR defaults only).

## 5. Tested fix (scratch, no repo code changed)

`piece_table` was wrapped in a scratch script (`outputs/reports/challenge/alloc.py`) and every video was re-stitched and
re-exported.

| | Current | Readable pieces placed first | Unreadable reads never name a piece |
|---|---|---|---|
| Training TI-HOTA (mean of 8) | 0.9026 | 0.9026 | 0.9026 |
| Training TI-HOTA per game | – | identical on all 8 | identical on all 8 |
| IMG_0115 conflicts | 10 | 9 | **2** |
| IMG_0115 rows at the sideline spot | 7,926 (B1 7,904) | 900 | 927 |
| IMG_0115 B1 rows at the spot | 7,904 | 184 | 184 |
| IMG_0114 conflicts | 4 | 4 | 3 |
| IMG_0114 A9 rows at the referee spot | 235 | 235 | 235 |
| Output ids (IMG_0114 / IMG_0115) | 38 / 24 | – | 42 / 23 |

- **"Unreadable reads never name a piece" is the preferred rule.** It costs nothing on training, gives the real B1 its
  number back, and cuts IMG_0115's conflicts from 10 to 2.
- **Rows at the spot are left over** (B5 686, B1 184, B3 35 and A12 22 on IMG_0115; A9 235 on IMG_0114). These now come
  from the export step, not the vote: an unnamed piece still gets its colour team's free number by elimination
  (doc 19, "Unknown identities"). Some of them may be real players passing that spot; a 120 cm radius is only a proxy.

**Open policy question:** should a piece with zero readable reads get a guessed identity at export? On training,
guessed rows were 98.9% right, but those pieces had readable reads. This needs a count of guessed training rows that
come from pieces with zero readable reads before deciding.

## 6. Moments to watch in the overlays (29.97 fps)

**IMG_0114** (`outputs/overlays/IMG_0114__ft_guard_lock30__jnr.mp4`)
- 21.6–22.6 s (f648–677) and 33.2–34.0 s (f996–1019): A9 hops between the real A9 and the referee ("trk 100").
- 243.5–246.5 s (f7298–7387): the worst stretch, 3.9 rows per frame, four players under the hoop; A9 and B11 missing.
- 277.1–278.4 s (f8304–8344): B11 switches to trk 115, and a merged yellow and green box is labelled A10.
- 364.8 s (f10933), then 370–373 s (f11087–11190): an A10 jump of 500 cm, then A6 jumps to trk 129.
- 382–385 s (f11447–11536): low coverage, and B11 jitters at the end.

**IMG_0115** (`outputs/overlays/IMG_0115__ft_guard_lock30__jnr.mp4`)
- 22.3–24.6 s (f667–737): B1 alternates between the sideline person and the real player.
- 66.7 s (f2000): the sideline person is "B JNR 1" and the real B1 is "trk 104".
- 180–190 s (f5402–5700): a burst of B1 jumps.
- 207.4–209.6 s (f6217–6281): the worst stretch for missing rows.
- 241.9–243.6 s (f7249–7300): A12 and B5 jumps in a crowded cluster.
- 387.4–390.4 s (f11610–11700): a burst of B3 jumps.

## 7. Also checked: the reviewer's five findings (2026-11-02)

| # | Finding | Held? | Did it affect the scored results? | Action |
|---|---|---|---|---|
| 1 | Readability checkpoint mismatch; JNR cache not tied to its weights | Yes | No: all 8 games used `4d403c88` | Pinned `4d403c88`; skip check now compares the weights (commit b71872e) |
| 2 | `--n-frames` taken from the detection file can cut off frames | In principle | No | Doc 19 gives the decoded counts; validate without `--frames` |
| 3 | Reused detections left out of the zip | In the code | No: the detector ran fresh | Reused detections and their source are now saved |
| 4 | JNR join by box, with no detection-cache check | In the code | No: identical to a join on detection row | Not changed (optional) |
| 5 | Unnamed pieces get guessed identities | Yes, but guessing helps | +0.010–0.018 TI-HOTA where it occurs | Policy written into doc 19 |

## 8. Next steps

1. Implement "unreadable reads never name a piece" in `jnr_tracks.py` with a unit test, check that training TI-HOTA is
   unchanged, then rebuild both challenge submissions.
2. Decide the export policy for pieces with zero readable reads (§5), using a count on training.
3. Optionally keep a stationary, bib-less, colourless person out of the tracker. A smaller court margin (`--margin -100`)
   removed most of the sideline rows (7,977 → 694) but lost about 1,000 real rows near the court edge, so it is not used.
4. Make `annotate_jnr.py` find the challenge video and court file by itself.
5. Results are due 2026-11-03, 23:59 AoE. Validate the folder again after any rebuild.

## Files

- Submissions: `outputs/submission/IMG_0114.txt`, `outputs/submission/IMG_0115.txt` (current code, before the fix)
- Overlays: `outputs/overlays/IMG_0114__ft_guard_lock30__jnr.mp4`, `outputs/overlays/IMG_0115__ft_guard_lock30__jnr.mp4`
- Calibration: `outputs/team/<V>__rfdetr-small-ft-471b3fff__720p.calibration.json`
- Tracks and stitching: `outputs/team_tracker/<V>__ft_guard_lock30{,__jnr}.txt`, with `.segments.csv` and `.scores.json`
- Colab results: `outputs/colab_jnr/jnr_test_0ddd8626/`
- Fix test and evaluation scripts: `outputs/reports/challenge/` (`alloc.py`, `alloc_results.jsonl`, `IMG_0114_eval/`, `IMG_0115_eval/`)
