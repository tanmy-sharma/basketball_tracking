# Jersey numbers on tracks: attach, split, stitch

Written 2026-11-02. Follows [15_team_tracker.md](15_team_tracker.md) (the team-aware tracker) and
[16_models_and_training_runs.md](16_models_and_training_runs.md) (the JNR models, section 7.5 for the evidence file).
Code: `src/tracker/jnr_tracks.py`. Tests: `src/tests/test_jnr_tracks.py` (5, synthetic). The existing tracker is unchanged:
this runs after it, on its output file.

## Status

Built and tested on synthetic data only. Not yet run on real tracks, because no JNR output is local yet: it comes from the
`jnr_all_videos.ipynb` Colab run. Every threshold below is a first guess and a command-line flag. Tune them on the training
games, never on IMG_0108 (the held-out test game) or the challenge videos.

First real-data results on all 8 training games, and the `vote_2nd` problem, are in [18_jnr_results.md](18_jnr_results.md).

## Why

The team-aware tracker gives clean tracks (oracle TI-HOTA 0.758, purity 0.979, doc 15), but the real TI-HOTA stays near
0.2 because the jersey numbers on those tracks are guessed. Tracks also fragment: one player becomes several ids over a
game. The JNR evidence fixes both: it names each track, and pieces with the same name are one player.

## Inputs

**Assumption:** every tracker row stores the detection box it matched in that frame, next to its own. Each tracker row then
joins exactly one JNR row by frame and box, with no IoU guessing.

| File | Layout (headerless) | Notes |
|---|---|---|
| Tracks, `ucmc` | `frame,id,court_x,court_y,raw_x,raw_y,kind,conf,bx,by,bw,bh[,team,belief,det_row]` | What `run_ucmc.py` already writes. `bx..bh` is the detection box (`-1` on coast rows); `team` is the box's colour call, +1 A, -1 B, 0 none. Court cm, so a speed check is possible |
| Tracks, `mot` | `frame,id,x,y,w,h,conf,dx,dy,dw,dh[,team]` | Tracker box, then the matched detection box (`-1` if none), full-resolution pixels. `team` is `A`/`B` or `+1`/`-1`/`0`. No speed check |
| JNR | `src/infer/jnr_video.py` output (`*__jnr.csv.gz`) | Must come from the **same detection file** as the tracks. Covers every 2nd frame and boxes with conf >= 0.3 only |

`--format auto` picks `ucmc` when column 7 is `det`/`coast`, else `mot`. The roster comes from
`data/Final challenge/games_roster.csv`, then `data/Training/games_roster.csv`, or `--roster`.

## Steps

1. **Join.** Per frame, each tracker row's detection box is matched to the JNR row with the same box, one to one. The
   largest coordinate gap must be under `--tol` 1 px, because the JNR file keeps 5 significant digits.
2. **Evidence per read.** Identity = (team, number). P(number | team) is the ViTSTR probability renormalised over that
   team's 3 roster numbers. P(team) for one read comes from that box's colour call (`--team-q` 0.95). This keeps the 6 and
   9 that both IMG_0114 teams wear apart.
3. **Split ID switches.** Binary segmentation of a track's eligible reads (elig >= 0.5) on log-probability (floored at
   `--p-floor` 0.01 so one bad crop can't veto). A cut needs a different best identity on each side, at least `--min-side`
   6 reads each side, and a gain of at least `--split-gain` 8 nats over one identity for the whole track. The cut goes
   midway between the last read of one run and the first of the next. Pieces get ids `track + k * 1,000,000`, as in
   `identity.py`.
4. **Vote per piece.** Mean P(number | team) over the piece's eligible reads (all non-degenerate reads if none are
   eligible, which is the fallback rule from doc 16 section 7.3). Multiplied by the piece's P(team) from all its colour
   calls (`--team-nats` 0.2 log-odds each, since neighbouring calls are correlated), then normalised. A piece is labelled
   if it has at least `--min-reads` 3 reads and a top score of at least `--min-score` 0.6.
5. **Assign and stitch.** Strongest pieces first (reads x (top - second)). Each takes its best identity, provided that:
   - no piece already holding it shares more than `--max-overlap` 10 frames with it;
   - in `ucmc` layout, the player could have run between the pieces (`--vmax` 1000 cm/s plus `--slack` 150 cm).
   Otherwise it tries its next best (`vote_2nd`), else stays unlabelled (`conflict`). Pieces with one identity are one id,
   which is the stitch.
6. **Bridge.** An unlabelled piece takes an identity if a piece of that identity ended at most `--bridge-gap` 90 frames
   before it, it fits the overlap and speed checks, and its own colour calls agree on the team. Neighbours on both sides
   beat one side only. It is assigned only when exactly one identity fits.
7. **Rewrite.** Same rows, same columns, same number formatting; only the id column changes, on **every** row of a piece,
   before and after its reads. Identity ids are 1-6 (team A's numbers in roster order, then B's); other pieces get 100 and
   up, in start order. Where two pieces of one id overlap (at most `--max-overlap` frames), the weaker piece's rows in those
   frames are dropped, so no id appears twice in a frame.

## Outputs

| File | Contents |
|---|---|
| `OUT.txt` | The track file with ids rewritten. Downstream tools (`identity.py`, `make_submission.py`) read it unchanged |
| `OUT.segments.csv` | One row per piece: original id, frames, reads, eligible reads, `source` (eligible/all/none), `p_team_a`, best identity, score, second, strength, a score per identity, `identity`, `how` (vote, vote_2nd, bridge, conflict, none), `new_id` |
| `OUT.scores.json` | Roster, per-id number scores (the `jersey_scores` hook of `identity.make_rows`), and run stats |
| `--submission SUB.txt` | Optional, `ucmc` only, needs `--n-frames`: submission rows from `identity.make_rows` with these scores. `--score` grades it on a training video |

The stats line printed at the end gives rows joined to JNR, eligible rows, splits, pieces per `how`, rows identified and
rows dropped by overlap. Check those first on a real run.

## How to run it once the JNR files arrive

Tracks and JNR must use the same fine-tuned detections (`*__rfdetr-small-ft-471b3fff__720p.txt`).

```
python src/tracker/run_ucmc.py <dets> <registration.csv> T.txt \
    --labels outputs/labels/<video>__rfdetr-small-ft-471b3fff__720p.csv.gz --team-scale 6 --nms-iou 0.7
python src/tracker/identity.py T.txt --video <video> --roster data/Training/games_roster.csv \
    --n-frames <N> --out BEFORE.txt --score                                # baseline: numbers guessed
python src/tracker/jnr_tracks.py T.txt <video>__rfdetr-small-ft-471b3fff__720p__jnr.csv.gz --video <video> \
    --out J.txt --submission AFTER.txt --n-frames <N> --score              # numbers from JNR
```

Compare real TI-HOTA before and after on the training games. Then use `OUT.segments.csv` to check the splits and
conflicts against GT on those games (GT used for grading only).

## Tests (synthetic, both layouts)

`src/tests/test_jnr_tracks.py` builds a scene in which:

- A10 is fragmented into two tracks, with an unread track between them;
- one B track switches from 11 to 6 halfway;
- one track is never read.

The tests check:

- every read joins despite 5-digit rounding;
- both A10 fragments and the bridge get one id on every row;
- the switching track is split, with each side relabelled over its whole history;
- the unread track keeps a 100+ id;
- no id appears twice in a frame, and the column count is unchanged;
- a weaker track read as A10 at the same time as a stronger one doesn't take A10;
- the join is one to one;
- the change-point search finds the switch.

## Known limits

- The cut in step 3 sits midway between reads. The real switch is usually at an occlusion inside that gap, and up to about
  half the gap can land on the wrong side.
- In `mot` layout there is no speed check (pixel positions), so stitching relies on time overlap and the number alone.
- Shared numbers (6, 9 on both IMG_0114 teams) depend on the colour calls for the team. With no colour calls, both teams
  get P = 0.5 and only 10 vs 11 separate them.
- JNR is every 2nd frame at conf >= 0.3, so short pieces may have no reads and depend on the bridge.
- ViTSTR leans on bib colour and person appearance (doc 16, cross-colour runs). Whole-track votes were 1.0 on IMG_0108, but
  the challenge camera is angled and untested.
