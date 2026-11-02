# Challenge videos: running the full pipeline

Written 2026-11-02. How to produce the submission for IMG_0114 and IMG_0115 with the pipeline scored in
[18_jnr_results.md](18_jnr_results.md): real TI-HOTA 0.907 on the held-out IMG_0108, 0.902 mean over the 7 in-sample games.

## The two videos

| Video | Team A | Team B | Training game with the same matchup |
|---|---|---|---|
| IMG_0114 | yellow 6, 9, 10 | green 6, 9, 11 | IMG_0108 (held out): same rosters, shared 6 and 9, closest bib colours (gap 16.3) |
| IMG_0115 | purple 4, 10, 12 | pink 1, 3, 5 | IMG_0109: same rosters and colours (gap 20.7) |

Every number on both rosters was in the jersey reader's training data, on the same bib colour. The risk is the
camera: it is angled, the files are `.MP4`, and neither the colour labeller nor the reader has run on it.

## 1. Colab: detections and jersey numbers

`colab/jnr_test_videos.ipynb`, built by `colab/build_jnr_test_notebook.py`. Code version 0ddd8626, the same as the
training-game JNR run.

1. Put `IMG_0114.MP4` and `IMG_0115.MP4` in `My Drive/test_videos`.
2. Open the notebook in Colab with an A100 runtime, then Run all.
3. It writes to `My Drive/tracking_results/<video>/0ddd8626/` and downloads `jnr_test_0ddd8626.zip`.

## 2. Mac: everything else

Unzip into its own folder, then copy the files into place:

```
outputs/detections_ft/<video>__rfdetr-small-ft-471b3fff__720p.txt
outputs/jnr/<video>__rfdetr-small-ft-471b3fff__720p__jnr.csv.gz
```

Then, per video (`V=IMG_0114`, then `IMG_0115`):

```
# colour labels (no GT on challenge videos; the run skips scoring)
uv run python -m src.team.run $V --boxes outputs/detections_ft/${V}__rfdetr-small-ft-471b3fff__720p.txt

# tracker, kept setting (docs/15, docs/setbacks)
uv run python src/tracker/run_ucmc.py outputs/detections_ft/${V}__rfdetr-small-ft-471b3fff__720p.txt \
    "data/Final challenge/court_registration.csv" outputs/team_tracker/${V}__ft_guard_lock30.txt \
    --labels outputs/labels/${V}__rfdetr-small-ft-471b3fff__720p.csv.gz --cdt 8 --ambiguity 1.0 --team-scale 6.0 \
    --overlap-raw 1 --q-bins "0:0.9999,0.001:0.9996,0.2:0.9926,0.4:0.9375,0.5:0.8615,0.6:0.8215" \
    --guard-iou 0.4 --guard-front-px 20 --team-lock-frames 30

# JNR stitching and the submission rows (roster from data/Final challenge)
uv run python src/tracker/jnr_tracks.py outputs/team_tracker/${V}__ft_guard_lock30.txt \
    outputs/jnr/${V}__rfdetr-small-ft-471b3fff__720p__jnr.csv.gz --video $V \
    --out outputs/team_tracker/${V}__ft_guard_lock30__jnr.txt \
    --submission outputs/submission/${V}.txt --n-frames <frames in the video>
```

`--n-frames` is the decoded frame count: **IMG_0114 = 11536, IMG_0115 = 11823**. Both match the Colab detector log
and docs/00. To re-check, use `ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames
-of csv=p=0 <video>`. Don't use the last frame in the detection file, which is short if the last frames have no
detections. Don't use metadata `nb_frames` either: it over-counts IMG_0104 by 24.

Then validate the folder with `tools/validate_submission.py` **without** `--frames`, so it checks every frame against
the video's own length.

**Unknown identities.** Pieces the stitcher could not name (ids 100+) are still written. They take their colour team and
the roster number that no named piece of that team holds at those frames. On the training games these guesses were
98.9% right (3,059 of 3,094 rows) and added 0.010–0.018 TI-HOTA where they occur (IMG_0104, 0108, 0109), so they stay.
Use only `jnr_tracks.py --submission` for the challenge. `src/tracker/make_submission.py` predates JNR and ignores the
jersey scores.

**Provenance.** Keep each video's `reports/weights_used.json` from the zip with the submission. It should show detector
`471b3fff`, eligibility `4d403c88` and ViTSTR `7319ecec`.

## Checks without ground truth

Compare these against the training games (docs/18) before trusting the output:

- **Colour calibration:** look for warnings in `outputs/team/<video>__*.calibration.json`, and for the gap between the two
  colours. It was 16.3 on IMG_0108.
- **Reads:** the share of JNR reads with `elig >= 0.5`, and the mean top probability.
- **Stitching:** from the stats line, the number of pieces per `how`, unnamed rows, conflicts and output ids. Training
  games end with 9–15 ids for 6 players.
- **The video:** draw it with `src/tracker/annotate_jnr.py` and watch the shared numbers 6 and 9 on IMG_0114.
