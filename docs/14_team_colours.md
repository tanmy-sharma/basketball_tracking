# Team colours: what was built, how well it works, and its limits

Written 2026-11-02. Code is in `src/team/`, tests in `src/tests/test_team.py`, and the summary in
[13_end_to_end_plan.md](13_end_to_end_plan.md), under "Team colours".

## What it does

For each box it decides team A, team B, `none` (not a player's colour) or `unsure`. It uses no model or trained weights, only
colour.

```
10 random frames from the middle 90% of the video
  -> their median = empty-court background image (the camera is fixed, so players disappear from the median)
  -> each box: chest region, minus pixels that match the background, minus court blue, minus black, white and grey
  -> median Lab a, b of what is left = the box's colour
  -> calibration: group clear, confident boxes by the nearest of the two roster bib hues, then measure both colours on this video
  -> any box: distance to the two colours -> colour and team, or none or unsure
```

| Piece | Where | What it does |
|---|---|---|
| `crop_colour(crop, whole_player, background)` | `colour.py` | Crop to `ok` with a, b, or to `no_colour` or `degenerate` |
| `calibrate(video, samples, teams, frames)` | `colour.py` | Measures the video's two bib colours and names them from the roster |
| `TeamColours.run(crop, background=...)` | `colour.py` | The run function. Takes a crop and returns colour, team, scores and status |
| `TeamColours.save` and `.load` | `colour.py` | The calibration as JSON, so a video is calibrated once |
| `python -m src.team.run VIDEO --boxes gt\|DETECTIONS` | `run.py` | Calibrate, label every box, score against GT on training videos, draw the video |

Options: `--every N` labels every Nth frame, `--seed` picks the calibration frames, `--seeds N` repeats calibration on N other draws,
and `--video` draws a video showing only team A and team B boxes.

**Outputs**

| File | Contents |
|---|---|
| `outputs/labels/<video>__<boxes>.csv.gz` | The label file from `docs/13`: `frame, det_row, x, y, w, h, p_green, p_yellow, p_pink, p_purple, p_none, status, colour, team, pose_min_conf`, plus a `.sha256` of the detection file |
| `outputs/team/<video>__<boxes>.calibration.json` | Colours, roster mapping, spread, calibration frames, sample counts, warnings |
| `outputs/team/<video>__<boxes>.background.png` | The empty-court image |
| `outputs/team/<video>__<boxes>.metrics.json` | Precision against GT, plus the repeated draws |
| `outputs/team/<video>__<boxes>.mp4` | Team A and B boxes in bib colour, 720p |

**Main settings** (in `colour.py`): chest region 25-75% of box width and 15-50% of height. A pixel belongs to the person if its Lab
distance from the background exceeds 20, and counts as vivid above chroma 15. A box needs 15% person pixels and 15% of those vivid. A box
further than 4 spreads from both colours is `none`, and a winning score below 0.8 is `unsure`. Calibration uses boxes with confidence at
least 0.5 and overlap below IoU 0.2. Labels cover boxes with confidence at least 0.1 whose feet are within 100 cm of the court.

## How we got here

| Step | Finding |
|---|---|
| k-means with k=2 per crop, keeping the brighter or more colourful cluster (the first idea) | Court blue is vivid and gets picked. 84-89% on clear boxes |
| Median of vivid pixels that aren't court blue | Better: 93% on clear boxes in the first test |
| First frame as the colour reference | Matched fixed colours from training. Replaced by 10 random frames, which also cover changes over the video |
| First test showed accuracy falling late in the video | **Wrong:** that test seeked by time, which drifted up to 4 frames from the boxes. With frames picked by index, accuracy is flat |
| Errors on RF-DETR boxes | Most came from grass behind the player: a black shirt in front of grass reads as green, and a side-on purple bib gets outvoted by grass |
| Background removal | Fixed both. Clear-box precision rose from 95-99.9% to 99.8-100% per game, and non-players rejected rose from 64% to 96% |

## Results

**Precision** is the share of team calls on GT-matched players that name the right team. **Coverage** is the share of GT-matched
players that get a team call rather than `none` or `unsure`. Calibration frames are not scored.

**Ground-truth boxes, every 5th frame** (perfect boxes, no non-players, so a best case for colour alone):

| Game | Bibs (A / B) | Precision | Coverage | Clear | Same-team overlap | Cross-team overlap |
|---|---|---:|---:|---:|---:|---:|
| IMG_0104 | purple / green | 97.8% | 98.6% | 100.0% | 99.8% | 87.9% |
| IMG_0105 | yellow / pink | 98.4% | 99.9% | 100.0% | 100.0% | 89.7% |
| IMG_0106 | green / pink | 98.4% | 99.2% | 100.0% | 99.8% | 90.7% |
| IMG_0107 | yellow / purple | 98.1% | 98.9% | 99.96% | 99.8% | 88.9% |
| IMG_0108 | green / yellow | 98.8% | 95.7% | 99.96% | 100.0% | 92.2% |
| IMG_0109 | purple / pink | 98.3% | 99.0% | 99.96% | 99.6% | 89.7% |
| IMG_0110 | purple / green | 97.7% | 99.5% | 99.8% | 99.9% | 89.8% |
| IMG_0111 | pink / yellow | 98.0% | 99.3% | 99.98% | 99.7% | 90.5% |
| **Mean** | | **98.2%** | **98.8%** | **99.96%** | **99.8%** | **89.9%** |

**RF-DETR Large boxes, every frame** (the real input):

| Game | Bibs (A / B) | Precision | Coverage | Clear | Same-team overlap | Cross-team overlap | Non-players rejected | Wrong-team calls, conf ≥ 0.3 | Non-player calls, conf ≥ 0.3 | Worst of 20 draws | Players right by vote |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IMG_0105 | yellow / pink | 98.9% | 99.8% | 99.9% | 97.3% | 85.7% | 96.4% | 0.5% | 0.8% | 98.8% | 6/6 |
| IMG_0106 | green / pink | 98.8% | 99.5% | 99.9% | 97.3% | 87.4% | 96.3% | 0.5% | 1.0% | 98.7% | 6/6 |
| IMG_0107 | yellow / purple | 98.5% | 99.6% | 99.9% | 96.0% | 84.0% | 98.1% | 0.7% | 0.3% | 98.4% | 6/6 |
| IMG_0108 | green / yellow | 99.2% | 94.0% | 99.9% | 97.6% | 89.7% | 98.7% | 0.3% | 0.3% | 98.8% | 6/6 |
| IMG_0109 | purple / pink | 98.6% | 99.2% | 99.9% | 97.0% | 83.8% | 99.9% | 0.7% | 0.0% | 98.5% | 6/6 |
| IMG_0110 | purple / green | 98.6% | 99.3% | 99.9% | 96.6% | 85.1% | 96.7% | 0.7% | 0.9% | 98.5% | 6/6 |
| IMG_0111 | pink / yellow | 98.9% | 99.5% | 100.0% | 97.1% | 87.9% | 97.2% | 0.5% | 0.5% | 98.8% | 6/6 |
| **Mean** | | **98.8%** | | | | **86.2%** | **97.6%** | | | | |

IMG_0104 has no detections yet. "Non-players rejected" counts boxes under IoU 0.1 with every GT player that came out `none` or
`unsure`. The two "calls" columns are shares of all team calls at confidence 0.3 or above.

**On RF-DETR boxes, with confidence at least 0.3:** of all team calls on IMG_0106, 94.9% are the right team, 0.5% the wrong team, 3.6%
loose boxes around a real player (under IoU 0.5 with GT, usually still the right colour), and 1.0% non-players.

**Repeated draws:** with 20 random sets of 10 frames per run, precision never fell below 97.7% on GT boxes or 98.4% on RF-DETR boxes,
and no calibration warned. The lowest coverage was 89.3%, on IMG_0108 (yellow and green, the closest pair, where more boxes come out
`unsure`).

**Over each whole player:** the most common team call was right for all 6 players in every game.

## Advantages

- **No training and no weights.** Nothing to fine-tune or download, no GPU, and nothing that touches the ViTPose or OpenMMLab question.
  It runs on a Mac in a few minutes per video.
- **Calibrated on each video.** Colours are measured on the video being labelled, so a change in camera, light or white balance moves
  the reference colours with it. Training only says which colour is which.
- **Teams named automatically.** The roster gives each team's numbers, and the numbers give the bib colour, so nothing is typed in per
  video.
- **Rejects most non-players.** People in black, waiting teams in other bib colours, and people standing still (who end up in the
  background image) mostly come out as `none`.
- **Precise where it matters most.** Clear boxes are almost never wrong, and `unsure` takes the doubtful cases instead of guessing.
- **Matches the tracker plan.** Writes the label format from `docs/13`, keyed by detection row and checked against the detection file
  hash.
- **Can be checked by eye.** The video and calibration JSON show what was decided and why.

## Disadvantages and risks

- **Cross-team overlaps.** On RF-DETR boxes, 10-16% of calls are wrong when a player overlaps someone from the other team: the chest region holds
  both bibs. This is the moment the tracker most needs team. Use a soft penalty and a vote over whole tracks, not a hard block.
- **The challenge videos are unmeasured.** They have no GT, and their camera angle is new: training is 4K end-line, the challenge is 720p
  oblique. Both challenge cameras moved at most about 0.3 pixels between calibration frames, so background removal should hold.
  Accuracy there can only be checked by watching the video.
- **Needs a fixed camera.** Panning, zoom or a bumped tripod would break the background image. A fallback would be to skip the
  background and accept the grass errors.
- **Close colours.** Yellow and green are about 13 Lab units apart, against 40 or more for the other pairs. They still reach 98.8%
  precision on IMG_0108, but with 4% `unsure`. This pair is in IMG_0114.
- **Someone standing still all game.** A player who barely moves for most of the 10 frames would partly enter the background image.
  Not seen in testing.
- **A non-player in a team's colour.** Someone wearing a bib of a playing team, near the court, would be called that team. The scorekeeper
  in IMG_0114 wears purple, which isn't a colour in that game.
- **Tuned on the same games it was scored on.** The bib reference hues and thresholds were set on the training games, so the numbers may
  be slightly optimistic. The margins are wide, and each video's colours are measured on that video.
- **Fixed chest region.** It assumes upright players. Crouching, jumping or boxes cut off by the frame edge put the wrong part of the body
  in the region. A ViTPose torso crop could replace it, and `run(crop, whole_player=False)` accepts one.
- **Team only.** It can't separate the 3 players on a team. That needs jersey reading.
- **Scores aren't calibrated probabilities.** `p_*` come from distances. The tracker plan calls for a calibration table before they're
  used as probabilities.

## Rules and disclosure

- No model is trained or updated, and challenge GT is never read: `load_boxes` refuses GT for challenge videos.
- The background image and colours are computed automatically from each challenge video. Organizers allow automatic whole-video
  processing ([Email 5](competition/organizer_emails.md)). The paper should describe this.
- The colour-to-team mapping comes from the roster plus bib colours learned in training. The paper should say so.

## How to run

```bash
# One video from detections, with the team video
uv run python -m src.team.run IMG_0114 --boxes outputs/<...>/IMG_0114__rfdetr-large__720p.txt --video

# Precision check on a training game with GT boxes, every 5th frame, 20 repeated draws
uv run python -m src.team.run IMG_0108 --boxes gt --every 5 --seeds 20

# Tests
uv run python -m pytest src/tests/test_team.py -q
```

## Next

1. Run IMG_0114 and IMG_0115 once their detections exist, and watch both videos.
2. Read these labels in the tracker: a soft team penalty in matching, then a team vote over each track piece.
3. Calibrate `p_*` into probabilities on training games before using them as log-odds.
