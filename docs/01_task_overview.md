# Task Overview

## The task in one paragraph
Two static, oblique, 720p videos of 3x3 basketball games (IMG_0114, IMG_0115; ~6.5 min each, ~11.5k frames).
For **every frame**, output where each of the **6 on-court players** stands on the court (cm, standardized
15.05 × 11.05 m 3x3 court) and **who they are**, as team (A/B) plus jersey number. Rosters are given, so there are
exactly 6 known identities per video. Scored by TI-HOTA ([02_evaluation_metric.md](02_evaluation_metric.md)).
One blind shot: no leaderboard, no feedback (resubmission allowed, the last counts).

![Challenge view](assets/challenge_IMG_0114_t120s.jpg)

## Where this comes from
- Organized by Keisuke Fujii's STAL Lab (Nagoya University) as the **9th ABC Challenge** (Activity and Behavior
  Computing, ABC 2027, Sydney, Jan 2027), co-organized with Sozo Lab (Kyushu Institute of Technology).
- Built on **TrackID3x3** (MMSports '25), whose "Outdoor" subset consists of 12 games among 4 student teams
  (double round-robin) at the Ryutsu Keizai University (RKU) outdoor court, filmed with **two iPhone 13 cameras**.
  - Training set = 8 games from the **end-line ("top view") camera**, 4K, with full annotations.
  - Challenge = 2 other games from the **second, oblique camera**, downscaled to 720p. GT not released.
  - (The remaining two games, presumably IMG_0112/0113, are not distributed.)
- It is the basketball analogue of **SoccerNet Game State Reconstruction (GSR)**; TI-HOTA is GS-HOTA renamed.

## What makes it hard
| Difficulty | Detail |
|---|---|
| **Viewpoint/domain shift** | All labels are from the end-line 4K camera; the challenge is an elevated corner view at 720p. No legal oblique-view labels exist |
| **Strict localization** | Full credit needs ≤13 cm error, zero credit beyond 100 cm, measured against a bbox-bottom-centre convention from the *other* camera |
| **Shared jersey numbers** | IMG_0114: both teams have #6 and #9, so team must come from bib colour |
| **Distractors** | Referee on the court edge, staff in black, the waiting team in bibs on the sideline, a scorekeeper in a **purple #10 bib** |
| **Occlusion** | The oblique view stacks players near the key; 3x3 involves constant screens and close contact |
| **Small players** | ~60–130 px tall at 720p; numbers are readable only on some frames |
| **All frames count** | Pre-game warm-up, dead balls, and players leaving the court are all scored |
| **Single blind submission** | You must build your own validation |

## What makes it tractable
- Static camera, the same pose for both videos, and official 4-point calibration (add more landmarks manually if wanted).
- Exactly 6 identities, known in advance; 3 per team; distinct bib colours with large printed numbers.
- The same 4 teams and the same 14 people appear in the training games, so appearance priors transfer.
- The metric has no ID-switch term, so offline global assignment and smoothing are fully rewarded.
- Training GT reveals the exact annotation convention (bbox bottom-centre → homography), so the target can be modelled.
