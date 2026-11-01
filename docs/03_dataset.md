# Dataset (in depth)

All facts here were measured on the files in `data/` unless marked otherwise.

## 1. Inventory

```
data/
├── Training/                                   (official Training set: the only data you may train on)
│   ├── court_registration.csv                  4 corners, 4K pixel coords   [downloaded from Drive 2026-10-25]
│   ├── games_roster.csv                        8 games                       [downloaded from Drive 2026-10-25]
│   ├── vidoes/                                 (sic)
│   │   └── IMG_0106.MOV                        ⚠ only 1 of 8 videos local; 0104,0105,0107–0111 are on Drive (~9.5 GB)
│   └── full player annotations/
│       ├── Image-space bounding boxes/IMG_0104…0111.txt   (6 local before; all 8 now)
│       └── court position/IMG_0104…0111.txt               (1 local before; all 8 now)
└── Final challenge/
    ├── IMG_0114.MP4, IMG_0115.MP4
    ├── court_registration.csv                  corrected Oct 21 version
    └── games_roster.csv
```

## 2. Videos

| | Training (e.g. IMG_0106) | Challenge IMG_0114 | Challenge IMG_0115 |
|---|---|---|---|
| Camera | iPhone 13, end-line, high, behind the far line | iPhone 13 (2nd camera), elevated corner/side | same as 0114 |
| Codec / res | HEVC 3840×2160 | H.264 1280×720 | H.264 1280×720 |
| FPS | 29.97 (avg 29.98, slightly VFR) | 29.97 CFR | 29.97 CFR |
| Frames | 11272 | **11536** | **11823** |
| Duration | 376 s | 385 s | 394 s |
| Static? | yes | yes (<0.1 px drift) | yes, **same pose as 0114** (<0.7 px) |

Frame counts per training game (from GT): 0104 11376 · 0105 13228 · 0106 11272 · 0107 11921 · 0108 12097 ·
0109 10759 · 0110 10706 · 0111 12222.

Each video covers one 5-minute game (no timeouts) plus a pre-game segment of 3–8 s (training
`game_start_frame` = 90–232) and dead-ball time.

## 3. Annotation formats

**Court position** (`court position/IMG_01xx.txt`), headerless:
```
frame, court_x_cm, court_y_cm, team(A/B), jersey
1,223.481,501.318,A,6
```
99.8 % of frames have exactly 6 rows. Exceptions: **5-row frames** where a player is out of the end-line camera's
view (0104: 80 frames, 0106: 149 frames, 0111: frame 1), and **7-row frames** where an identity is duplicated
(0105: 25 frames with A9 twice, an annotation glitch). No frame is missing entirely. The format equals the submission
format. If challenge GT behaves the same way, a player outside the end-line camera's view has no GT row, so your row
for them is an FP. The effect is tiny, but it is a reason not to hallucinate positions far off court.

**Bounding boxes** (`Image-space bounding boxes/IMG_01xx.txt`), MOT-style, headerless:
```
frame, track_id, x_left, y_top, w, h, conf, -1, -1      (4K pixels)
1,1,2033.10,1363.50,137.70,391.50,1.00,-1,-1
```
6 tracks per game; occasionally only 5 boxes (0104: 80 frames, 0106: 149 frames), when a player is out of view.
Median box is 115–122 × 269 px at 4K (≈ 40 × 90 px if downscaled to 720p, but the challenge camera is closer to
some players). The conf column (0.97–1.00) suggests the boxes started as detector output and were then corrected
(paper: half manual, half BoT-SORT-ReID output corrected by hand; YOLOX was the detector).

**Roster, training** (`Training/games_roster.csv`):
```
video, game_start_frame, "track_id: jersey", offense_track_ids, defense_track_ids, "team: jersey"
IMG_0104,117,"1:10, 3: 6, 4: 4, 5: 12, 6: 9, 8: 11","1, 4, 5","3, 6, 8","A: 4, 10, 12; B: 6, 9, 11"
```
(The spacing is inconsistent; parse robustly.) Offense IDs = the team that starts with the ball. Team A is the offense
in 0104 and 0106 but **B** in 0107, 0109 and 0111, so "A" does not mean offense.

**Roster, challenge**: `IMG_0114,"A: 6, 9, 10; B: 6, 9, 11"` and `IMG_0115,"A: 4, 10, 12; B: 1, 3, 5"`.

**Court registration**: `point_id,image_x,image_y,court_x,court_y` for RKU0 (0,0), RKU1 (0,1105),
RKU10 (1505,0), RKU11 (1505,1105). The keypoints are line centres (paper §3.1).

| | RKU0 | RKU1 | RKU10 | RKU11 |
|---|---|---|---|---|
| Training (4K px) | 2698.8, 1172.4 | 3749.9, 1726.6 | 875.7, 1237.9 | 108.4, 1970.3 |
| Challenge (720p px) | 1113, 379 | 955, 531 | 512, 305 | 141, 374 |

## 4. ★ How GT positions were made (key finding)

Applying the training 4-point homography (image→court) to the **bbox bottom-centre (x + w/2, y + h)** reproduces
the court-position file **exactly (0.00 cm median, p90 and max)** on IMG_0104, 0106, 0108 and 0110.

Using 95 % of box height instead gives errors of about **30 cm** median (p90 about 35 cm).

Consequences:
- The GT is not "true foot position". It is **where the end-line camera's box bottom-centre lands on the ground plane**.
  At the end line, the bottom edge of the box is the foot lowest in the image, i.e. the foot nearest the end-line camera
  (largest court_y), plus any annotator padding. When a player jumps, the bottom-centre leaves the ground and the GT
  "position" moves toward the camera (+y).
- Challenge GT was almost certainly produced the same way from the **synchronized end-line camera** (the paper used
  "only one camera" for the public set, and the organizers forbid "synchronized views"). So in the oblique video,
  you are predicting what another camera's box would say.
- This can be learned from training: model the offset between a physically grounded estimate (ankle keypoints,
  silhouette contact) and the GT convention, as a function of pose, motion, and position relative to the end-line camera.

## 5. Spatial statistics of GT

| Game | % rows >50 cm outside the court | % rows >300 cm outside | max single-frame jump (cm) |
|---|---|---|---|
| 0104 | 1.00 | 0.13 | 153 |
| 0105 | 0.48 | 0.11 | 197 |
| 0106 | 0.49 | 0.16 | 309 |
| 0107 | 0.91 | 0.32 | 177 |
| 0108 | 1.27 | 0.41 | 135 |
| 0109 | 0.47 | 0.11 | 140 |
| 0110 | 1.21 | 0.19 | **994** |
| 0111 | 0.15 | 0.00 | **583** |

- Per-frame displacement (0108): median 4.0 cm, p99 26 cm. Sprinting is about 25–30 cm per frame (≈ 8–9 m/s).
  Jumps above ~60 cm per frame are annotation glitches (box swaps or bad boxes); filter them before training motion models.
- Off-court positions go as far as y ≈ −1278 cm (behind the end line, chasing the ball), and **are still annotated**.
  Do not hard-clip predictions to the court.

## 6. Teams, bibs and people

The paper says 14 male players formed 4 teams of 3 (plus 2 reserves) and played a double round-robin of 12 games
in **green, purple, pink and yellow bibs**. Mapping verified from video frames:

| Team (jerseys) | Bib colour | Games in training | Challenge |
|---|---|---|---|
| {6, 9, 11} | **green** | 0104, 0106, 0108, 0110 | IMG_0114 as **B** |
| {6, 9, 10} | **yellow** (players also in blue shorts) | 0105, 0107, 0108, 0111 | IMG_0114 as **A** |
| {1, 3, 5} | **pink** | 0105, 0106, 0109, 0111 | IMG_0115 as **B** |
| {4, 10, 12} | **purple** | 0104, 0107, 0109, 0110 | IMG_0115 as **A** |

Verified directly: green #9/#6 and pink in 0106; yellow #10 and green #11 in 0114; pink #3/#5 in 0115. The other
cells are inferred from the rosters and the paper.
**Every challenge team appears in 4 training games**, so team appearance models can be learned from training.

Number collisions to handle: #6 and #9 (green and yellow), #10 (yellow and purple).

![Zoomed challenge crops: yellow #10 = team A, green #11 = team B in IMG_0114](assets/challenge_IMG_0114_zoom_bibs.jpg)

## 7. Non-players visible in the challenge view
- Referee in black with red shorts, often standing on or near the left court edge
- Staff in a black T-shirt and light shorts, near the right side of the end line
- Waiting team in bibs on the right sideline (IMG_0114: purple; IMG_0115: yellow)
- Scorekeeper holding the scoreboard: **purple bib #10** in IMG_0114
- Seated spectators on the grass; passers-by in the background

## 8. Calibration quality

![Challenge court overlay](assets/challenge_IMG_0115_court_overlay.jpg)

Projected FIBA lines (key 4.9 × 5.8 m, 6.75 m arc, 1.25 m no-charge semicircle) line up well with the painted
lines in both views. The training overlay shows small arc deviations, likely mild lens distortion plus the 4-point fit.

![Training court overlay](assets/training_IMG_0106_court_overlay.jpg)

Note: the GT coordinate frame is defined by the **training-style camera's 4-point homography**. Any error in that
fit is baked into the GT. A "more accurate" challenge calibration from many landmarks could in principle drift
from the GT frame. Keep the 4 official corners as hard constraints if you refine.

## 9. What's not available (and must not be obtained)
- No oblique-view labels of any kind.
- No synchronized end-line footage of the challenge games (prohibited).
- Pose keypoints: the public TrackID3x3 release has some, but they are outside the official Training set, so prohibited.
  Use public pretrained pose models instead.
