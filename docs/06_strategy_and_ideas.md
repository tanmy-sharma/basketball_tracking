# Strategy and Ideas (ideation, not a committed plan)

Based on the metric analysis ([02](02_evaluation_metric.md)), the data findings ([03](03_dataset.md)) and the rules
([competition/rules_and_data_policy.md](competition/rules_and_data_policy.md)). Ideas are ranked by expected score impact.

---

## 0. Score budget: where the points are

Rough guess for a solid but unrefined pipeline on the oblique view, compared with the measured sensitivities:

| Error source | Plausible naive level | Score cost | Fixable? |
|---|---|---|---|
| Localization noise and bias (box-bottom jitter, convention mismatch, calibration) | σ ≈ 20–30 cm | **−25 to −45 %** | Largely: smoothing plus learned convention mapping |
| Identity errors (team, number, swaps after crossings) | 5–10 % of frames | −5 to −15 % | Mostly: global assignment plus colour |
| Missed detections and occlusion gaps | 5–10 % of rows | −2 to −5 % | Interpolation |
| Distractors (referee, scorekeeper) emitted as players | a few % | −3 to −7 % | Colour gating plus 6-ID constraint |

**Conclusion: localization quality is the main battleground**, then identity robustness. Most teams will spend their
effort on tracking and OCR and will leave the bias/variance of the ground point untreated. That is the edge.

---

## 1. Problem formulation: 6 fixed identities, solved offline

Don't do generic MOT and attach labels afterwards. Instead:

1. Detect people on every frame and build short, high-precision **tracklets** (cut at any ambiguity).
2. For each tracklet, compute evidence: bib-colour posterior over {team A, team B, non-player}, a jersey-number
   posterior from many crops, court trajectory, and timing.
3. **Globally assign tracklets to the 6 roster identities** (plus a "non-player" sink) by solving a min-cost problem:
   - at most 1 tracklet per identity per frame; at most 3 active per team;
   - reachability: consecutive tracklets of one identity must be physically linkable (≤ ~9 m/s);
   - cost = −log P(colour) − log P(number) + motion-gap cost.
   ILP (OR-Tools/PuLP), or a min-cost-flow / Hungarian over time windows. 23k frames is small.
4. Fill gaps with physically plausible interpolation (§2). Abstain where the identity is genuinely unknown (see the algebra in 02 §4).

The metric has no ID-switch term, so all that matters is the per-frame label. Offline global assignment is the
principled way to maximize that. (The organizers confirmed this is allowed if automatic: `competition/organizer_emails.md`, Email 5.)

## 2. Localization: predict the GT convention, not the truth

GT = homography_endline(bbox bottom-centre in the end-line camera). We see the scene from a different angle, so:

**2a. Get a stable physical ground point in the oblique view**
- Use a top-down pose model (RTMPose/ViTPose, public) on upscaled crops and take the ankle keypoints. Ground contact ≈ the
  ankle midpoint pushed down by a small offset. Fuse with box-bottom estimates, weighted by keypoint confidence.
- Detect jumps (hip height above the median standing height, or ankles rising). While airborne, use the pre-jump
  ground position or interpolate across the jump.
- Smooth in court space with a constant-velocity Kalman filter plus an RTS smoother. Tune process noise on training GT
  (also run in court space) so that you do not over-smooth genuine cuts.

**2b. Convert "physical ground point" to the "end-line bbox-bottom-centre" convention**
- We know where the end-line camera is (from its homography, by decomposing it with an assumed focal length, or
  directly from the 4K training images).
- On training videos, compute both a pose-based ground point and the GT (bbox bottom-centre), and **learn the offset**
  as a function of position, motion direction, speed, jump state, and pose (stride width, facing). Expected effect: the
  bbox bottom is the foot nearest that camera, so the GT is biased toward +y by up to half a stride, plus any box padding.
- Apply the learned offset to the oblique-view physical estimate. This is legal: it is learned on the Training set,
  then applied at inference.
- Even a constant learned offset helps. A bias under 13 cm is free (02 §4b), so it only needs to be roughly right.

**2c. Calibration hygiene**
- Use the official 4 corners. Check them against painted lines. If needed, add manually clicked landmarks (key corners,
  free-throw line ends, arc–baseline intersections; allowed) and estimate lens distortion, while keeping the official
  corners as anchors (the GT frame is defined by them).
- Check the GT side too: project the training GT onto the training 4K frames and see whether the projected lines match.
  Any distortion in the training camera is baked into the GT frame, and there is nothing to do about that.

**2d. Validate localization without oblique labels**
- **Synthetic oblique views from training**: in training 4K frames you have GT boxes and court positions. Render a virtual
  oblique view by warping the ground plane with the challenge-camera homography. People are not planar, so this is
  approximate. It tests calibration and convention, not appearance.
- **Cross-camera geometry check**: the challenge homography and the training homography map the same court. Project a
  synthetic standing person (a vertical segment in 3D) into both views to see what offset the convention implies. This is
  a pure-geometry estimate of the bias.

## 3. Team identification (bib colour)

- 4 bib colours (green, yellow, pink, purple), each team always in its own colour, and the same people in the training
  games. Learn a colour or appearance classifier on training crops (8 games, ~550k labelled boxes), with
  augmentations for 720p, white balance, motion blur and shadow. Bonus: train on **synthetic downscales and
  viewpoint warps** of training crops (augmentation is allowed).
- At inference, use a per-tracklet colour vote with a "non-player" class (referee black, staff, waiting team colour,
  scorekeeper). The waiting team's colour is never one of the two rostered teams, so it is rejected automatically.
- **Mapping colour to A/B must be automatic, not hand-typed.** Use OCR evidence: in IMG_0114, the colour cluster whose
  tracklets read "10" is team A (roster), and the one reading "11" is team B. In IMG_0115 all numbers are distinct. A
  training-learned colour→team-roster mapping (e.g. "yellow bibs = team {6, 9, 10}") is also legitimate, since it is
  learned from the Training set, but disclose it.

## 4. Jersey numbers

- Closed set: within a team, only 3 numbers. It is a 3-way classification per tracklet.
- Evidence sources: (1) a small CNN trained on training-set torso crops (end-line view, so mostly backs and fronts
  are visible; augment to oblique 720p), (2) zero-shot OCR or a VLM on upscaled crops, (3) consistency
  constraints (each number exactly once per team at any time).
- Weight votes by legibility (crop size, sharpness, facing). Many frames will be illegible. You only need a few good
  reads per tracklet, and the global assignment carries identity through.
- Watch for 6↔9 (upside-down or mirrored crops) and 10↔11 in IMG_0114.

## 5. Detection and tracking details

- Pretrained person detector (YOLOX-SportsMOT, RF-DETR, Co-DETR) on frames upscaled ×1.5–2 or tiled; keep a low score
  threshold, then filter with the court mask and colour.
- Static camera: background subtraction gives a cheap extra cue for recall under occlusion.
- Short tracklets via ByteTrack/BoT-SORT with **court-space** gating (≤ 30 cm per frame); split on uncertainty.
- Occlusion clusters near the key: consider SAM2 mask propagation for the hard segments (public checkpoint).
- Pre-game: GT exists from frame 1. Players may be warming up or walking. The same pipeline applies, but expect more
  non-players on court. The 6-identity constraint plus colour handles it.

## 6. Training-data plan (all legal)

| Use | Data | How |
|---|---|---|
| Colour/team classifier | training bboxes plus roster → team labels | crop, downscale to 720p-equivalent, augment |
| Jersey classifier | training bboxes plus jersey labels | same; balance the classes; rotation augmentation |
| Ground-point convention model | training GT plus pose-model keypoints | regression of offset |
| Motion priors and smoother tuning | training court GT (filter glitches over 60 cm per frame) | fit Kalman/RTS noise |
| Detector fine-tune (optional) | training frames, viewpoint-warped and downscaled | low priority; pretrained detectors are good on people |
| Validation | hold out 2 training games (one with each challenge team pair if possible) | full pipeline end to end |

Download the 7 missing training videos first (~9.5 GB). Decoding 4K HEVC is slow, so extract downscaled frames
once (e.g. 1920×1080 or 1280×720) and cache them.

## 7. Validation protocol (single blind shot, so this is critical)

1. **Training-view proxy:** run the whole pipeline on 2–3 held-out training games (end-line view) and score with
   `tools/trackid/metric.py`. This measures detection, tracking, identity and convention correctness in that view.
2. **Synthetic oblique proxy** (§2d) for calibration and convention.
3. **Visual audit on the challenge videos:** overlay predictions with `tools/viz_overlay.py` on sampled frames and look for
   teleports, identity flips and off-court ghosts. Looking is allowed, but use it to fix *code*, never to hand-label.
4. **Internal consistency metrics** on challenge outputs (no GT needed): per-frame row count, max speed, number of
   identity changes per tracklet, colour-cluster purity, OCR agreement rate.
5. Run `tools/validate_submission.py` before every upload.

## 8. Suggested schedule (9 days)

| Day | Date | Goal |
|---|---|---|
| 0 | Sun Oct 25 | **Register the team.** Download training videos. Email the organizers about the grey zones |
| 1 | Mon Oct 26 | Detector + tracklets on the challenge videos; court projection; visual audit |
| 2 | Tue Oct 27 | Colour/team classifier; naive identity; **first valid submission uploaded** |
| 3–4 | Oct 28–29 | Jersey evidence plus global assignment; training-view validation loop |
| 5–6 | Oct 30–31 | Localization: pose ground point, smoother, learned convention offset |
| 7 | Nov 1 | Robustness: pre-game, distractors, occlusion clusters; abstention policy |
| 8 | Nov 2 | Freeze; full reproducible run from a clean environment; final upload with margin |
| 9 | Nov 3 | Buffer only. Deadline 23:59 AoE = Nov 4 17:29 IST |
| 10–16 | Nov 4–10 | Paper (8–12 pages) plus code/weights/README package |

## 9. Paper angle (judged too, and required)

The novelty story writes itself if you do §2: *"Cross-view annotation-convention transfer for court localization"*,
plus *"roster-constrained global identity assignment"*. Include ablations on held-out training games (smoothing,
convention offset, global assignment versus online labelling) and a full disclosure table of checkpoints.
