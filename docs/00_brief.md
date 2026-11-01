# TrackID3x3 Challenge (ABC 2027) — Competition Brief

Compiled 2026-10-25. Sources: challenge Google Site (all pages), organizer emails (Oct 2026),
TrackID3x3 paper arXiv 2503.18282v2 (MMSports '25), official repo open-starlab/TrackID3x3,
and direct analysis of the data. Items marked (verified) were checked against data/code directly.

---

## 1. Deadlines (AoE = UTC−12)

| What | When | Notes |
|---|---|---|
| **Team registration** | **Oct 27, 2026** | Form: https://docs.google.com/forms/d/e/1FAIpQLScl7duLqA5MUZMO4Lz-k4JNYty8IjSwBKWypIE63f9XBoNFag/viewform — every participant, even solo |
| **Results** | **Nov 3, 2026 23:59 AoE** (= Nov 4 11:59 UTC = Nov 4 17:29 IST) | https://forms.gle/rMwK4pHm1Xp2joV27 — one ZIP, re-submittable, **latest before deadline counts**, no scores/feedback |
| Paper + code + weights + README | Nov 10, 2026 | CMT, details TBA. Code/weights must be the exact ones used for Nov 3 predictions |
| Workshop | ABC 2027, Jan 19–22, 2027, Sydney (hybrid) | ≥1 member must present |

Prize: 100,000 JPY + wearable devices for the winner; certificates for all. Paper: English, ABC format,
8–12 pages incl. refs; sections Abstract / Methodology / Results & Discussion / Conclusion / References;
must cite the TrackID3x3 paper (K. Yamada et al., MMSports '25, pp.163–173, DOI 10.1145/3728423.3759400).
Published by default in IJABC (J-STAGE). Eligibility: students/working people; no team-size limit;
students of the host/organizing labs ineligible. Contacts: trackid3x3-challenge-2026@g.sp.m.is.nagoya-u.ac.jp,
abc2027@autocare.ai (subject "ABC Challenge").

## 2. Task

Fixed oblique camera, 3x3 basketball (one hoop, 15.05 m x 11.05 m court). For **every frame** output the
court position (cm) of all 6 on-court players with identity = team (A/B) + jersey number. Roster per
video is given → exactly 6 valid identities.

Challenge rosters (`Final challenge/games_roster.csv`):
- IMG_0114: A = 6, 9, 10 · B = 6, 9, 11 → **#6 and #9 exist on both teams; team must come from bib color**
- IMG_0115: A = 4, 10, 12 · B = 1, 3, 5 → all numbers unique; jersey alone determines team

## 3. Submission format (supersedes older formats)

- Files `IMG_0114.txt`, `IMG_0115.txt`, zipped at ZIP root (no folders).
- Headerless CSV, 5 columns: `frame_id,court_x,court_y,team_id,jersey_number`
- frame_id = original numbering from 1; **all frames evaluated, including pre-game**.
  Frame counts (ffprobe, verified): IMG_0114 = 11536, IMG_0115 = 11823 (CFR 29.97, 1280x720).
- Coordinates in **cm** (not m). team_id ∈ {A, B}. No track-ID column.
- **No spaces in fields**: the official parser does `line.split(',')` with no strip, and identity is string
  equality (e.g. `"A10"`), so `A, 10` would never match.

## 4. Metric — TI-HOTA (official code read line by line, verified)

```
LocSim = exp(ln(0.05) * d² / τ²),  τ = 100 cm
IdSim  = 1 if attribute string equal else 0
Sim    = LocSim * IdSim
For α in 0.05, 0.10, …, 0.95 (19 values):
  per frame: Hungarian on 1−Sim, pairs with Sim < α dropped → TP/FP/FN
  DetA_α = TP / (TP+FP+FN)
  AssA_α = mean over GT identities g of TPA_g / (TPA_g + FNA_g + FPA_g)
           (FPA_g = unmatched predictions labelled g)   ← organizers confirmed this, not paper Eq. 5
  HOTA_α = sqrt(DetA_α · AssA_α)
TI-HOTA = mean_α HOTA_α ; computed per video, then (presumably) averaged over the 2 videos
```

What this implies:
- **There is no track-ID / ID-switch term.** "Association" is just per-identity precision/recall over
  frames. Being right about *who* each point is in each frame is all that matters.
- **Localization is scored on a steep curve.** The distance that still counts as a match at each α:
  | α | 0.05 | 0.25 | 0.50 | 0.75 | 0.95 |
  |---|---|---|---|---|---|
  | max error (cm) | 100 | 68 | 48 | 31 | 13 |
  Roughly, TI-HOTA ≈ the mean LocSim of correctly identified points: 20 cm error → ~0.89,
  30 cm → ~0.76, 50 cm → ~0.47. **Centimeter-level localization is the main lever.**
- A point with the right identity but over 100 cm off counts as an FP plus an FN.
  A point with the wrong identity counts as an FP plus an FN for both identities' AssA.
- Extra rows (a 7th person, a duplicate identity) are pure FP.

## 5. Data (verified)

### Official Training set = 8 games, IMG_0104 … IMG_0111 (end-line "top view", iPhone 4K HEVC, 29.97 fps)
- `Training/vidoes/IMG_01xx.MOV`: ~1.2–1.5 GB each. **Only IMG_0106.MOV is downloaded locally**;
  the other 7 are on the Drive (folder id 1lJGS6Amp0MWLnoNysH100JTJhyx_MHaM).
- `Training/full player annotations/court position/IMG_01xx.txt`: `frame,court_x,court_y,team,jersey` (cm),
  **6 rows on every frame** (all 8 now in `data/`).
  (Correction: 99.8 % of frames have 6 rows. A few frames have 5 rows, when a player is out of the camera view, or 7 rows,
  from a duplicate-ID glitch in 0105. See 03_dataset.md §3.)
- `.../Image-space bounding boxes/IMG_01xx.txt`: MOT `frame,track_id,x,y,w,h,conf,-1,-1` (4K px, top-left),
  6 tracks per game, occasional 5-box frames (all 8 now in `data/`).
- `Training/games_roster.csv`: `video, game_start_frame, track_id:jersey, offense_track_ids,
  defense_track_ids, team:jersey`. game_start_frame is 90–232 (pre-game segment ≈ 3–8 s).
- `Training/court_registration.csv`: 4 corners in 4K pixel coordinates.
- (These two CSVs and 14 annotation files were missing locally; I downloaded them from the official Drive
  and copied them into `data/Training/` without overwriting anything. Checksums of the 3 files you already
  had match exactly.)

Frames per game: 104: 11376 · 105: 13228 · 106: 11272 · 107: 11921 · 108: 12097 · 109: 10759 · 110: 10706 · 111: 12222.

**Key finding: GT court position = 4-point homography applied to the bbox bottom-centre in the end-line view,
exact to 0.00 cm on all games tested.** Moving the foot point up by 5 % of box height moves it about 30 cm on court.
The challenge GT was almost certainly made the same way from the synchronized end-line camera, which the
public TrackID3x3 Outdoor set also uses. So the target is "where an end-line annotator's box bottom-centre
lands", not the true foot position.

Other stats: about 0.5–1.3 % of GT positions lie more than 50 cm outside the court (players chasing the ball) and must still
be predicted. Some GT label glitches exist: single-frame jumps of 300–994 cm in 0106, 0110 and 0111.

Teams are four groups that recur across games, with fixed bib colors (verified on 0106, 0114 and 0115):
- {6, 9, 11} = **green**
- {6, 9, 10} = **yellow** (blue shorts)
- {1, 3, 5} = **pink**
- {4, 10, 12} = **purple**
- IMG_0114: A = yellow, B = green. IMG_0115: A = purple, B = pink.
- The same four teams and the same people appear in training, so appearance priors transfer.

### Final challenge
- Oblique corner-elevated view, 720p. Players are about 60–130 px tall and the bib numbers are large and
  readable when upscaled. Camera is **static and identical for both videos** (<1 px drift, verified).
- `court_registration.csv` (corrected Oct 21; your copy matches the corrected file):
  RKU0 (1113,379)→(0,0), RKU1 (955,531)→(0,1105), RKU10 (512,305)→(1505,0), RKU11 (141,374)→(1505,1105).
  Projected key and 3-point arc line up well with the painted lines.
- Distractors in view: a referee (black, often on the court edge), staff in a black T-shirt, a seated
  person, the waiting team in other-colored bibs on the sideline, and a **scorekeeper in a purple #10 bib
  in IMG_0114** (#10 is a real roster number there). Gate identity by bib color.

Domain gap: training is end-line 4K and challenge is oblique 720p. Same venue and same players, different
viewpoint and resolution, and no oblique-view GT you are allowed to use.

## 6. Rules (all emails + site, consolidated)

ALLOWED
- Training or fine-tuning on the **official Training set** only, including augmentations of it.
- Public general-purpose pretrained checkpoints (COCO/CrowdHuman detectors, pose models, PARSeq, VLMs such
  as Qwen-VL, SigLIP, …), provided they were not trained on the challenge sequences. **Disclose
  checkpoints and sources.**
- Synthetic data generated independently, from the Training set, or from permitted public sources
  (Oct 23 amendment). Disclose how it was generated.
- Inference on the challenge videos, including normal online tracker state updates.
- Manual selection of fixed court landmarks for calibration (describe it in the paper).

PROHIBITED
- Any other real training data, **including public TrackID3x3 annotations outside the official set**
  (the public repo has Outdoor GT, i.e. same games from the other camera, so **do not touch it**).
- Training directly on external real footage or its ordinary augmentations, even under the synthetic exception.
- Self-training, pseudo-labels, or fine-tuning weights on the challenge videos.
- Manual frame-by-frame player annotation, identity assignment, or correction.
- Using public same-game GT trajectories to produce predictions.
- Sharing code or data outside the team.

GREY: ask the organizers before relying on these
- Offline/bidirectional processing (global tracklet assignment, RTS smoothing, interpolation). Only "online
  state updates" is explicitly blessed, but the official baseline itself interpolates and extrapolates offline,
  and the rule's intent is about training. Likely fine; a one-line email would make it certain.
- Hard-coding the bib-color↔team mapping or a per-video color prior picked by a human. This edges toward "manual
  identity assignment". Safer: derive it automatically, from jersey OCR plus color clustering, with roster logic.

## 7. Baseline to beat (paper v2, Outdoor subset, τ = 1 m)

Track-ID baseline: YOLOX (SportsMOT weights, no fine-tune) + CAMELTrack + homography of bbox bottom-centre +
rule-based non-player removal + jersey-number-pipeline (PARSeq + legibility) + 3D color histogram (JS divergence)
for team + ID-switch merging + interpolation/extrapolation, capped at 6 per frame.
→ **TI-HOTA 71.03 ± 17.8 (DetA 67.15, AssA 75.51)**; 66.20 without merging. Those scores are on the end-line view;
the oblique 720p challenge view is likely harder.

## 8. Ideas for winning (ideation: where the points are)

1. **Model the problem as 6 fixed identities.** Output exactly the 6 roster identities on every frame, never
   more and never fewer. Run tracking as tracklet generation followed by global assignment of tracklets to
   identities (Hungarian/ILP over the whole video, with no-overlap constraints, 3 per team, colour evidence and
   number evidence). There is no ID-switch penalty, only per-frame identity correctness, so global offline
   assignment beats online tracking.
2. **Localization is the biggest lever** (see §4). Predict the GT convention, not the true foot position:
   - Use a pose model for ankles, plus the box bottom, and fuse them into a ground contact point.
   - Learn the bias between the true ground point and the end-line bbox bottom-centre convention from training.
     Train a small regressor from pose and box features to the GT point. The end-line bbox bottom is roughly the
     foot nearest that camera (+y), plus the annotator's box padding.
   - Smooth in court space (Kalman/RTS, or a learned smoother) tuned against GT jitter. The GT is itself jittery
     box data, so do not over-smooth.
   - Optionally refine the homography with extra manual court landmarks (key corners, free-throw line, arc ends)
     and lens-distortion correction. The GT frame is defined by the other camera's 4 corners, so check that the
     corner fit stays exact.
3. **Team by bib colour** is almost trivial with 4 known colours. Use it to reject distractors (referee,
   staff, waiting team, the purple-bib scorekeeper) and to resolve the shared 6/9 numbers in IMG_0114.
4. **Jersey number by tracklet voting.** Given the team, it is a 3-way choice. Combine a public OCR/VLM
   (PARSeq, Qwen2.5-VL/Qwen3-VL) with a small classifier trained on training-set crops, downscaled to 720p and
   warped to the oblique viewpoint as augmentation (allowed). Weight votes by legibility.
5. **Occlusion and crowding** in the oblique view (players stacked near the key): use a strong pretrained
   detector at upscaled resolution, keep low-confidence boxes in the ByteTrack style, use court-space motion
   gating (max speed is about 30 cm per frame in GT), and fill gaps with motion-model interpolation.
6. **Pre-game frames count.** GT has 6 players from frame 1. Handle warm-up and lineup periods; the training
   game_start_frame annotations tell you how long these last.
7. **A local validation proxy is required** because there is one blind submission. Run the full pipeline on the
   8 training games (end-line view; download the 7 missing videos first) for identity and tracking. For oblique
   localization, render synthetic oblique views or simulate noise with the sensitivity tool. Include the
   pre-game segment in validation.
8. **Engineering.** Decode-count every frame and never trust metadata. Write exactly N×6 rows. Validate the
   format with `tools/validate_submission.py`. Submit early and resubmit as you improve; only the last one counts.

## 9. Corrections to earlier research notes

- The site's "Test set" mentions are obsolete. There are only Training and Challenge sets.
- "Submit once" was later relaxed by the results-form email: re-submitting is allowed and the latest counts.
- The paper's Eq. 5 (AssA averaged over TP pairs) is NOT what is scored; the public code averages over GT identities.
- Paper v1 used τ = 0.5 m and older numbers. Use v2 (τ = 1 m = 100 cm).
- The "LOPDO evaluation" wording on the site is a copy-paste error (organizers confirmed).
- Licenses: jersey-number-pipeline is CC BY-NC (fine for a challenge); Ultralytics YOLO is AGPL.
