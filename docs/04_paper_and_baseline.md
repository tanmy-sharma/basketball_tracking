# The TrackID3x3 Paper and Official Baseline

Paper: Yamada et al., *TrackID3x3*, MMSports '25 (arXiv 2503.18282 **v2**, Aug 2025). A local copy is at
[reference/TrackID3x3_arXiv2503.18282v2.pdf](reference/TrackID3x3_arXiv2503.18282v2.pdf).
Repo: https://github.com/open-starlab/TrackID3x3 (code Apache-2.0; `jersey-number-pipeline/` CC BY-NC 3.0;
dataset CC BY 4.0).

## 1. Dataset subsets in the paper

| Subset | Videos | Frames | Camera | Resolution | Identity |
|---|---|---|---|---|---|
| Indoor | 42 short clips | 7,531 | Sony HDR-CX680, fixed | 1280×720 | team (O/D) + initial position (top/left/right); bibs have no numbers |
| **Outdoor** | **12 games** | 143,276 | **iPhone 13, fixed (1 of 2 cameras used)** | 3840×2160 | **team + jersey number** |
| Drone | 6 | 127,119 | DJI Air 2S | 3840×2160 | tracking only, no court annotation |

The challenge is drawn from **Outdoor**: 14 male university players, 4 teams of 3, double round-robin, 5-minute games
under FIBA 3x3 rules with no timeouts and no 21-point knockout. The court is 11.05 × 15.05 m and the keypoints are
annotated at line centres. Annotation: half manual, half BoT-SORT-ReID output corrected by hand; YOLOX was the detector.

## 2. Baseline pipeline ("Track-ID algorithm", §4.2)

1. **Detection + tracking**: CAMELTrack with a pre-trained YOLOX (the same weights used during annotation),
   with post-processing deliberately set to *reduce* detection power. Association weights are pre-trained on SportsMOT.
   No fine-tuning.
2. **Court projection**: homography from the manually annotated court keypoints; position = **bbox bottom-edge midpoint**.
3. **Non-player removal**, rule-based:
   - Detection level: drop detections more than 3 m outside the court lines.
   - Tracklet level: drop tracklets that (a) are on court for fewer than 10 consecutive frames, (b) spend more than half their
     frames in the end-line band (3 m outside to 1 m inside), or (c) spend more than half their frames outside the paint's
     long-side extensions and more than 10 m from the end line ("coffin corner"). (b) and (c) target referees.
4. **ID-switch detection and merging** (Algorithm 1): IDs that appear mid-video are merged into an original ID if
   their co-existing on-court frame count is below T_overlap (Outdoor: 50). Cost = overlap + missing frames.
   It cannot handle *exchanges* between two existing IDs.
5. **Jersey number**: tracklet-level jersey-number-pipeline (Koshkina & Elder, CVPRW 2024: pose-guided torso crop,
   legibility classifier, scene-text recognition, then tracklet consolidation). Tracklets with no recognized number are dropped
   (this also removes referees).
6. **Team**: 8×8×8 3D colour histograms of torso crops (median over up to 100 frames), compared with
   Jensen–Shannon divergence. Offense = the tracklet nearest the check-ball spot (or free-throw line) at the first frame;
   the 2 tracklets most similar to it = offense; the rest = defense.
7. **Post-processing**: linear interpolation of gaps, endpoint extrapolation, cap at 6 rows per frame (drop the tracks
   with the fewest on-court frames).

## 3. Reported results (Table 3, τ = 1 m)

| Subset | Setting | TI-HOTA | TI-DetA | TI-AssA |
|---|---|---|---|---|
| Indoor | with merging | 85.53 ± 7.09 | 84.86 | 86.23 |
| **Outdoor** | **with merging** | **71.03 ± 17.79** | **67.15** | **75.51** |
| Outdoor | without merging | 66.20 ± 18.07 | 61.97 | 71.08 |

Important context:
- Outdoor was scored on **in-play segments only** (96 segments; 74 start with a check-ball, 22 with a free throw;
  1203 ± 1026 frames each). **The challenge scores whole videos including pre-game and dead balls**, which is harder.
- The paper's AssA follows Eq. 5; the challenge uses the code's per-identity version. The numbers are not directly comparable.
- It was evaluated on the **end-line** camera, the same view the detector annotations came from. Expect the
  baseline to do worse on the oblique 720p challenge view.
- The large SD (±17.8) shows that some segments fail badly, typically from ID exchanges that persist (Fig. 3 of the paper).

## 4. Failure modes the paper itself names
- **ID exchange** between two players during occlusion. After the exchange, the attributes are wrong for the rest
  of the track. Global identity re-assignment is needed.
- Detectors that are too strong pick up non-players. The authors weakened YOLOX on purpose.
- Pose estimation is weakest on wrists and at low resolution (Indoor 720p).

## 5. Using the official repo legally
- ✅ Code (BoT-SORT, CAMELTrack, scripts) and **public checkpoints** (YOLOX/SportsMOT, CAMELTrack, scene-text recogniser and legibility
  weights trained on hockey/SoccerNet) are allowed. Disclose them.
- ❌ `ground_truth/Outdoor/**` and any Outdoor pose or keypoint annotations are outside the official Training set.
  **Do not use them.** Don't even keep them in the project tree, so you can certify that cleanly in the paper.
- ⚠ CAMELTrack's association weights were trained on SportsMOT by its authors. That counts as a public pretrained
  checkpoint, so it is allowed. You training on SportsMOT yourself is not.
