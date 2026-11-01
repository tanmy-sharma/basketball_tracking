# Related Work and Tools

Gathered by a research pass and spot-checked. Items marked **(unverified)** came from search summaries; check them
before citing in the paper. Licences matter for what you redistribute with the code ZIP.

## 1. SoccerNet Game State Reconstruction (closest analogue; GS-HOTA ≈ TI-HOTA)
- Task and metric: Somers et al., arXiv 2404.11335; code https://github.com/SoccerNet/sn-gamestate.
- **2024 winner, "From Broadcast to Minimap"** (Constructor Tech, arXiv 2504.06357): GS-HOTA **63.81**
  (DetA 49.5, AssA 82.2) vs baseline 23.4.
  - YOLOv5m detector, fine-tuned; DeepSORT run *in pitch coordinates* with physical motion constraints.
  - ResNet-50 ReID; team via OSNet features + 3-way clustering (2 teams + referees).
  - Jersey: a small CNN on 32×32 torso crops with 2 digit heads.
  - Calibration via SegFormer camera-parameter regression + keypoint refinement, partly trained on **synthetic** game-engine images.
  - **Post-processing is where the gains came from**: split tracklets so each has one team and one number, merge by jersey
    number under time and reachability constraints, merge by ReID, then interpolate.
- **2025 winner, KIST-GSR / "Broadcast2Pitch"** (code https://github.com/yinmayoo185/SoccernetGSR; WACV 2026):
  GS-HOTA ≈ **63.9** (unverified). YOLOX + OSNet ReID + **ID-Aware Tracklet Refinement**; jersey via CLIP or a LoRA-tuned
  VLM. Extended version: Broadcast2Pitch++ (MVA 2026), with depth-aware tracking and soft tracklet merging (unverified).
- Lesson for us: **tracklet-level identity reasoning plus reachability-constrained merging** is the common winning pattern.
  Our case is easier (static camera, 6 known IDs, 4 known bib colours) except for localization strictness.

## 2. Jersey number recognition
- Koshkina & Elder, *A General Framework for Jersey Number Recognition in Sports Video*, CVPRW 2024. Code:
  https://github.com/mkoshkina/jersey-number-pipeline (CC BY-NC). Pose-guided torso crop, legibility classifier (ResNet-34),
  PARSeq STR, tracklet aggregation. This is the TrackID3x3 baseline component.
- Vision-language models (Qwen2.5-VL / Qwen3-VL, Florence-2, etc.) can read numbers zero-shot from upscaled crops
  (no verified sports benchmark found). They are allowed as public checkpoints. Use them for tracklet voting.
- Our setting is a closed set: given the team, only 3 candidate numbers, of which {6, 9, 10, 11} are confusable
  (6↔9 under rotation; 10↔11). A tiny classifier trained on training-set crops (allowed) can do the 3-way choice well.

## 3. Team assignment
- Roboflow `sports` (MIT): SigLIP embeddings → UMAP → KMeans(2). https://github.com/roboflow/sports
- Colour histograms (TrackID3x3 baseline, JS divergence). With bright distinct bibs, HSV hue clustering is enough.
  Use more than 2 clusters to absorb the referee, staff and waiting team.

## 4. Detection
- YOLOX (MIT; SportsMOT/ByteTrack weights): what the annotation and baseline used.
- RF-DETR (Roboflow, Apache-2.0 for N–L sizes; https://github.com/roboflow/rf-detr): a strong COCO-pretrained detector.
- Ultralytics YOLOv8/YOLO11 (AGPL-3.0): fine for a challenge, but the licence is viral if you redistribute.
- Co-DETR / CrowdHuman-pretrained models handle crowded scenes better (not evaluated here).
- At 720p, players are 60–130 px tall, so upscaling ×1.5–2 or using SAHI-style tiling helps recall.

## 5. Multi-object tracking
- ByteTrack, BoT-SORT (MIT): reliable baselines; BoT-SORT includes camera-motion compensation (not needed here).
- **CAMELTrack** (Apache-2.0): the TrackID3x3 v2 tracker, with learned association and HOTA > 80 on SportsMOT.
- **McByte** (CVPRW 2025, https://github.com/tstanczyk95/McByte): training-free, mask-guided, designed for sports.
- MOTIP (CVPR 2025, https://github.com/MCG-NJU/MOTIP), SambaMOTR (ICLR 2025): learned end-to-end trackers, but
  they need training data we don't have for this view.
- **SportsSUSHI** (arXiv 2502.21242, https://github.com/mkoshkina/sports-SUSHI): hierarchical graph tracking that
  uses jersey number, team and field coordinates. Very aligned with this task.
- SAM2 / SAMURAI-style mask propagation: strong through occlusion, and allowed as a public checkpoint. Slow, but
  only 23k frames in total.

## 6. Pose / ground-contact estimation (for localization)
- RTMPose / RTMW, ViTPose, HRNet (all public; the TrackID3x3 baseline used RTMPose, HRNet and SwinPose). Ankle keypoints give a
  better ground contact point than the box bottom, especially during jumps and for overlapping boxes.
- Monocular metric depth or human mesh (e.g. 4DHumans, SMPL-based) to resolve jumping. Probably overkill.

## 7. Basketball-specific
- Oo et al., *Jump-Aware Player Position Rectification and Identification*, CVPRW 2025 (basketball/volleyball).
  **Directly relevant**: corrects court position when a player is airborne.
  https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Oo_Jump-Aware_Player_Position_Rectification_and_Identification_in_Dynamic_Sports_Using_CVPRW_2025_paper.pdf
- DeepSportradar basketball player re-ID challenge (not reviewed).

## 8. Prior ABC challenges
ABC challenges (Activity and Behavior Computing) are usually sensor or activity-recognition contests run with
Sozo Lab. Winning papers are typically method write-ups of 8–12 pages. No public results on vision tracking
challenges were found. Expect a small field, where a careful, well-validated pipeline can win.
