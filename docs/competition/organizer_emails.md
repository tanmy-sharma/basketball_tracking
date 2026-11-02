# Organizer Emails (verbatim) and Change Log

These emails come from Keisuke Fujii (TrackID3x3 Challenge Organizing Team, Nagoya University).
**Where they conflict with the website, the emails win**, and later emails win over earlier ones.

## Change log (newest first)

| # | Date (approx.) | Change | Supersedes |
|---|---|---|---|
| 5 | By Nov 1, 2026 | Reply to our question: **offline, non-causal processing of each whole Challenge video is allowed** if automatic | Nothing; resolves an open question |
| 4 | ~Oct 23–24, 2026 | Results form opened: one ZIP with both TXT at root; **resubmission allowed, latest counts**; no feedback | Email 1 "submit once" |
| 3 | Oct 23, 2026 | **Synthetic data allowed** (independent, from the Training set, or from permitted public sources) | Oct 19/21 "Training set only" rule (narrowly relaxed) |
| 2 | Oct 21, 2026 | `Final challenge/court_registration.csv` corrected (RKU0); manual calibration, public checkpoints and Training-set augmentation explicitly allowed | Old registration CSV |
| 1 | ~Oct 19, 2026 | Deadlines extended; no Test set; 5-column format in cm; TI-AssA averaged over GT identities; data-use rules tightened; LOPDO wording withdrawn | Original website rules / format |

Your local `data/Final challenge/court_registration.csv` has RKU0 = (1113, 379), 129 bytes, the same size as the
Drive file dated Oct 21. It overlays correctly on the court, so it is the corrected version.

---

## Email 1: Updates and clarifications

> Dear TrackID3x3 Challenge participants,
>
> Following recent discussions with the ABC 2027 organizers, we would like to share several important updates and clarifications. Some of these replace earlier challenge instructions, so please review them carefully.
>
> **1. Extended deadlines**
>
> Team registration: October 13 → October 27, 2026
> Challenge results: October 27 → November 3, 2026 (23:59 AoE, UTC−12)
> Challenge paper: November 3 → November 10, 2026
>
> **2. Challenge structure and evaluation**
>
> There are two sets: Training and Challenge. There is no separate intermediate Test set or Test-set leaderboard.
> Each team may submit its Challenge predictions once. All frames in the Challenge videos, including those before the game starts, are evaluated.
> TI-HOTA follows the public evaluation implementation. In particular, TI-AssA is averaged equally over ground-truth player identities (team and jersey number), rather than over true-positive instances as described in Eq. (5) of the paper.
>
> **3. Correct results-submission format**
> Submit one headerless, comma-separated text file per Challenge video, named IMG_0114.txt and IMG_0115.txt, with five columns in this order:
>
> `frame_id,court_x,court_y,team_id,jersey_number`
>
> Use original frame IDs starting from 1, court positions in centimeters (not meters), and team IDs A/B. No separate track-ID column is required. This five-column format supersedes the earlier proposed format. The results-upload link and instructions will be provided separately.
>
> **4. Important clarification of data-use rules**
>
> Training and fine-tuning are limited to the official Training set. Other external training datasets, including publicly released TrackID3x3 annotations outside this set, are not permitted.
> Publicly available general-purpose pretrained models are allowed only if they have not been trained or fine-tuned on the corresponding Challenge sequences or annotations. Disclose the checkpoints, sources, and additional training data used.
> Self-training, pseudo-label training, or weight fine-tuning on the Challenge videos is not allowed. Normal online tracking-state updates during inference are allowed.
> Manual selection of fixed court landmarks for calibration is allowed, but manual frame-by-frame player annotation, identity assignment, or correction is prohibited. Publicly available same-game ground-truth trajectories must not be used to generate Challenge predictions.
>
> **5. Paper and reproducibility materials**
> Please preserve the exact code, model weights, and configuration used for your November 3 predictions. We are coordinating how to upload supporting code and execution instructions with the November 10 paper via CMT; the details will follow. No additional CMT registration is needed for team registration at this stage.
>
> The earlier website reference to "LOPDO evaluation" does not apply to TrackID3x3. Please cite the TrackID3x3 dataset paper.
>
> For the challenge details, resources, and subsequent submission instructions, please check the official website:
> https://sites.google.com/view/abc2027-trackid3x3-challenge/the-challenge
>
> If the clarified data-use rules affect your work already underway, please contact us promptly at trackid3x3-challenge-2026@g.sp.m.is.nagoya-u.ac.jp.

## Email 2: Corrected court registration

> We corrected an error in Final challenge/court_registration.csv.
>
> If you downloaded this file earlier, please re-download the corrected version:
> https://drive.google.com/file/d/1HOS2t-eodxaWYBUXG5R3KEpYMR52NYbP/view
>
> For clarity, manual calibration using fixed court landmarks, publicly available pretrained checkpoints not trained on the Challenge sequences, and augmentation of the official Training data are allowed. Any additional training must use only the official Training set; manual player annotation is prohibited.

## Email 3: Synthetic data amendment

> We are amending the October 19/21 training-data rule to permit synthetic data generated independently, from the official Training set, or using publicly available source material permitted for this use. We apologize for the late revision.
>
> Non-public real-world videos/annotations and Challenge evaluation sequences, synchronized views, or corresponding annotations remain prohibited as sources for training or synthetic-data generation, directly or indirectly. This exception does not permit direct training of the prediction model on external real footage or its ordinary augmentations. The existing public pretrained-checkpoint exception remains unchanged.
>
> Please disclose your sources and generation/training procedures in the paper and supplementary materials. This policy applies equally to all teams, including eligible synthetic data created before this update.
>
> All other rules and deadlines remain unchanged.

## Email 4: Results Submission Form

> The official Results Submission Form is now available:
> https://forms.gle/rMwK4pHm1Xp2joV27
>
> Please upload a single ZIP file containing IMG_0114.txt and IMG_0115.txt (with no additional folders). Each TXT must use the announced five-column, headerless format.
>
> Deadline: November 3, 2026, 23:59 AoE (UTC−12).
>
> You may submit more than once; only the latest submission received before the deadline will be considered for evaluation. Please use the same registered team name and contact email for every submission. No intermediate scores or feedback will be provided.
>
> Paper, code, checkpoints and README will be collected separately; the November 10 paper deadline remains unchanged.

## Email 5: Offline processing confirmed (private reply to our question)

> Dear Tanmay,
>
> Thank you for your question. Yes, the offline processing you describe is permitted. The TrackID3x3 Challenge does not require online-only or strictly causal inference; the reference to online tracking-state updates is an example of allowed inference-time processing.
>
> You may process each Challenge video as a whole, using both past and future frames for automatic assignment of tracklets to the rostered identities, bidirectional trajectory smoothing, and gap interpolation.
>
> As you proposed, these steps must remain automatic inference/post-processing, without manual player annotation, identity assignment or correction, self-training, pseudo-label-based training, or model-weight updates using Challenge videos. Challenge ground-truth annotations or trajectories must not be used.
>
> Please describe these steps in your paper and preserve the exact code, weights, and configuration used to generate your submitted predictions.
>
> Best regards,
> Keisuke Fujii
> TrackID3x3 Challenge Organizing Team
