# Rules and Data-Use Policy (consolidated)

The rules are merged from the website's Registration and Dataset pages and the four organizer emails
([organizer_emails.md](organizer_emails.md)). Every rule is tagged with its source:
**[E1]–[E4]** = emails, **[REG]** = Registration page, **[DS]** = Dataset & Tutorial page.
Where sources conflict, the latest email applies.

---

## 1. Eligibility and team rules

| Rule | Source |
|---|---|
| Every participant, including solo challengers, must complete the **Team Registration Form by Oct 27, 2026** | [REG], [E1] |
| Open to students and working individuals; no limit on team size | [REG] |
| Students from the host or organizing labs (STAL Lab Nagoya, Sozo Lab Kyutech) are **not eligible** | [REG] |
| A student may join several ABC challenges, but only supervisors may be on more than one team in the same challenge | [REG] |
| Team mergers or member/email changes must be emailed to the organizers (subject "ABC Challenge") | [REG], [FAQ] |
| No separate CMT registration is needed yet | [E1], [REG] |
| Use the **same registered team name and contact email** on every results submission | [E4] |
| Private sharing of competition code or data outside the team is prohibited (except required submissions) | [REG] |

## 2. Training data: what you may learn from

### Allowed
| Item | Source | Notes |
|---|---|---|
| The **official Training set** (IMG_0104–IMG_0111: videos, bboxes, court positions, rosters, registration) | [E1], [REG] | Training, fine-tuning and local validation |
| **Augmentation** of the official Training data | [E2] | Includes geometric/viewpoint warps, downscaling to 720p, colour jitter, etc. |
| **Publicly available general-purpose pretrained checkpoints**, if not trained or fine-tuned on the challenge sequences or annotations | [E1], [E2], [E3] | COCO/CrowdHuman/SportsMOT detectors, pose models, ReID, OCR, CLIP/SigLIP, VLMs. **Disclose each checkpoint and source.** |
| **Synthetic data** generated (a) independently, (b) from the official Training set, or (c) from public source material permitted for this use | [E3] | Disclose sources and the generation/training procedure. Applies retroactively to synthetic data made before the update |

### Prohibited
| Item | Source | Why it matters |
|---|---|---|
| Any other external training dataset, **including public TrackID3x3 annotations outside the official set** | [E1], [REG] | The public repo `open-starlab/TrackID3x3` contains `ground_truth/Outdoor`: end-line-view GT for games that probably include the challenge games. **Do not download it into the project, train on it, or use it to validate.** |
| Non-public real-world videos or annotations | [E3] | |
| Challenge evaluation sequences, **synchronized views**, or their annotations, as sources for training or synthetic data, directly or indirectly | [E3] | The challenge games were filmed by two cameras. The end-line camera's footage or GT for IMG_0114/0115 is off limits |
| Training the prediction model directly on **external real footage or its ordinary augmentations**, even under the synthetic exception | [E3] | E.g. you may not fine-tune on SportsMOT yourself. Using a public checkpoint someone else trained on SportsMOT is fine |
| Self-training, pseudo-label training, or weight fine-tuning on the Challenge videos | [E1], [REG] | No test-time training, no TTA that updates weights |

## 3. Inference on the challenge videos

| Item | Status | Source |
|---|---|---|
| Running inference on the challenge videos | Allowed | [REG] |
| Normal online tracking-state updates (Kalman state, appearance memory banks, etc.) | Allowed | [E1] |
| **Manual selection of fixed court landmarks** for calibration (you may click more points than the 4 given) | Allowed; **describe it in the paper** | [E1], [E2], [REG] |
| Manual frame-by-frame player annotation, identity assignment, or correction | **Prohibited** | [E1], [E2], [REG] |
| Using publicly available same-game GT trajectories to generate or copy predictions | **Prohibited** | [E1], [REG] |

### Grey zones (ask the organizers before relying on them)
1. **Offline processing: resolved, allowed** ([organizer_emails.md](organizer_emails.md), Email 5). Each Challenge
   video may be processed as a whole, using past and future frames, for tracklet-to-identity assignment,
   bidirectional smoothing and gap interpolation. These steps must stay automatic, use no Challenge GT, and be
   described in the paper.
2. **Human-set per-video priors**, e.g. typing "yellow = A in IMG_0114" into a config. That is arguably
   manual identity assignment. Derive it automatically (roster plus OCR plus colour clustering) and say so in
   the paper.
3. **Hyper-parameters tuned by looking at challenge videos**, e.g. detector thresholds picked by eyeballing
   challenge frames. These are not weights, so probably fine, but disclose them. Avoid tuning on anything that
   looks like a manual label.

## 4. Evaluation rules

| Rule | Source |
|---|---|
| Two sets only, Training and Challenge; **no Test set and no leaderboard** | [E1] (the DS page still mentions a test set; ignore it) |
| **All frames are evaluated, including pre-game frames** | [E1], [REG] |
| Metric: TI-HOTA as in the public implementation, τ = 100 cm | [E1], [REG] |
| TI-AssA = unweighted mean over GT identities (team + jersey), **not** paper Eq. 5 | [E1], [REG] |
| Multiple submissions allowed; **only the latest before the deadline counts**; no scores or feedback | [E4] (overrides E1's "once") |

Full metric analysis: [../02_evaluation_metric.md](../02_evaluation_metric.md).

## 5. Completion requirements (all three needed to rank)

1. **Predictions** by Nov 3, 23:59 AoE, via the results form ([submission_and_deliverables.md](submission_and_deliverables.md)).
2. **Code and models**: a ZIP with source, configs, a README to reproduce the predictions, and model weights or
   fixed-version download links. **Must be the exact versions used for the Nov 3 predictions.** [REG], [E1]
3. **Paper** by Nov 10 via CMT: English, ABC format, **8–12 pages including references**. Sections: Abstract,
   Methodology, Results & Discussion, Conclusion, References. Must cite the TrackID3x3 paper and the "three main
   challenge papers". Disclose checkpoints, data sources, synthetic-data procedure, and manual calibration
   steps. **At least one member presents at the workshop** (ABC 2027, Sydney, hybrid, Jan 19–22, 2027).
   Accepted papers are published by default in IJABC (J-STAGE). [REG], [E1], [E3]

## 6. Prizes

- Winning team: **100,000 JPY + wearable activity/behaviour-tracking devices** [REG]
- All participants: certificate of participation [REG]
- Only the Google Site states this; the ABC 2027 site had no challenge details when fetched.

## 7. Required citation

> K. Yamada, L. Yin, Q. Hu, N. Ding, S. Iwashita, J. Ichikawa, K. Kotani, C. Yeung, K. Fujii.
> "TrackID3x3: A Dataset and Algorithm for Multi-Player Tracking with Identification and Pose Estimation in
> 3x3 Basketball Full-court Videos." Proc. 8th ACM Workshop on Multimedia Content Analysis in Sports
> (MMSports '25), pp. 163–173, 2025. DOI: 10.1145/3728423.3759400. arXiv: 2503.18282.

The "three main challenge papers" are not named on any page we could read. Ask the organizers which ones, or
check the FAQ images on the site.

## 8. Compliance checklist (keep updated while building)

- [x] Team registered by Oct 27, done 2026-10-26 (team name: ________, contact email: ________ — fill in; must match on every submission)
- [ ] Every checkpoint used is listed with URL, version/hash and its training data (so far: `rfdetr-large`, COCO; add its hash)
- [ ] No file from `open-starlab/TrackID3x3/ground_truth/Outdoor` (or any public Outdoor GT) on disk
- [ ] No weights updated using challenge frames (detector, ReID, OCR, classifier)
- [ ] Manual calibration points recorded (image coords + court coords) and described
- [ ] No per-video manual identity choices in configs
- [ ] Synthetic data generation scripted and documented
- [ ] Code, weights and config frozen and hashed at the moment of final submission
