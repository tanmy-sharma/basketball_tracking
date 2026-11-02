# First detector and tracker run (IMG_0106)

RF-DETR Large (COCO weights, 720p input), all 11,272 frames, 40 fps on Colab A100. Results zip: code version `8224dc69`.
Training-view, single-game numbers: use them to compare options, not to predict the challenge score.

## Detection vs ground truth (IoU 0.5)

| Court margin | Recall at conf 0.5 | Precision | F1 | False boxes/frame | Recall at conf 0.05 |
|---|---|---|---|---|---|
| 150 cm | 0.943 | 0.674 | 0.786 | 2.73 | 0.982 |
| 100 cm | 0.941 | 0.756 | 0.839 | 1.82 | 0.981 |
| 50 cm | 0.939 | 0.828 | 0.880 | 1.17 | 0.979 |
| 25 cm | 0.937 | 0.852 | 0.893 | 0.97 | 0.977 |
| 0 cm | 0.934 | 0.856 | 0.893 | 0.94 | 0.974 |

- Recall at conf 0.7 falls with overlap: 0.986 with no overlap, 0.848 at IoU 0.1-0.3, 0.541 at 0.3-0.5, 0.400 above 0.5.
- Most of the false boxes left at margin 0 are one person standing just inside a court corner: 9,751 boxes, about 0.87 per
  frame, at nearly the same spot for almost the whole video.
- 0.7% of GT foot points are outside the court lines, so margin 0 loses some real players.
- Frame alignment holds at the start, middle and end of the video. The ffmpeg dts warnings in the log are not a frame shift.

## Tracker results

Box HOTA by margin (image IoU, global track IDs; no team or jersey):

| Config | 150 | 100 | 50 | 25 | 0 |
|---|---|---|---|---|---|
| bytetrack__default | 23.8 | 23.4 | 24.6 | 27.7 | 25.7 |
| cbiou__default | 24.9 | 24.1 | 24.7 | 24.6 | 24.8 |
| ocsort__default | 22.0 | 22.7 | 23.4 | 23.6 | 23.7 |
| botsort__default | 21.8 | 21.8 | 22.9 | 23.3 | 23.4 |
| cbiou__lowthr | 21.1 | 21.2 | 22.2 | 22.5 | 22.4 |
| botsort__lowthr | 19.3 | 20.4 | 21.1 | 21.3 | 21.6 |
| bytetrack__lowthr | 18.8 | 20.9 | 21.8 | 22.0 | 21.5 |
| ocsort__lowthr | 19.4 | 19.5 | 19.8 | 20.3 | 20.5 |
| sort__lowthr | 16.6 | 18.0 | 18.5 | 18.8 | 18.9 |
| sort__default | 15.8 | 17.0 | 17.7 | 18.0 | 18.1 |

Going from margin 150 to 25 lifts DetA from about 55 to 68 and MOTA from about 50 to 78, while AssA stays near 5-11.
Identity continuity is what holds HOTA down, not detection. ByteTrack's 27.7 at margin 25 comes from a jump in AssA that its
neighbouring margins don't show, so don't choose a tracker from it.

Share of GT boxes (best two runs at margin 150): tracked 0.95-0.96, ID switch 0.002-0.003, tracker miss 0.015-0.026,
low-score miss 0.004, detector miss 0.017. Tracks per player (median): 5 to 38, where 1 is ideal.

## What TI-HOTA these tracks could reach (ground truth used as an oracle)

| Setup, margin 150 | TI-HOTA |
|---|---|
| Detections with perfect per-frame identity, conf >= 0.05 | 0.819 |
| Same, conf >= 0.5 | 0.809 |
| Same at conf >= 0.5, constant offset removed (fitted on this video) | 0.855 |
| Whole-track perfect label: sort__default (38.5 tracks per player) | 0.733 |
| Whole-track perfect label: cbiou__default (8 tracks per player) | 0.530 |
| Whole-track perfect label: bytetrack__default (5 tracks per player) | 0.367 |

Longer tracks score worse once each track gets one identity, since they contain undetected identity mix-ups. Short, pure
tracks joined later by colour and jersey look better than a long-lived tracker. Median court error at conf 0.5 is 12.5 cm,
biased -11.3 cm in y.

## UCMCTrack in court space (`src/tracker/`)

Runs on the same cached detections. Output rows are limited to inside the court lines. Score is TI-HOTA with one
ground-truth label per track, using the same inside-court filter for every tracker.

This score always rises as tracks get shorter, reaching about 0.82 with many tiny pieces, so compare trackers only at
similar track counts. The earlier ranking of SORT first ignored this.

| Tracker | Tracks per player | TI-HOTA | Identity purity | Median track length (frames) |
|---|---|---|---|---|
| ByteTrack, default | 6.5 | 0.388 | 0.618 | 488 |
| CBIoU, default | 8.5 | 0.547 | 0.785 | 906 |
| UCMCTrack, tuned | 8 | 0.557 | 0.810 | 546 |
| BoT-SORT, default | 12.5 | 0.591 | 0.825 | 551 |
| OC-SORT, default | 16 | 0.657 | 0.886 | 418 |
| UCMCTrack, tuned | 15 | 0.703 | 0.932 | 454 |
| SORT, low thresholds | 27 | 0.701 | 0.920 | 192 |
| UCMCTrack, tuned | 24.5 | 0.733 | 0.943 | 240 |
| SORT, default | 39.5 | 0.739 | 0.946 | 45 |
| **UCMCTrack, tuned (now the default)** | **38** | **0.787** | **0.978** | **149** |

UCMCTrack settings were chosen from 96 combinations on this one video, while the others ran at library defaults, so this
favours UCMCTrack. Upstream's default settings scored 0.58 at 20 tracks per player. Filtered and raw positions score almost the
same at the default settings (0.787 and 0.789), so the Kalman filter adds no position accuracy here.

Plain box scores with no ground-truth identity help, inside the court only (image IoU, global track IDs):

| Tracker | HOTA | DetA | AssA | IDF1 | MOTA | ID switches | Fragments | Track IDs |
|---|---|---|---|---|---|---|---|---|
| CBIoU, default | 27.0 | 67.6 | 10.8 | 22.3 | 78.1 | 145 | 545 | 55 |
| ByteTrack, default | 25.9 | 67.5 | 10.0 | 20.6 | 78.3 | 190 | 658 | 52 |
| OC-SORT, default | 24.1 | 65.5 | 8.9 | 19.1 | 75.3 | 156 | 641 | 95 |
| BoT-SORT, default | 23.6 | 67.9 | 8.2 | 17.2 | 78.0 | 140 | 526 | 78 |
| UCMCTrack, tuned for short clean pieces | 18.2 | 67.2 | 5.0 | 12.2 | 76.3 | 321 | 816 | 253 |
| SORT, default | 17.6 | 67.9 | 4.6 | 9.0 | 77.6 | 395 | 710 | 328 |

Even with ground-truth boxes as input, these UCMCTrack settings produce 35 tracks over the whole video instead of 6.
There is no real TI-HOTA yet, because the output has no team or jersey labels.
