# Submission Format, Timeline and Deliverables

## Timeline (today: Sun Oct 25, 2026)

| Date | Milestone | Days left |
|---|---|---|
| **Tue Oct 27, 2026** | **Team registration closes**: https://docs.google.com/forms/d/e/1FAIpQLScl7duLqA5MUZMO4Lz-k4JNYty8IjSwBKWypIE63f9XBoNFag/viewform | **2** |
| **Tue Nov 3, 2026, 23:59 AoE (UTC−12)** | **Results due** (= Wed Nov 4, 11:59 UTC = Wed Nov 4, 17:29 IST) | 9 |
| Tue Nov 10, 2026 | Paper + code + weights + README (CMT; details TBA) | 16 |
| Jan 19–22, 2027 | ABC 2027 workshop, Sydney (hybrid); a team member presents | |

Resubmission is allowed and the latest counts, so **submit a valid baseline early** (e.g. by Oct 29) and
overwrite it as you improve.

## Results file format

Form: https://forms.gle/rMwK4pHm1Xp2joV27. Upload **one ZIP** with **exactly** these files at its root:

```
submission.zip
├── IMG_0114.txt
└── IMG_0115.txt
```

Each TXT is headerless, comma-separated, with 5 fields per line:

```
frame_id,court_x,court_y,team_id,jersey_number
1,412.37,655.10,A,10
1,980.02,301.55,B,6
...
```

| Field | Type | Rules |
|---|---|---|
| `frame_id` | int | Original video numbering **starting at 1**. Every frame is scored. IMG_0114: 1…11536, IMG_0115: 1…11823 (ffprobe `nb_frames`; confirm with a full decode count) |
| `court_x` | float, **cm** | Along the end line, 0…1505 on court (can be negative or above 1505 off court) |
| `court_y` | float, **cm** | From the end line (hoop side, y = 0) to the far line (y = 1105) |
| `team_id` | `A` or `B` | As in `Final challenge/games_roster.csv` |
| `jersey_number` | int | Must be in that team's roster |

### Pitfalls that silently score zero
- **Spaces.** The official loader splits on `,` without stripping, and identity is the string `team+jersey`.
  `A, 10` or `A,10 ` gives an identity mismatch. Write `A,10`.
- **Metres instead of cm.** Every distance error becomes about 100× too small relative to τ, and nothing matches.
- **Frame numbering from 0.** Every prediction lands one frame early. That is a systematic offset (~4 cm median, up to ~25 cm while sprinting), and frame `0` itself is never scored.
- **Team-letter swap.** If A/B are flipped, identities like `A6`/`B6` swap and nearly everything is wrong.
  Cross-check against the roster (IMG_0114: the team with #10 is A, the team with #11 is B).
- **Extra people or duplicate identities.** Each extra row is an FP. Write exactly one row per roster identity per frame.
- **Missing frames.** Every GT player on a missing frame is an FN. Always write all 6.
- Folders inside the ZIP, or extra files, may break the organizers' pipeline.

Validate with `tools/validate_submission.py` (it also builds the ZIP).

## Coordinate frame (verified on the data)

```
court_y = 0  ── end line, hoop side (hoop centre ≈ (752.5, 157.5))
   RKU0 (0,0) ─────────────────────── RKU10 (1505,0)
      │                                   │
      │            3x3 court               │
      │       15.05 m wide x 11.05 m deep  │
   RKU1 (0,1105) ──────────────────── RKU11 (1505,1105)
court_y = 1105 ── far line (top of the court; both cameras are on this side)
```

Keypoints are at the **centre of the 5 cm painted lines** (paper §3.1). In the challenge image, RKU0 is the
far-right corner at the end line and RKU11 the near-left corner.

## Deliverables after Nov 3

| Item | Due | Contents |
|---|---|---|
| Code ZIP | Nov 10 (with the paper) | Source, configs, README with exact commands, environment (requirements/lockfile), seeds |
| Weights | Nov 10 | Files or fixed-version download links, **identical to those used for the final submission** |
| Paper | Nov 10 | 8–12 pages, ABC template, sections listed in [rules_and_data_policy.md](rules_and_data_policy.md) §5 |
| Disclosures | in paper | All checkpoints and sources, synthetic-data procedure, manual calibration points, hardware/runtime |

**Freeze discipline:** when you submit, tag the code (`git tag submission-YYYYMMDD-HHMM`), hash every weight
file (`shasum -a 256`), and store the exact submitted ZIP. Because resubmission is allowed, keep a log of which
tag produced which upload.
