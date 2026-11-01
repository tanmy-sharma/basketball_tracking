# TI-HOTA: How You Are Scored (in depth)

Source of truth: the public implementation in `open-starlab/TrackID3x3/script/compute_TI-HOTA.ipynb`, copied in
[reference/compute_TI-HOTA_official.py](reference/compute_TI-HOTA_official.py). The organizers confirmed (Email 1)
that this code, **not** paper Eq. 5, is what will be used. Everything below was derived by reading that code line by line.

---

## 1. Definition (as implemented)

For each video:

```
τ = 100 (cm)
LocSim(P,G) = exp( ln(0.05) · ‖P−G‖² / τ² )              # 1 at 0 cm, 0.05 at 100 cm
IdSim(P,G)  = 1 if attr(P) == attr(G) else 0              # attr = team+jersey string, e.g. "A10"
Sim         = LocSim · IdSim

for α in {0.05, 0.10, …, 0.95}:                            # 19 thresholds
    for each frame:
        Hungarian assignment on cost = 1 − Sim, with Sim < α forbidden
        TP += #matches,  FP += #pred − #matches,  FN += #gt − #matches
        per GT identity g:  TPA[g] += matched,  FNA[g] += unmatched
        per unmatched prediction p:  FPA[attr(p)] += 1
    DetA_α = TP / (TP + FP + FN)
    AssA_α = mean over GT identities g of  TPA[g] / (TPA[g] + FNA[g] + FPA[g])
    HOTA_α = sqrt(DetA_α · AssA_α)

TI-HOTA = mean_α HOTA_α        (then presumably averaged over IMG_0114 and IMG_0115)
```

The reported DetA and AssA are also means over α.

## 2. What this code really measures

1. **No temporal association term.** The GT `id` column is parsed and never used. There is no ID-switch,
   fragmentation, or track-continuity term. "AssA" here is per-identity *frame-level* accuracy:
   for each of the 6 players, the fraction of frames where the row you labelled with their identity lies
   near their true position, penalized by frames where you labelled someone else with it.
   → **What matters is getting the right label at the right place on each frame.** How you get there (tracking, global
   assignment, smoothing) is up to you.
2. **Identity is binary and exact.** The wrong team or the wrong number gives Sim = 0 regardless of distance.
3. **Because identities are unique per frame**, Hungarian matching reduces to: for each identity g, match iff
   your g-row is within the distance threshold of GT g. (If you output two rows with the same identity, one is
   always an FP.)
4. Every GT identity is a "track" for AssA, so the 6 identities weigh equally per video regardless of how
   many frames they appear in (they appear in all frames anyway).

## 3. Localization tolerance

Maximum distance that still counts as a match at each α (τ = 100 cm):

| α | .05 | .10 | .20 | .30 | .40 | .50 | .60 | .70 | .80 | .90 | .95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| max error (cm) | 100 | 88 | 73 | 63 | 55 | 48 | 41 | 35 | 27 | 19 | **13** |

Fraction of the 19 thresholds a correctly identified row passes, as a function of its error:

| error (cm) | 0–13 | 15 | 20 | 25 | 30 | 40 | 50 | 60 | 75 | 90 | >100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fraction of α passed | 1.00 | 0.95 | 0.89 | 0.84 | 0.79 | 0.63 | 0.47 | 0.32 | 0.16 | 0.05 | 0 |

Context: 13 cm is about a shoe length. A player's two feet are 30–80 cm apart mid-stride, and in GT the
"position" is a **bounding-box bottom-centre** in another camera ([03_dataset.md](03_dataset.md) §4). Moving that
point by 5 % of box height moves it about 30 cm on court. **The top quarter of the α range is effectively a
competition on matching the annotation convention.**

## 4. Algebra that drives strategy

Take one video, one α, 6 GT players per frame. Let p be the fraction of GT rows you match.

**Case A: always output 6 rows** (wrong ones become an FP plus an FN):
TP = 6p, FP = 6(1−p), FN = 6(1−p) → DetA = p / (2−p); per-identity AssA similarly ≈ p / (2−p).
→ **HOTA_α ≈ p / (2 − p)**: 0.9 → 0.82, 0.8 → 0.67, 0.6 → 0.43.

**Case B: output only rows that will match** (perfect abstention; wrong rows are dropped):
TP = 6p, FP = 0, FN = 6(1−p) → DetA = p, AssA ≈ p → **HOTA_α ≈ p**.

So **a wrong row costs about twice as much as a missing row.** In practice you don't know which rows are wrong.
For a candidate row with probability q of matching at a given α, the marginal effect on DetA = TP/(TP+FP+FN) is
positive iff

```
q > DetA / (1 + DetA)        ≈ 0.44 when DetA ≈ 0.8,  ≈ 0.47 when DetA ≈ 0.9
```

The same row is scored at all 19 α. It might be within 40 cm (passes α ≤ 0.6) but not 15 cm. So the right
decision is roughly: **output the row if, averaged over α, its chance of matching exceeds about 45 %.**

Implications:
- During short occlusions, interpolated positions are usually within 50 cm, so **output them**.
- During long gaps where you have lost the player, or when the identity is a coin flip between two players,
  **omitting the row can beat guessing.**
- Never output a 7th row "just in case": it is pure FP.
- For identity confusion between two same-team players (e.g. A9 vs A10 after a crossing), guessing wrong
  costs FP+FN on *both* identities, while omitting both costs only 2 FN. When identity confidence is low, abstain.

This is exactly why the paper's baseline caps rows at 6 and why ID-switch merging gains ~5 points.
`tools/sensitivity.py` measures these effects empirically on training GT; see `tools/sensitivity_results.md`.

## 4b. Measured sensitivity (training GT as "predictions", mean over all 8 games)

Produced by `tools/sensitivity.py` (corruptions in 30-frame blocks, seed 0). Full table: `tools/sensitivity_results.md`.

| Corruption | TI-HOTA |
|---|---|
| Gaussian noise σ = 5 / 10 / 20 / 30 / 50 cm (per axis) | 0.997 / 0.943 / **0.755** / 0.554 / 0.280 |
| Constant bias 10 / 20 / 30 / 50 cm | 1.000 / 0.895 / 0.790 / 0.474 |
| Team letters swapped on 5 / 10 / 25 % of frames | 0.924 / 0.853 / 0.663 |
| Two same-team IDs swapped on 5 / 10 / 25 % of frames | 0.968 / 0.938 / 0.857 |
| One player missing on 10 / 25 / 50 % of frames | 0.983 / 0.958 / 0.917 |
| One player held at a stale position on 10 / 25 / 50 % of frames | 0.982 / 0.953 / 0.902 |
| 7th spurious row on every frame | 0.926 |

Reading it:
- **Localization noise dominates everything.** Per-axis jitter of 20 cm already costs a quarter of the score.
  A constant bias is cheaper than the same amount of random noise, and below 13 cm it is free. So
  **temporal smoothing (reduce variance) and convention matching (reduce bias) are both first-order.**
- **Team errors cost about twice as much as same-team swaps**, because they hit 3+ identities at once.
- **Missing and stale are nearly equal** for one player. Holding a stale position is no better than dropping
  the row, which matches the abstention algebra in §4. Prefer a good interpolation; when you have none, dropping the row is fine.
- A constant extra row costs about 7 %. Never emit one.

## 5. Per-video averaging

The paper and notebook score each video separately and take the mean. With 2 videos, **each one is 50 % of the
score.** A team-letter swap or a systematic failure on one video (e.g. IMG_0114's shared numbers 6/9) costs half
the competition.

## 6. Paper vs. code differences (know them for the paper write-up)

| Aspect | Paper (v2) | Code (what is scored) |
|---|---|---|
| AssA averaging | over TP pairs (Eq. 5) | over GT identities (organizers confirmed) |
| Association key | GT track | attribute string (team + jersey) |
| τ | "1 m" | `tau = 100` in coordinate units; challenge coordinates are cm, so consistent |
| v1 of the paper | τ = 0.5 m, BoT-SORT-ReID baseline | obsolete |

Other implementation quirks:
- `load_frames` needs at least 5 comma fields and treats field 5 as the attribute string. The challenge GT/pred
  format differs (5 columns with team and jersey separate). The organizers will adapt the loader; the safest
  output is exactly their format with **no whitespace**.
- Frames present in only GT or only predictions are still evaluated: all GT rows on frames you omit are FN.
- An official-faithful and a fast vectorized port live in `tools/trackid/metric.py`, and tests assert they agree.
