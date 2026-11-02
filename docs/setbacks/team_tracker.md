# Setbacks: team-aware tracker (2026-11-02)

What we tried for team labels in the tracker that did not work, or worked less well than expected, and why. The things that did
work are in [docs/15_team_tracker.md](../15_team_tracker.md). Unless noted, numbers are oracle TI (each whole track gets its
true identity, so a ceiling), on fine-tuned RF-DETR Small boxes (`rfdetr-small-ft-471b3fff`), current setting (cdt 8,
ambiguity 1).

**Kept setting** (for comparison): ×6 team penalty + colour safeguard + team lock 30, sweep variant `s6_qb_back.4_lock30`. The
safeguard means overlap is measured on all boxes, trust in a call depends on overlap, and back or unclear boxes at IoU ≥ 0.4
get no call. On IMG_0104/0105/0108/0109 it scores TI 0.909/0.933/0.909/0.912, with 0 swaps over 100 frames.

## 1. Detach: no-call boxes can't continue an existing track (reverted)

**Idea:** if a box in a pile-up has no usable team call, don't let an existing track take it; let it start a new tracklet.
That turns identity switches into fragments, which a jersey reader could stitch back.

**Result:**

| Video | Kept setting: TI / AssA / coverage / tracks per player | + detach |
|---|---|---|
| IMG_0104 | 0.909 / 32.2 / 0.968 / 6 | 0.850 / 4.6 / 0.916 / 147 |
| IMG_0105 | 0.933 / 37.8 / 0.984 / 5 | 0.900 / 6.3 / 0.941 / 171 |
| IMG_0108 | 0.909 / 38.4 / 0.972 / 5 | 0.875 / 5.9 / 0.925 / 154 |

Cross-team rows did fall 50–80%. But each player became about 150–170 tracklets, and coverage dropped 4–5 points. On IMG_0104,
same-team rows rose from 26 to 809.

**Why it failed:** guarded boxes are common, since any player partly behind another produces one. A track that can't take them
dies after 8 frames, and the new tracklet needs 3 hits to confirm, so those frames are never output. Oracle TI already stitches
fragments perfectly and still fell 0.03–0.06, so even a perfect jersey reader would not recover this.

**Lesson:** prefer fragmentation over switching only where a switch is actually possible. The candidate is to detach only when
an established track of the opposite team could also take the box (two-track exchanges, about two-thirds of the remaining
swaps). Not yet tested.

## 2. The safeguard does not reduce how often swaps happen

It only makes them short. Swap events stayed the same (IMG_0109: 81 → 85), while the mean length fell from 32.5 to 3.8 frames.

At the first frame of each remaining swap, most switches happen on a box with no call, so position alone decides:
- IMG_0109: 67 of 83 swaps (257 of 314 rows)
- IMG_0104: 63 of 76 swaps (261 of 316 rows)

The biggest group is two players at the same depth ("unclear": feet within 20 px), where both calls are suppressed. Back boxes
are second.

Most swaps are **two-track exchanges**: IMG_0109 57/83, IMG_0104 50/76. Almost none are new tracklets born on the wrong player:
2 on IMG_0109 and 0 on IMG_0104.

Colour can't stop a switch at the moment it happens, because there is no readable call then. A fix has to be in the matching:
a joint check that compares the swapped pairing, Astra's "arm 3".

## 3. NMS does not help the fine-tuned detector

NMS 0.7 helped the old RF-DETR Large boxes, which had duplicate boxes on the same player: TI 0.749 → 0.758. With the
fine-tuned boxes there are almost no duplicates (0.03 false boxes per frame), so overlapping boxes are two real players.
×6 + NMS 0.7 vs ×6:

| Videos | Without NMS | With NMS 0.7 |
|---|---|---|
| Mean over 7 games | 0.847 | 0.848 |
| IMG_0108 | **0.886** | 0.873 |
| Cross-team rows, mean | 1,306 | 1,410 |

Dropped for the fine-tuned boxes.

## 4. Fixing the overlap-after-NMS bug changed nothing (here)

Overlap was measured after the court filter and NMS, so a box whose neighbour was removed counted as "clear" (`overlap_raw`).
The bug is real and the fix is kept. But with NMS off it made no difference: IMG_0109 was identical to before. It only matters
in runs with NMS.

## 5. Safeguard variants that lost (IMG_0109)

| Variant | TI | Cross-team rows |
|---|---|---|
| Kept: back/unclear no call at IoU ≥ 0.4, trust by overlap | **0.912** | **326** |
| Same at IoU ≥ 0.6 | 0.842 | 3,020 |
| All heavy-overlap calls dropped at IoU ≥ 0.6, front included | 0.833 | 3,371 |
| Back box no call at 0.4, old flat trust (93.9%) | 0.853 | 2,601 |
| Trust by overlap only, no back-box rule | 0.872 | 1,861 |

- **IoU 0.6 is too late.** Back-box calls are already anti-correlated with the truth from 0.5: 16–34% right against an
  other-team neighbour.
- **Dropping the front box's call wastes good evidence.** It is 95–97% right.
- **Each half alone is weak.** The back-box rule and trust by overlap only work together.

## 6. Team lock alone: helps, with a side effect

Lock 30 (a track sure of one team for 30 matched frames in a row never changes team) **without** the safeguard:
- TI: IMG_0109 0.848 → 0.903, IMG_0108 0.886 → 0.906.
- But same-team rows on IMG_0109 rose from 31 to 142: a locked track that loses its player can still land on a teammate.
- AssA and fragmentation are much worse than with the safeguard (IMG_0108 AssA 25.3 vs 38.4; 10.5 vs 5 tracks per player).

With the safeguard, the lock changes nothing on IMG_0104/0105/0108/0109. It is kept only as a safety net for the challenge
videos.

## 7. Ideas not pursued, with reasons

- **Output team = mode of the last 10 frames.** The remaining swaps last about 4 frames, so a 10-frame mode still shows the old
  team for nearly all of them. It also can't fix the jersey: the row keeps the main player's number. `identity.py` already
  splits tracks on persistent call flips (30-frame window) and votes one team per piece.
- **"Cross-team rows" are not colour mistakes.** They count frames where a track is on a different person from its main player
  (an identity switch). The colour calls in those frames are usually right for the person in the box.

## 8. Earlier, on the old RF-DETR Large boxes (see docs/15)

These all lost to plain ×6 (+ NMS 0.7):
- veto (a clear box can't join a track sure of the other team);
- birth block (no new track when a team already has 3 sure tracks);
- overlap q-table by neighbour call;
- slower belief learning on overlapped boxes (`team_learn_overlap` 0 and 0.25);
- a higher cap (20);
- the "loose" setting (cdt 20, ambiguity 0).

The q-table and slower learning were never retested on the fine-tuned boxes.

The real TI-HOTA (with guessed jerseys) went *down* as the team penalty went up (0.208 → 0.179), because jersey guessing
dominates the score. Team gains only show up once the jersey reader exists.

## Files

- Results: `outputs/reports/team_sweep_ft.jsonl`, `team_sweep_guard_0109.jsonl`, `team_lock_test.jsonl`,
  `team_sweep_lock.jsonl`, `team_sweep_detach.jsonl`, `team_sweep_detach_0104.jsonl`.
- Swap lists: `outputs/reports/IMG_0109__ft_guard_lock30_events.txt`.
- Annotated video: `outputs/overlays/IMG_0109__ft_guard_lock30.mp4`.
