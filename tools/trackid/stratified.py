# Copied from object_tracking_paper/src/eval/stratified.py; only the eval_mot import path changed.
"""Stratified mirror of `eval_mot.eval_hota` / `eval_clear` / `eval_identity`.

Why this exists
----------------
HOTA is not decomposable per GT box in general -- `DetPr`/`DetA`/`HOTA` all
contain false positives, which have no GT box and therefore no occlusion
stratum. But a substantial subset of its internals ARE exactly decomposable,
because their denominators are made only of GT boxes: `HOTA_TP`, `HOTA_FN`,
`DetRe`, `AssA`/`AssRe`/`AssPr` (each a TP-weighted mean, see
`eval_mot.eval_hota` line ~312), `LocA`, and the CLEAR/Identity TP/FN/IDSW
counts. This module recomputes those quantities per GT-box-frame stratum
(e.g. an occlusion bin) by re-deriving `eval_hota`/`eval_clear`/
`eval_identity`'s inner loops with the stratum of each GT box exposed, rather
than by approximating or re-deriving the metric from scratch.

The caveat this module's output must ALWAYS be printed with: the per-match
association score `A(c) = matches_count[g,t] / (gt_count[g] + tr_count[t] -
matches_count[g,t])` is a property of the GLOBAL (gt_id, tracker_id) pair,
accumulated over the WHOLE sequence -- not of any single frame. Binning
attributes that already-global score to the stratum of the GT box AT THAT
TIMESTEP. So `AssA_bin` answers "how well-associated are the matches that
happen while the object is in this stratum", not "how well-associated is
this stratum's portion of the trajectory in isolation".

`assert_recomposes` is the hard gate that makes this reimplementation
trustworthy: summing every per-stratum quantity back up must reproduce
`eval_hota`'s un-stratified output to floating-point precision, on real data,
not just on synthetic fixtures. If it does not, do not trust anything
downstream of the ledger.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackid.eval_mot import EPS, HOTA_ALPHA_VALUES, SequenceData

__all__ = [
    "STRAT_FIELDS",
    "stratified_hota",
    "stratified_clear",
    "stratified_identity",
    "assert_recomposes",
    "clear_ledger",
    "identity_map",
]

STRAT_FIELDS = ("HOTA_TP", "HOTA_FN", "AssA_sum", "AssRe_sum", "AssPr_sum", "LocA_sum")

_DERIVED_FIELDS = ("DetRe", "AssA", "AssRe", "AssPr", "LocA", "OWTA")


def _check_strata(data: SequenceData, strata: list[np.ndarray]) -> None:
    if len(strata) != len(data.gt_ids):
        raise ValueError(
            f"strata has {len(strata)} timesteps, data.gt_ids has {len(data.gt_ids)} -- "
            "these must be the same sequence's per-frame GT arrays"
        )
    for t, (s, g) in enumerate(zip(strata, data.gt_ids)):
        if len(s) != len(g):
            raise ValueError(
                f"strata[{t}] has length {len(s)} but data.gt_ids[{t}] has length {len(g)} -- "
                "gt_keep/class filtering has diverged between the two builders (sanity S5)"
            )


def _finalize_strat(out: dict[str, np.ndarray], n_strata: int, n_alpha: int) -> dict[str, np.ndarray]:
    tp = out["HOTA_TP"]
    fn = out["HOTA_FN"]
    out["DetRe"] = tp / np.maximum(1.0, tp + fn)
    out["AssA"] = out["AssA_sum"] / np.maximum(1.0, tp)
    out["AssRe"] = out["AssRe_sum"] / np.maximum(1.0, tp)
    out["AssPr"] = out["AssPr_sum"] / np.maximum(1.0, tp)
    out["LocA"] = out["LocA_sum"] / np.maximum(1.0, tp)
    out["OWTA"] = np.sqrt(np.maximum(0.0, out["DetRe"] * out["AssA"]))
    assert out["HOTA_TP"].shape == (n_strata, n_alpha)
    return out


def stratified_hota(
    data: SequenceData,
    strata: list[np.ndarray],
    n_strata: int,
    alpha_values: tuple[float, ...] = HOTA_ALPHA_VALUES,
) -> dict[str, np.ndarray]:
    """Per-stratum `HOTA_TP`/`HOTA_FN`/`AssA_sum`/`AssRe_sum`/`AssPr_sum`/`LocA_sum`,
    plus derived `DetRe`/`AssA`/`AssRe`/`AssPr`/`LocA`/`OWTA` -- every array
    shaped `[n_strata, len(alpha_values)]`. FP is deliberately absent: an FP
    has no GT box and therefore no stratum, so DetPr/DetA/HOTA are NOT
    emitted per stratum (see module docstring).
    """
    _check_strata(data, strata)
    n_alpha = len(alpha_values)
    out = {f: np.zeros((n_strata, n_alpha), dtype=np.float64) for f in STRAT_FIELDS}

    if data.num_tracker_dets == 0:
        for t, gt_ids_t in enumerate(data.gt_ids):
            if len(gt_ids_t) == 0:
                continue
            bins, counts = np.unique(strata[t], return_counts=True)
            out["HOTA_FN"][bins, :] += counts[:, None]
        return _finalize_strat(out, n_strata, n_alpha)
    if data.num_gt_dets == 0:
        return _finalize_strat(out, n_strata, n_alpha)

    # -- Global pass, identical to eval_hota: the association score is a
    #    property of the (gt_id, tracker_id) pair over the WHOLE sequence.
    potential_matches_count = np.zeros((data.num_gt_ids, data.num_tracker_ids), dtype=float)
    gt_id_count = np.zeros((data.num_gt_ids, 1), dtype=float)
    tracker_id_count = np.zeros((1, data.num_tracker_ids), dtype=float)

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        if similarity.size:
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            mask = sim_iou_denom > EPS
            sim_iou[mask] = similarity[mask] / sim_iou_denom[mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tracker_ids_t] += 1

    denom = gt_id_count + tracker_id_count - potential_matches_count
    global_alignment_score = np.divide(
        potential_matches_count, denom, out=np.zeros_like(potential_matches_count), where=denom > EPS
    )
    matches_counts = [np.zeros_like(potential_matches_count) for _ in alpha_values]
    # per-alpha list of (stratum, gt_global_id, tracker_global_id) for every TP occurrence.
    tp_occurrences: list[list[tuple[int, int, int]]] = [[] for _ in alpha_values]

    for t, (gt_ids_t, tracker_ids_t, similarity) in enumerate(
        zip(data.gt_ids, data.tracker_ids, data.similarity_scores)
    ):
        strata_t = strata[t]
        if len(gt_ids_t) == 0:
            continue  # FP-only frame: nothing with a GT box to attribute.
        if len(tracker_ids_t) == 0:
            bins, counts = np.unique(strata_t, return_counts=True)
            out["HOTA_FN"][bins, :] += counts[:, None]
            continue

        score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity
        match_rows, match_cols = linear_sum_assignment(-score_mat)

        for alpha_index, alpha in enumerate(alpha_values):
            matched = similarity[match_rows, match_cols] >= alpha - EPS
            alpha_match_rows = match_rows[matched]
            alpha_match_cols = match_cols[matched]
            matched_row_set = set(alpha_match_rows.tolist())

            for r, c in zip(alpha_match_rows, alpha_match_cols):
                s = int(strata_t[r])
                out["HOTA_TP"][s, alpha_index] += 1
                out["LocA_sum"][s, alpha_index] += float(similarity[r, c])
                g_glob = int(gt_ids_t[r])
                tr_glob = int(tracker_ids_t[c])
                matches_counts[alpha_index][g_glob, tr_glob] += 1
                tp_occurrences[alpha_index].append((s, g_glob, tr_glob))

            for r in range(len(gt_ids_t)):
                if r not in matched_row_set:
                    s = int(strata_t[r])
                    out["HOTA_FN"][s, alpha_index] += 1

    for alpha_index in range(n_alpha):
        matches_count = matches_counts[alpha_index]
        ass_a = matches_count / np.maximum(1.0, gt_id_count + tracker_id_count - matches_count)
        ass_re = matches_count / np.maximum(1.0, gt_id_count)
        ass_pr = matches_count / np.maximum(1.0, tracker_id_count)
        for s, g, tr in tp_occurrences[alpha_index]:
            out["AssA_sum"][s, alpha_index] += ass_a[g, tr]
            out["AssRe_sum"][s, alpha_index] += ass_re[g, tr]
            out["AssPr_sum"][s, alpha_index] += ass_pr[g, tr]

    return _finalize_strat(out, n_strata, n_alpha)


def stratified_clear(
    data: SequenceData, strata: list[np.ndarray], n_strata: int, threshold: float = 0.5
) -> dict[str, np.ndarray]:
    """Per-stratum `CLR_TP`, `CLR_FN`, `IDSW` -- `[n_strata]` int arrays.

    Reuses `eval_clear`'s matching rule verbatim (score = 1000*(id ==
    prev_timestep_id) + sim, zeroed below `threshold`) so summed IDSW
    reproduces `eval_clear`'s IDSW exactly (sanity S1).
    """
    _check_strata(data, strata)
    out = {
        "CLR_TP": np.zeros(n_strata, dtype=np.int64),
        "CLR_FN": np.zeros(n_strata, dtype=np.int64),
        "IDSW": np.zeros(n_strata, dtype=np.int64),
    }
    if data.num_tracker_dets == 0:
        for t, gt_ids_t in enumerate(data.gt_ids):
            if len(gt_ids_t) == 0:
                continue
            bins, counts = np.unique(strata[t], return_counts=True)
            out["CLR_FN"][bins] += counts
        return out
    if data.num_gt_dets == 0:
        return out

    prev_tracker_id = np.nan * np.zeros(data.num_gt_ids)
    prev_timestep_tracker_id = np.nan * np.zeros(data.num_gt_ids)

    for t, (gt_ids_t, tracker_ids_t, similarity) in enumerate(
        zip(data.gt_ids, data.tracker_ids, data.similarity_scores)
    ):
        strata_t = strata[t]
        if len(gt_ids_t) == 0:
            continue
        if len(tracker_ids_t) == 0:
            bins, counts = np.unique(strata_t, return_counts=True)
            out["CLR_FN"][bins] += counts
            continue

        score_mat = tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]]
        score_mat = 1000 * score_mat + similarity
        score_mat[similarity < threshold - EPS] = 0

        match_rows, match_cols = linear_sum_assignment(-score_mat)
        matched = score_mat[match_rows, match_cols] > EPS
        match_rows = match_rows[matched]
        match_cols = match_cols[matched]

        matched_gt_ids = gt_ids_t[match_rows]
        matched_tracker_ids = tracker_ids_t[match_cols]
        prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
        is_idsw = (~np.isnan(prev_matched_tracker_ids)) & (matched_tracker_ids != prev_matched_tracker_ids)

        matched_row_set = set(match_rows.tolist())
        for i, r in enumerate(match_rows):
            s = int(strata_t[r])
            out["CLR_TP"][s] += 1
            if is_idsw[i]:
                out["IDSW"][s] += 1
        for r in range(len(gt_ids_t)):
            if r not in matched_row_set:
                out["CLR_FN"][int(strata_t[r])] += 1

        prev_tracker_id[matched_gt_ids] = matched_tracker_ids
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

    return out


def identity_map(data: SequenceData, threshold: float = 0.5) -> np.ndarray:
    """The GLOBAL gtID -> trackerID map `eval_identity` solves internally
    (a sequence-level bipartite assignment), exposed standalone.

    `[num_gt_ids]` int64, RELABELED ids (see `build_sequence_data`); -1 where
    a GT id is mapped to its own "no partner" dummy column (never actually
    matched to any tracker id under the optimal identity assignment).
    """
    num_gt_ids = data.num_gt_ids
    num_tracker_ids = data.num_tracker_ids
    if data.num_tracker_dets == 0 or data.num_gt_dets == 0:
        return -np.ones(num_gt_ids, dtype=np.int64)

    potential_matches_count = np.zeros((num_gt_ids, num_tracker_ids))
    gt_id_count = np.zeros(num_gt_ids)
    tracker_id_count = np.zeros(num_tracker_ids)

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        matches_mask = np.greater_equal(similarity, threshold)
        match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
        potential_matches_count[gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]] += 1
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[tracker_ids_t] += 1

    fp_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
    fn_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
    fp_mat[num_gt_ids:, :num_tracker_ids] = 1e10
    fn_mat[:num_gt_ids, num_tracker_ids:] = 1e10
    for gt_id in range(num_gt_ids):
        fn_mat[gt_id, :num_tracker_ids] = gt_id_count[gt_id]
        fn_mat[gt_id, num_tracker_ids + gt_id] = gt_id_count[gt_id]
    for tracker_id in range(num_tracker_ids):
        fp_mat[:num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
        fp_mat[tracker_id + num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
    fn_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count
    fp_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count

    match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)
    gt_to_tracker = -np.ones(num_gt_ids, dtype=np.int64)
    for r, c in zip(match_rows, match_cols):
        if r < num_gt_ids and c < num_tracker_ids:
            gt_to_tracker[r] = c
    return gt_to_tracker


def stratified_identity(
    data: SequenceData, strata: list[np.ndarray], n_strata: int, threshold: float = 0.5
) -> dict[str, np.ndarray]:
    """Per-stratum `IDTP`, `IDFN`, `IDR`, given the GLOBAL gtID->trID map
    solved exactly as `eval_identity` solves it (a sequence-level bipartite
    assignment, not something that can be solved per-stratum). `IDFP`/`IDP`/
    `IDF1` are not emitted -- they involve tracker ids with no GT box.
    """
    _check_strata(data, strata)
    out = {
        "IDTP": np.zeros(n_strata, dtype=np.int64),
        "IDFN": np.zeros(n_strata, dtype=np.int64),
    }
    if data.num_tracker_dets == 0:
        for t, gt_ids_t in enumerate(data.gt_ids):
            if len(gt_ids_t) == 0:
                continue
            bins, counts = np.unique(strata[t], return_counts=True)
            out["IDFN"][bins] += counts
        out["IDR"] = out["IDTP"] / np.maximum(1, out["IDTP"] + out["IDFN"])
        return out
    if data.num_gt_dets == 0:
        out["IDR"] = np.zeros(n_strata)
        return out

    gt_to_tracker = identity_map(data, threshold)

    # Now walk the sequence again: a GT box is an IDTP iff, at that frame, its
    # globally-mapped tracker id is present AND similarity to it >= threshold.
    for t, (gt_ids_t, tracker_ids_t, similarity) in enumerate(
        zip(data.gt_ids, data.tracker_ids, data.similarity_scores)
    ):
        strata_t = strata[t]
        if len(gt_ids_t) == 0:
            continue
        tracker_pos = {int(tid): j for j, tid in enumerate(tracker_ids_t)}
        for r, g in enumerate(gt_ids_t):
            s = int(strata_t[r])
            mapped = int(gt_to_tracker[g])
            is_tp = False
            if mapped >= 0 and mapped in tracker_pos:
                j = tracker_pos[mapped]
                if similarity[r, j] >= threshold:
                    is_tp = True
            if is_tp:
                out["IDTP"][s] += 1
            else:
                out["IDFN"][s] += 1

    out["IDR"] = out["IDTP"] / np.maximum(1, out["IDTP"] + out["IDFN"])
    return out


def assert_recomposes(strat: dict[str, np.ndarray], ref: dict[str, Any], atol: float = 1e-9) -> None:
    """HARD GATE: summed per-stratum quantities must reproduce `eval_hota`'s
    (or `combine_bundles`'s) un-stratified output, elementwise per alpha.

    Checks `sum_s TP_s == ref['HOTA_TP']`, same for `HOTA_FN`, and
    `sum_s X_sum_s / sum_s TP_s == ref[X]` within `atol` for
    `X in {AssA, AssRe, AssPr, LocA}`. Raises `AssertionError` with a
    per-alpha diff table on failure -- this is the check that makes the
    reimplementation trustworthy (same pattern as the M4b vendored-postprocess
    hard gate).
    """
    tp_sum = strat["HOTA_TP"].sum(axis=0)
    fn_sum = strat["HOTA_FN"].sum(axis=0)
    ref_tp = np.asarray(ref["HOTA_TP"], dtype=np.float64)
    ref_fn = np.asarray(ref["HOTA_FN"], dtype=np.float64)

    problems: list[str] = []
    if not np.allclose(tp_sum, ref_tp, atol=1e-6):
        problems.append(f"HOTA_TP: sum_s={tp_sum.tolist()} vs ref={ref_tp.tolist()}")
    if not np.allclose(fn_sum, ref_fn, atol=1e-6):
        problems.append(f"HOTA_FN: sum_s={fn_sum.tolist()} vs ref={ref_fn.tolist()}")

    for field, sum_field in (("AssA", "AssA_sum"), ("AssRe", "AssRe_sum"), ("AssPr", "AssPr_sum")):
        recomposed = strat[sum_field].sum(axis=0) / np.maximum(1.0, tp_sum)
        ref_val = np.asarray(ref[field], dtype=np.float64)
        if not np.allclose(recomposed, ref_val, atol=atol * 100):
            diffs = np.abs(recomposed - ref_val)
            problems.append(
                f"{field}: recomposed={recomposed.tolist()} vs ref={ref_val.tolist()} "
                f"max_abs_diff={diffs.max():.3e}"
            )

    loca_recomposed = strat["LocA_sum"].sum(axis=0) / np.maximum(1.0, tp_sum)
    ref_loca = np.asarray(ref["LocA"], dtype=np.float64)
    if not np.allclose(loca_recomposed, ref_loca, atol=atol * 100):
        diffs = np.abs(loca_recomposed - ref_loca)
        problems.append(
            f"LocA: recomposed={loca_recomposed.tolist()} vs ref={ref_loca.tolist()} "
            f"max_abs_diff={diffs.max():.3e}"
        )

    if problems:
        raise AssertionError("stratified reimplementation does not recompose:\n" + "\n".join(problems))


def clear_ledger(data: SequenceData, threshold: float = 0.5) -> dict[str, list[np.ndarray]]:
    """`eval_clear`'s inner loop with its state exposed, per timestep.

    Returns, per timestep (each a `list[np.ndarray]` of length
    `num_timesteps`, one entry per frame, aligned with `data.gt_ids[t]`):
      - `assoc_tr_id`: relabeled tracker id matched to this GT box this
        frame at >= `threshold` by `eval_clear`'s rule; -1 if unmatched.
      - `prev_tr_id`: the last non--1 `assoc_tr_id` for this GT id BEFORE
        this frame; -1 if never matched before.
      - `is_idsw`: bool, True iff this frame's match is an identity switch
        by `eval_clear`'s exact rule.
      - `matched_iou`: the raw similarity (IoU) of the matched pair, 0.0 if
        unmatched.
      - `is_run_start`: bool, True iff this row is a matched row where the
        GT id was NOT "currently tracked" as of the immediately preceding
        frame (`eval_clear`'s `not_previously_tracked`) -- i.e. this match
        begins a new maximal tracked run. Because `eval_clear` resets its
        `prev_timestep_tracker_id` bookkeeping to NaN every "normal" frame
        (both GT and tracker nonempty) and only re-populates it for GT ids
        matched THAT frame, a GT id simply being ABSENT from a frame (no
        row at all, not even an unmatched one) also breaks its run -- this
        flag is computed from that exact same state machine, not
        re-derived from the compacted per-row ledger afterwards, so it
        replicates that gap-breaks-runs behaviour for free. Grouping by GT
        id, `sum(is_run_start) - n_ids_with_>=1_run_start` reproduces
        `eval_clear`'s `Frag` exactly (sanity S2).

    This is the backbone `build_ledger.py` uses to build the row-aligned
    ledger; summing `is_idsw` over all frames/ids reproduces `eval_clear`'s
    `IDSW` exactly by construction (same score matrix, same threshold, same
    Hungarian call).
    """
    n_t = len(data.gt_ids)
    assoc_tr_id_by_t: list[np.ndarray] = [np.full(len(g), -1, dtype=np.int64) for g in data.gt_ids]
    prev_tr_id_by_t: list[np.ndarray] = [np.full(len(g), -1, dtype=np.int64) for g in data.gt_ids]
    is_idsw_by_t: list[np.ndarray] = [np.zeros(len(g), dtype=bool) for g in data.gt_ids]
    matched_iou_by_t: list[np.ndarray] = [np.zeros(len(g), dtype=np.float32) for g in data.gt_ids]
    is_run_start_by_t: list[np.ndarray] = [np.zeros(len(g), dtype=bool) for g in data.gt_ids]

    if data.num_gt_dets == 0:
        return {
            "assoc_tr_id": assoc_tr_id_by_t,
            "prev_tr_id": prev_tr_id_by_t,
            "is_idsw": is_idsw_by_t,
            "matched_iou": matched_iou_by_t,
            "is_run_start": is_run_start_by_t,
        }

    prev_tracker_id = np.nan * np.zeros(data.num_gt_ids)  # last matched tracker id ever, by GLOBAL gt id
    prev_timestep_tracker_id = np.nan * np.zeros(data.num_gt_ids)  # matched LAST timestep only

    if data.num_tracker_dets == 0:
        return {
            "assoc_tr_id": assoc_tr_id_by_t,
            "prev_tr_id": prev_tr_id_by_t,
            "is_idsw": is_idsw_by_t,
            "matched_iou": matched_iou_by_t,
            "is_run_start": is_run_start_by_t,
        }

    for t in range(n_t):
        gt_ids_t = data.gt_ids[t]
        tracker_ids_t = data.tracker_ids[t]
        similarity = data.similarity_scores[t]
        if len(gt_ids_t) == 0:
            continue

        # record "prev_tr_id" BEFORE this frame updates prev_tracker_id.
        prev_tr_id_by_t[t] = np.where(
            np.isnan(prev_tracker_id[gt_ids_t]), -1, prev_tracker_id[gt_ids_t]
        ).astype(np.int64)

        if len(tracker_ids_t) == 0:
            # NOTE: eval_clear's own `len(tracker_ids_t) == 0` branch does
            # NOT reset `prev_timestep_tracker_id` here (it only resets it
            # inside the "normal" branch below) -- mirrored verbatim, even
            # though it looks like it should reset, so is_idsw stays exactly
            # bit-identical to eval_clear on every frame shape.
            continue

        # Snapshot BEFORE this frame's reset -- "not previously tracked" per
        # eval_clear's gt_frag_count trigger.
        not_previously_tracked_row = np.isnan(prev_timestep_tracker_id[gt_ids_t])

        score_mat = tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]]
        score_mat = 1000 * score_mat + similarity
        score_mat[similarity < threshold - EPS] = 0

        match_rows, match_cols = linear_sum_assignment(-score_mat)
        matched = score_mat[match_rows, match_cols] > EPS
        match_rows = match_rows[matched]
        match_cols = match_cols[matched]

        matched_gt_ids = gt_ids_t[match_rows]
        matched_tracker_ids = tracker_ids_t[match_cols]
        prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
        is_idsw = (~np.isnan(prev_matched_tracker_ids)) & (matched_tracker_ids != prev_matched_tracker_ids)

        for i, r in enumerate(match_rows):
            assoc_tr_id_by_t[t][r] = matched_tracker_ids[i]
            is_idsw_by_t[t][r] = bool(is_idsw[i])
            matched_iou_by_t[t][r] = similarity[r, match_cols[i]]
            is_run_start_by_t[t][r] = bool(not_previously_tracked_row[r])

        prev_tracker_id[matched_gt_ids] = matched_tracker_ids
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

    return {
        "assoc_tr_id": assoc_tr_id_by_t,
        "prev_tr_id": prev_tr_id_by_t,
        "is_idsw": is_idsw_by_t,
        "matched_iou": matched_iou_by_t,
        "is_run_start": is_run_start_by_t,
    }
