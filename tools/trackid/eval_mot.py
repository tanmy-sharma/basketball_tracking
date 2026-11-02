# Copied from object_tracking_paper/src/eval/eval_mot.py (standalone: numpy and scipy only).
#!/usr/bin/env python3
"""Standalone MOT-format tracking evaluator.

Computes HOTA / CLEAR (MOTA, MOTP, ...) / Identity (IDF1, ...) metrics for a
folder of MOT-format prediction .txt files against a folder of ground-truth
.txt files. The metric definitions (HOTA, CLEAR, Identity) and the class /
distractor handling mirror what BoxMOT reports in its README benchmark
tables (HOTA, MOTA, IDF1) and computes internally in
boxmot/engine/eval/motmetrics.py.

This script has NO dependency on the boxmot package or repository - it only
needs numpy and scipy, and can be copied anywhere and run standalone.

Expected MOT txt format (comma-delimited), one detection per row:
    frame, id, x, y, w, h, conf, class, <...extra columns ignored>
`conf` in ground-truth files is treated as MOTChallenge's "consider this
box" flag (0 => ignore that row). `class` (if present) is used for class
filtering / distractor handling. Rows with fewer columns are padded with
conf=1 and class=1.

This script is strictly read-only with respect to --pred-dir and --gt-dir:
it never writes into either. Output reports are written only to --out-dir.

Usage:
    python eval_mot.py /path/to/predictions_folder \
        --gt-dir eval_folder --out-dir eval_results

    # generic dataset, no class/distractor filtering:
    python eval_mot.py /path/to/predictions_folder \
        --gt-dir /path/to/gt_folder --eval-class-id "" --distractor-class-ids

    # MOT17/MOT20: keep the GT-side defaults (class 1, distractors 2 7 8 12)
    # but disable filtering on the prediction side, because trackers here write
    # class -1 in column 8 and would otherwise be discarded wholesale:
    python eval_mot.py /path/to/predictions_folder \
        --gt-dir /path/to/gt_folder --pred-class-id ""
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

HOTA_ALPHA_VALUES: tuple[float, ...] = tuple(float(v) for v in np.arange(0.05, 0.99, 0.05))

_HOTA_ARRAY_FIELDS = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA")
_HOTA_COUNT_ARRAY_FIELDS = ("HOTA_TP", "HOTA_FN", "HOTA_FP")
_HOTA_FLOAT_FIELDS = ("HOTA(0)", "LocA(0)", "HOTALocA(0)")
_CLEAR_INTEGER_FIELDS = ("CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag", "CLR_Frames")
_CLEAR_FLOAT_FIELDS = ("MOTA", "MOTP", "MODA", "CLR_Re", "CLR_Pr", "MTR", "PTR", "MLR", "sMOTA")
_CLEAR_EXTRA_FLOAT_FIELDS = ("CLR_F1", "FP_per_frame", "MOTAL", "MOTP_sum")
_CLEAR_SUMMED_FIELDS = (*_CLEAR_INTEGER_FIELDS, "MOTP_sum")
_IDENTITY_INTEGER_FIELDS = ("IDTP", "IDFN", "IDFP")
_IDENTITY_FLOAT_FIELDS = ("IDF1", "IDR", "IDP")
_COUNT_INTEGER_FIELDS = ("Dets", "GT_Dets", "IDs", "GT_IDs", "Frames")

# Headline metrics BoxMOT's README reports per benchmark table.
HEADLINE_FIELDS = ("HOTA", "MOTA", "IDF1")

EPS = np.finfo(float).eps


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SequenceData:
    seq: str
    gt_ids: list[np.ndarray]
    tracker_ids: list[np.ndarray]
    similarity_scores: list[np.ndarray]
    num_timesteps: int
    num_gt_dets: int
    num_tracker_dets: int
    num_gt_ids: int
    num_tracker_ids: int


def _read_mot_txt(path: Path) -> np.ndarray:
    """Read a MOT-format txt file into an (N, 9) float array.

    Columns: frame, id, x, y, w, h, conf, class, extra. Missing conf/class
    columns are padded with 1.0 (i.e. "consider" / class 1).
    """
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 9), dtype=float)
    try:
        data = np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError:
        data = np.loadtxt(path, ndmin=2)
    if data.size == 0:
        return np.empty((0, 9), dtype=float)
    n_cols = data.shape[1]
    if n_cols < 9:
        pad = np.ones((data.shape[0], 9 - n_cols), dtype=float)
        if n_cols < 7:
            pad[:, : max(0, 7 - n_cols)] = 1.0
        data = np.hstack([data, pad])
    return data.astype(float, copy=False)


def _rows_for_frame(data: np.ndarray, frame_id: int) -> np.ndarray:
    if data.size == 0:
        return np.empty((0, data.shape[1] if data.ndim == 2 else 9), dtype=float)
    return data[data[:, 0].astype(int) == frame_id]


def _frame_count(*arrays: np.ndarray) -> int:
    max_frame = 0
    for data in arrays:
        if data.size and data.shape[1] > 0:
            max_frame = max(max_frame, int(np.max(data[:, 0])))
    return max_frame


def _relabel_ids(frame_ids: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    unique_ids: list[int] = []
    for ids in frame_ids:
        unique_ids.extend(int(v) for v in np.unique(ids.astype(int)))
    if not unique_ids:
        return [ids.astype(int, copy=False) for ids in frame_ids], 0
    id_map = {raw_id: index for index, raw_id in enumerate(sorted(set(unique_ids)))}
    relabeled = [
        np.asarray([id_map[int(v)] for v in ids], dtype=int) if len(ids) else np.empty(0, dtype=int)
        for ids in frame_ids
    ]
    return relabeled, len(id_map)


def _iou_matrix(gt_boxes: np.ndarray, tracker_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=np.float32)
    gt_x1y1 = gt_boxes[:, :2]
    gt_x2y2 = gt_boxes[:, :2] + np.maximum(gt_boxes[:, 2:4], 0.0)
    tr_x1y1 = tracker_boxes[:, :2]
    tr_x2y2 = tracker_boxes[:, :2] + np.maximum(tracker_boxes[:, 2:4], 0.0)
    inter_x1y1 = np.maximum(gt_x1y1[:, None, :], tr_x1y1[None, :, :])
    inter_x2y2 = np.minimum(gt_x2y2[:, None, :], tr_x2y2[None, :, :])
    inter_wh = np.maximum(0.0, inter_x2y2 - inter_x1y1)
    intersection = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    gt_area = np.maximum(gt_boxes[:, 2], 0.0) * np.maximum(gt_boxes[:, 3], 0.0)
    tr_area = np.maximum(tracker_boxes[:, 2], 0.0) * np.maximum(tracker_boxes[:, 3], 0.0)
    union = gt_area[:, None] + tr_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


# Sentinel for "filter predictions by whatever the GT is filtered by", which is
# the historical behaviour and stays the default so existing call sites are
# bit-identical. MOT17/MOT20 need the two decoupled: their GT genuinely carries
# class 1 plus distractor classes, while our predictions write class -1.
_SAME_AS_GT = object()


def build_sequence_data(
    *,
    seq_name: str,
    gt: np.ndarray,
    tracker: np.ndarray,
    eval_class_id: int | None,
    distractor_ids: set[int],
    pred_class_id: int | None | object = _SAME_AS_GT,
) -> SequenceData:
    if pred_class_id is _SAME_AS_GT:
        pred_class_id = eval_class_id
    num_timesteps = _frame_count(gt, tracker)

    gt_ids_by_frame: list[np.ndarray] = []
    tracker_ids_by_frame: list[np.ndarray] = []
    similarity_scores: list[np.ndarray] = []
    num_gt_dets = 0
    num_tracker_dets = 0

    for frame_id in range(1, num_timesteps + 1):
        gt_frame = _rows_for_frame(gt, frame_id)
        tracker_frame = _rows_for_frame(tracker, frame_id)

        gt_ids = gt_frame[:, 1].astype(int) if gt_frame.size else np.empty(0, dtype=int)
        gt_boxes = gt_frame[:, 2:6] if gt_frame.size else np.empty((0, 4), dtype=np.float32)
        gt_consider = gt_frame[:, 6] if gt_frame.size else np.ones(len(gt_ids))
        gt_classes = gt_frame[:, 7].astype(int) if gt_frame.size else np.ones(len(gt_ids), int)

        tracker_ids = tracker_frame[:, 1].astype(int) if tracker_frame.size else np.empty(0, dtype=int)
        tracker_boxes = tracker_frame[:, 2:6] if tracker_frame.size else np.empty((0, 4), dtype=np.float32)
        tracker_classes = (
            tracker_frame[:, 7].astype(int) if tracker_frame.size else np.ones(len(tracker_ids), int)
        )

        tracker_keep = (
            np.ones(len(tracker_ids), dtype=bool) if pred_class_id is None else tracker_classes == pred_class_id
        )
        kept_tracker_ids = tracker_ids[tracker_keep]
        # Official TrackEval raises on this; a repeated id in one frame would silently inflate the
        # identity-based metrics (IDF1, AssA), so refuse it rather than score it.
        if len(np.unique(kept_tracker_ids)) != len(kept_tracker_ids):
            ids, counts = np.unique(kept_tracker_ids, return_counts=True)
            raise ValueError(
                f"{seq_name}: tracker predicts the same ID more than once in a single timestep "
                f"(frame {frame_id}, id {int(ids[counts > 1][0])})"
            )
        kept_tracker_boxes = tracker_boxes[tracker_keep] if tracker_boxes.size else tracker_boxes
        similarity = _iou_matrix(gt_boxes, kept_tracker_boxes)

        # Remove tracker predictions matched to distractor GT classes (e.g. static
        # person, reflection) so they aren't counted as false positives.
        if distractor_ids and len(gt_ids) and len(kept_tracker_ids) and similarity.size:
            matching_scores = similarity.copy()
            matching_scores[matching_scores < 0.5 - EPS] = 0
            match_rows, match_cols = linear_sum_assignment(-matching_scores)
            actually_matched = matching_scores[match_rows, match_cols] > EPS
            match_rows = match_rows[actually_matched]
            match_cols = match_cols[actually_matched]
            remove_cols = match_cols[np.isin(gt_classes[match_rows], list(distractor_ids))]
            if remove_cols.size:
                kept_tracker_ids = np.delete(kept_tracker_ids, remove_cols, axis=0)
                similarity = np.delete(similarity, remove_cols, axis=1)

        class_keep = np.ones(len(gt_ids), dtype=bool) if eval_class_id is None else gt_classes == eval_class_id
        gt_keep = (gt_consider != 0) & class_keep
        kept_gt_ids = gt_ids[gt_keep]
        similarity = (
            similarity[gt_keep, :] if similarity.size else np.empty((len(kept_gt_ids), len(kept_tracker_ids)))
        )

        gt_ids_by_frame.append(kept_gt_ids)
        tracker_ids_by_frame.append(kept_tracker_ids)
        similarity_scores.append(similarity)
        num_gt_dets += len(kept_gt_ids)
        num_tracker_dets += len(kept_tracker_ids)

    gt_ids_by_frame, num_gt_ids = _relabel_ids(gt_ids_by_frame)
    tracker_ids_by_frame, num_tracker_ids = _relabel_ids(tracker_ids_by_frame)

    return SequenceData(
        seq=seq_name,
        gt_ids=gt_ids_by_frame,
        tracker_ids=tracker_ids_by_frame,
        similarity_scores=similarity_scores,
        num_timesteps=num_timesteps,
        num_gt_dets=num_gt_dets,
        num_tracker_dets=num_tracker_dets,
        num_gt_ids=num_gt_ids,
        num_tracker_ids=num_tracker_ids,
    )


# --------------------------------------------------------------------------- #
# Metrics: HOTA / CLEAR / Identity (mirrors boxmot/engine/eval/motmetrics.py)
# --------------------------------------------------------------------------- #


def _compute_final_hota_fields(res: dict[str, Any]) -> dict[str, Any]:
    res["DetRe"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"])
    res["DetPr"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FP"])
    res["DetA"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"])
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    res["OWTA"] = np.sqrt(res["DetRe"] * res["AssA"])
    res["HOTA(0)"] = float(res["HOTA"][0])
    res["LocA(0)"] = float(res["LocA"][0])
    res["HOTALocA(0)"] = res["HOTA(0)"] * res["LocA(0)"]
    return res


def eval_hota(data: SequenceData, alpha_values: tuple[float, ...] = HOTA_ALPHA_VALUES) -> dict[str, Any]:
    res: dict[str, Any] = {
        field: np.zeros(len(alpha_values), dtype=float) for field in (*_HOTA_ARRAY_FIELDS, *_HOTA_COUNT_ARRAY_FIELDS)
    }
    for field in _HOTA_FLOAT_FIELDS:
        res[field] = 0.0

    if data.num_tracker_dets == 0:
        res["HOTA_FN"] = data.num_gt_dets * np.ones(len(alpha_values), dtype=float)
        res["LocA"] = np.ones(len(alpha_values), dtype=float)
        return _compute_final_hota_fields(res)
    if data.num_gt_dets == 0:
        res["HOTA_FP"] = data.num_tracker_dets * np.ones(len(alpha_values), dtype=float)
        res["LocA"] = np.ones(len(alpha_values), dtype=float)
        return _compute_final_hota_fields(res)

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

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        if len(gt_ids_t) == 0:
            for i in range(len(alpha_values)):
                res["HOTA_FP"][i] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            for i in range(len(alpha_values)):
                res["HOTA_FN"][i] += len(gt_ids_t)
            continue

        score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity
        match_rows, match_cols = linear_sum_assignment(-score_mat)

        for alpha_index, alpha in enumerate(alpha_values):
            matched = similarity[match_rows, match_cols] >= alpha - EPS
            alpha_match_rows = match_rows[matched]
            alpha_match_cols = match_cols[matched]
            num_matches = len(alpha_match_rows)
            res["HOTA_TP"][alpha_index] += num_matches
            res["HOTA_FN"][alpha_index] += len(gt_ids_t) - num_matches
            res["HOTA_FP"][alpha_index] += len(tracker_ids_t) - num_matches
            if num_matches:
                res["LocA"][alpha_index] += float(np.sum(similarity[alpha_match_rows, alpha_match_cols]))
                matches_counts[alpha_index][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1

    for alpha_index in range(len(alpha_values)):
        matches_count = matches_counts[alpha_index]
        ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
        res["AssA"][alpha_index] = np.sum(matches_count * ass_a) / np.maximum(1, res["HOTA_TP"][alpha_index])
        ass_re = matches_count / np.maximum(1, gt_id_count)
        res["AssRe"][alpha_index] = np.sum(matches_count * ass_re) / np.maximum(1, res["HOTA_TP"][alpha_index])
        ass_pr = matches_count / np.maximum(1, tracker_id_count)
        res["AssPr"][alpha_index] = np.sum(matches_count * ass_pr) / np.maximum(1, res["HOTA_TP"][alpha_index])

    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
    return _compute_final_hota_fields(res)


def _compute_final_clear_fields(res: dict[str, Any]) -> dict[str, Any]:
    num_gt_ids = res["MT"] + res["ML"] + res["PT"]
    res["MTR"] = res["MT"] / np.maximum(1.0, num_gt_ids)
    res["MLR"] = res["ML"] / np.maximum(1.0, num_gt_ids)
    res["PTR"] = res["PT"] / np.maximum(1.0, num_gt_ids)
    res["CLR_Re"] = res["CLR_TP"] / np.maximum(1.0, res["CLR_TP"] + res["CLR_FN"])
    res["CLR_Pr"] = res["CLR_TP"] / np.maximum(1.0, res["CLR_TP"] + res["CLR_FP"])
    res["MODA"] = (res["CLR_TP"] - res["CLR_FP"]) / np.maximum(1.0, res["CLR_TP"] + res["CLR_FN"])
    res["MOTA"] = (res["CLR_TP"] - res["CLR_FP"] - res["IDSW"]) / np.maximum(1.0, res["CLR_TP"] + res["CLR_FN"])
    res["MOTP"] = res["MOTP_sum"] / np.maximum(1.0, res["CLR_TP"])
    res["sMOTA"] = (res["MOTP_sum"] - res["CLR_FP"] - res["IDSW"]) / np.maximum(1.0, res["CLR_TP"] + res["CLR_FN"])
    res["CLR_F1"] = res["CLR_TP"] / np.maximum(1.0, res["CLR_TP"] + 0.5 * res["CLR_FN"] + 0.5 * res["CLR_FP"])
    res["FP_per_frame"] = res["CLR_FP"] / np.maximum(1.0, res["CLR_Frames"])
    safe_log_idsw = np.log10(res["IDSW"]) if res["IDSW"] > 0 else res["IDSW"]
    res["MOTAL"] = (res["CLR_TP"] - res["CLR_FP"] - safe_log_idsw) / np.maximum(1.0, res["CLR_TP"] + res["CLR_FN"])
    return res


def eval_clear(data: SequenceData, threshold: float = 0.5) -> dict[str, Any]:
    res: dict[str, Any] = {field: 0 for field in (*_CLEAR_INTEGER_FIELDS, *_CLEAR_FLOAT_FIELDS)}
    for field in _CLEAR_EXTRA_FLOAT_FIELDS:
        res[field] = 0.0

    if data.num_tracker_dets == 0:
        res["CLR_FN"] = data.num_gt_dets
        res["ML"] = data.num_gt_ids
        res["MLR"] = 1.0
        return res
    if data.num_gt_dets == 0:
        res["CLR_FP"] = data.num_tracker_dets
        res["MLR"] = 1.0
        return res

    gt_id_count = np.zeros(data.num_gt_ids)
    gt_matched_count = np.zeros(data.num_gt_ids)
    gt_frag_count = np.zeros(data.num_gt_ids)
    prev_tracker_id = np.nan * np.zeros(data.num_gt_ids)
    prev_timestep_tracker_id = np.nan * np.zeros(data.num_gt_ids)

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        if len(gt_ids_t) == 0:
            res["CLR_FP"] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            res["CLR_FN"] += len(gt_ids_t)
            gt_id_count[gt_ids_t] += 1
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
        res["IDSW"] += int(np.sum(is_idsw))

        gt_id_count[gt_ids_t] += 1
        gt_matched_count[matched_gt_ids] += 1
        not_previously_tracked = np.isnan(prev_timestep_tracker_id)
        prev_tracker_id[matched_gt_ids] = matched_tracker_ids
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
        currently_tracked = ~np.isnan(prev_timestep_tracker_id)
        gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

        num_matches = len(matched_gt_ids)
        res["CLR_TP"] += num_matches
        res["CLR_FN"] += len(gt_ids_t) - num_matches
        res["CLR_FP"] += len(tracker_ids_t) - num_matches
        if num_matches:
            res["MOTP_sum"] += float(np.sum(similarity[match_rows, match_cols]))

    tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
    res["MT"] = int(np.sum(np.greater(tracked_ratio, 0.8)))
    res["PT"] = int(np.sum(np.greater_equal(tracked_ratio, 0.2))) - res["MT"]
    res["ML"] = data.num_gt_ids - res["MT"] - res["PT"]
    res["Frag"] = int(np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1)))
    res["CLR_Frames"] = data.num_timesteps
    return _compute_final_clear_fields(res)


def _compute_final_identity_fields(res: dict[str, Any]) -> dict[str, Any]:
    res["IDR"] = res["IDTP"] / np.maximum(1.0, res["IDTP"] + res["IDFN"])
    res["IDP"] = res["IDTP"] / np.maximum(1.0, res["IDTP"] + res["IDFP"])
    res["IDF1"] = res["IDTP"] / np.maximum(1.0, res["IDTP"] + 0.5 * res["IDFP"] + 0.5 * res["IDFN"])
    return res


def eval_identity(data: SequenceData, threshold: float = 0.5) -> dict[str, Any]:
    res: dict[str, Any] = {field: 0 for field in (*_IDENTITY_INTEGER_FIELDS, *_IDENTITY_FLOAT_FIELDS)}
    if data.num_tracker_dets == 0:
        res["IDFN"] = data.num_gt_dets
        return res
    if data.num_gt_dets == 0:
        res["IDFP"] = data.num_tracker_dets
        return res

    potential_matches_count = np.zeros((data.num_gt_ids, data.num_tracker_ids))
    gt_id_count = np.zeros(data.num_gt_ids)
    tracker_id_count = np.zeros(data.num_tracker_ids)

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        matches_mask = np.greater_equal(similarity, threshold)
        match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
        potential_matches_count[gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]] += 1
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[tracker_ids_t] += 1

    num_gt_ids = data.num_gt_ids
    num_tracker_ids = data.num_tracker_ids
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
    res["IDFN"] = int(fn_mat[match_rows, match_cols].sum())
    res["IDFP"] = int(fp_mat[match_rows, match_cols].sum())
    res["IDTP"] = int(gt_id_count.sum() - res["IDFN"])
    return _compute_final_identity_fields(res)


def eval_count(data: SequenceData) -> dict[str, Any]:
    return {
        "Dets": data.num_tracker_dets,
        "GT_Dets": data.num_gt_dets,
        "IDs": data.num_tracker_ids,
        "GT_IDs": data.num_gt_ids,
        "Frames": data.num_timesteps,
    }


MetricBundle = dict[str, dict[str, Any]]


def eval_bundle(data: SequenceData, iou_threshold: float) -> MetricBundle:
    return {
        "HOTA": eval_hota(data),
        "CLEAR": eval_clear(data, threshold=iou_threshold),
        "Identity": eval_identity(data, threshold=iou_threshold),
        "Count": eval_count(data),
    }


def _combine_sum(all_res: dict[str, dict[str, Any]], field: str) -> Any:
    values = [value[field] for value in all_res.values()]
    if values and isinstance(values[0], np.ndarray):
        return np.sum(values, axis=0)
    return sum(values)


def _combine_weighted_average(
    all_res: dict[str, dict[str, Any]], field: str, combined: dict[str, Any], *, weight_field: str
) -> np.ndarray:
    weighted_sum = sum(value[field] * value[weight_field] for value in all_res.values())
    return np.divide(
        weighted_sum,
        np.maximum(1e-10, combined[weight_field]),
        out=np.zeros_like(weighted_sum, dtype=float),
        where=np.maximum(1e-10, combined[weight_field]) > 0,
    )


def combine_bundles(all_bundles: dict[str, MetricBundle]) -> MetricBundle:
    hota_all = {key: value["HOTA"] for key, value in all_bundles.items()}
    hota: dict[str, Any] = {field: _combine_sum(hota_all, field) for field in _HOTA_COUNT_ARRAY_FIELDS}
    for field in ("AssRe", "AssPr", "AssA"):
        hota[field] = _combine_weighted_average(hota_all, field, hota, weight_field="HOTA_TP")
    loca_weighted_sum = sum(value["LocA"] * value["HOTA_TP"] for value in hota_all.values())
    hota["LocA"] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, hota["HOTA_TP"])
    hota = _compute_final_hota_fields(hota)

    clear_all = {key: value["CLEAR"] for key, value in all_bundles.items()}
    clear = {field: _combine_sum(clear_all, field) for field in _CLEAR_SUMMED_FIELDS}
    clear = _compute_final_clear_fields(clear)

    identity_all = {key: value["Identity"] for key, value in all_bundles.items()}
    identity = {field: _combine_sum(identity_all, field) for field in _IDENTITY_INTEGER_FIELDS}
    identity = _compute_final_identity_fields(identity)

    count_all = {key: value["Count"] for key, value in all_bundles.items()}
    count = {field: _combine_sum(count_all, field) for field in _COUNT_INTEGER_FIELDS}

    return {"HOTA": hota, "CLEAR": clear, "Identity": identity, "Count": count}


def _percent(value: Any) -> float:
    return max(0.0, float(value) * 100.0)


def _count(value: Any) -> int:
    return max(0, int(value))


def summarize_bundle(bundle: MetricBundle) -> dict[str, Any]:
    hota, clear, identity, count = bundle["HOTA"], bundle["CLEAR"], bundle["Identity"], bundle["Count"]
    summary: dict[str, Any] = {}
    for field in _HOTA_ARRAY_FIELDS:
        summary[field] = _percent(np.mean(hota[field]))
    for field in _HOTA_FLOAT_FIELDS:
        summary[field] = _percent(hota[field])
    for field in _HOTA_COUNT_ARRAY_FIELDS:
        summary[field] = _count(np.sum(hota[field]))
    for field in _CLEAR_FLOAT_FIELDS:
        summary[field] = _percent(clear[field])
    for field in _CLEAR_INTEGER_FIELDS:
        summary[field] = _count(clear[field])
    for field in _IDENTITY_FLOAT_FIELDS:
        summary[field] = _percent(identity[field])
    for field in _IDENTITY_INTEGER_FIELDS:
        summary[field] = _count(identity[field])
    for field in _COUNT_INTEGER_FIELDS:
        summary[field] = _count(count[field])
    return summary


# --------------------------------------------------------------------------- #
# Report formatting
# --------------------------------------------------------------------------- #


def format_report(title: str, summary: dict[str, Any]) -> str:
    lines = [title, "=" * len(title), ""]
    lines.append("Headline (as reported in the BoxMOT README benchmark tables):")
    for field in HEADLINE_FIELDS:
        lines.append(f"  {field:<10s} {summary[field]:6.2f}")
    lines.append("")

    lines.append("HOTA breakdown:")
    for field in (*_HOTA_ARRAY_FIELDS, *_HOTA_FLOAT_FIELDS):
        lines.append(f"  {field:<12s} {summary[field]:6.2f}")
    lines.append("")

    lines.append("CLEAR breakdown:")
    for field in _CLEAR_FLOAT_FIELDS:
        lines.append(f"  {field:<12s} {summary[field]:6.2f}")
    for field in ("MT", "PT", "ML", "Frag", "IDSW"):
        lines.append(f"  {field:<12s} {summary[field]:6d}")
    lines.append("")

    lines.append("Identity breakdown:")
    for field in _IDENTITY_FLOAT_FIELDS:
        lines.append(f"  {field:<12s} {summary[field]:6.2f}")
    lines.append("")

    lines.append("Counts:")
    for field in _COUNT_INTEGER_FIELDS:
        lines.append(f"  {field:<12s} {summary[field]:6d}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a folder of MOT-format prediction .txt files against ground-truth "
        ".txt files and report HOTA / MOTA / IDF1 (and full breakdown) per sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pred_dir", type=Path, help="Folder containing predicted MOT-format .txt files")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "eval_folder",
        help="Folder containing ground-truth MOT-format .txt files (matched by filename)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval_results"),
        help="Folder to write per-sequence and summary reports into (created if missing)",
    )
    parser.add_argument(
        "--eval-class-id",
        type=str,
        default="1",
        help="GT/pred class id (column 8) to evaluate. Pass an empty string to disable class filtering.",
    )
    parser.add_argument(
        "--pred-class-id",
        type=str,
        default=None,
        help=(
            "Class id (column 8) to keep on the PREDICTION side. Omit to reuse --eval-class-id "
            '(the historical behaviour). Pass an empty string to keep all predictions -- required '
            "on MOT17/MOT20, where the GT-side default of 1 is correct but trackers write -1."
        ),
    )
    parser.add_argument(
        "--distractor-class-ids",
        type=int,
        nargs="*",
        default=[2, 7, 8, 12],
        help="GT class ids treated as distractors (matched tracker boxes are not counted as FP)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used for CLEAR/Identity matching",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    pred_dir: Path = args.pred_dir
    gt_dir: Path = args.gt_dir
    out_dir: Path = args.out_dir

    if not pred_dir.is_dir():
        print(f"error: predictions folder not found: {pred_dir}", file=sys.stderr)
        return 1
    if not gt_dir.is_dir():
        print(f"error: ground-truth folder not found: {gt_dir}", file=sys.stderr)
        return 1

    eval_class_id = int(args.eval_class_id) if args.eval_class_id.strip() != "" else None
    distractor_ids = set(args.distractor_class_ids or [])
    # Three states: flag absent -> follow eval_class_id; "" -> keep every
    # prediction; a number -> filter predictions on that class.
    if args.pred_class_id is None:
        pred_class_id: int | None | object = _SAME_AS_GT
    elif args.pred_class_id.strip() == "":
        pred_class_id = None
    else:
        pred_class_id = int(args.pred_class_id)

    pred_files = sorted(p for p in pred_dir.glob("*.txt"))
    if not pred_files:
        print(f"error: no .txt files found in {pred_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    per_seq_bundles: dict[str, MetricBundle] = {}
    skipped: list[str] = []

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            skipped.append(pred_path.name)
            continue

        seq_name = pred_path.stem
        gt = _read_mot_txt(gt_path)
        tracker = _read_mot_txt(pred_path)
        data = build_sequence_data(
            seq_name=seq_name,
            gt=gt,
            tracker=tracker,
            eval_class_id=eval_class_id,
            distractor_ids=distractor_ids,
            pred_class_id=pred_class_id,
        )
        bundle = eval_bundle(data, iou_threshold=args.iou_threshold)
        per_seq_bundles[seq_name] = bundle

        summary = summarize_bundle(bundle)
        report = format_report(f"{seq_name}  (pred: {pred_path.name} vs gt: {gt_path.name})", summary)
        (out_dir / pred_path.name).write_text(report)
        print(
            f"{seq_name:<20s} HOTA={summary['HOTA']:6.2f}  MOTA={summary['MOTA']:6.2f}  IDF1={summary['IDF1']:6.2f}"
            f"  -> {out_dir / pred_path.name}"
        )

    for name in skipped:
        print(f"warning: no matching ground-truth file for '{name}' in {gt_dir}, skipped", file=sys.stderr)

    if not per_seq_bundles:
        print("error: no predictions were matched against ground truth, nothing to report", file=sys.stderr)
        return 1

    if len(per_seq_bundles) > 1:
        combined = combine_bundles(per_seq_bundles)
        combined_summary = summarize_bundle(combined)
        report = format_report(f"OVERALL  ({len(per_seq_bundles)} sequences)", combined_summary)
        (out_dir / "summary.txt").write_text(report)
        print(
            f"\n{'OVERALL':<20s} HOTA={combined_summary['HOTA']:6.2f}  MOTA={combined_summary['MOTA']:6.2f}"
            f"  IDF1={combined_summary['IDF1']:6.2f}  -> {out_dir / 'summary.txt'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
