import numpy as np
import pandas as pd

from run_trackers import CONFIGS, LABELS, label_misses, run_tracker, safe_run
from trackid import eval_mot, io

VIDEO, N = "IMG_0106", 300


def gt_rows():
    gt = eval_mot._read_mot_txt(io.GT_BBOX / f"{VIDEO}.txt")
    return gt[gt[:, 0] <= N]


def as_dets(gt, drop=None):
    d = pd.DataFrame(gt[:, [0, 2, 3, 4, 5]], columns=["frame", "x", "y", "w", "h"])
    d["conf"] = 0.95
    d["frame"] = d.frame.astype(int)
    return d if drop is None else d[~drop]


def as_tracks(gt):
    return np.c_[gt[:, :6], np.ones((len(gt), 2))]


def test_gt_as_tracks_scores_perfect_and_all_tracked():
    gt = gt_rows()
    summary, labels = label_misses(gt, as_tracks(gt), as_dets(gt), 0.1)
    assert summary["HOTA"] > 99.9 and summary["IDSW"] == 0
    assert (labels.label == "tracked").all() and len(labels) == len(gt)


def test_missing_box_is_blamed_on_detector_or_tracker():
    gt = gt_rows()
    gone = gt[:, 0] == 100
    tracks = as_tracks(gt[~gone])
    _, with_det = label_misses(gt, tracks, as_dets(gt), 0.1)
    _, no_det = label_misses(gt, tracks, as_dets(gt, drop=gone), 0.1)
    assert (with_det[with_det.frame == 100].label == "lost_assoc").all()
    assert (no_det[no_det.frame == 100].label == "lost_no_det").all()


def test_low_score_box_is_labelled_low_conf():
    gt = gt_rows()
    gone = gt[:, 0] == 100
    d = as_dets(gt)
    d.loc[gone, "conf"] = 0.05
    _, labels = label_misses(gt, as_tracks(gt[~gone]), d, 0.1)
    assert (labels[labels.frame == 100].label == "lost_low_conf").all()


def test_swapped_ids_count_as_switches():
    gt = gt_rows()
    t = as_tracks(gt)
    ids = np.unique(gt[:, 1])[:2]
    late = t[:, 0] > 150
    a, b = t[:, 1] == ids[0], t[:, 1] == ids[1]
    t[late & a, 1], t[late & b, 1] = ids[1], ids[0]
    summary, labels = label_misses(gt, t, as_dets(gt), 0.1)
    assert summary["IDSW"] == 2 and (labels.label == "idsw").sum() == 2


def test_every_config_runs_and_recovers_clean_detections():
    gt = gt_rows()
    for cfg in CONFIGS:
        tracks = run_tracker(cfg, as_dets(gt), N, CONFIGS[cfg][2])
        summary, labels = label_misses(gt, tracks, as_dets(gt), CONFIGS[cfg][2])
        assert summary["DetA"] > 90, cfg
        assert set(labels.label) <= set(LABELS)


def test_failed_run_is_reported_not_raised():
    job, res, err = safe_run(("IMG_0106", "/no/such/detections.txt", "sort__default"))
    assert res is None and "No such file" in err
