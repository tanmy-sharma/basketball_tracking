from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.constants import NUMBER_TO_IDX as I
from src.evaluate import metrics as M


def onehot(idx, p=.9):
    out = np.full((len(idx), 9), (1 - p) / 8)
    out[np.arange(len(idx)), idx] = p
    return out


def test_masked_accuracy_where_nine_way_is_wrong_and_three_way_right():
    y = np.array([I["6"], I["9"]])
    probs = np.zeros((2, 9))
    probs[0, I["1"]] = .5; probs[0, I["6"]] = .3      # 9-way says 1, roster {6,9,11} says 6
    probs[1, I["9"]] = .6; probs[1, I["12"]] = .4
    allowed = np.zeros((2, 9), bool)
    allowed[:, [I["6"], I["9"], I["11"]]] = True
    assert M.per_crop_accuracy(y, probs) == .5
    assert M.within_team_accuracy(y, probs, allowed) == 1.0


def test_confusion_counts():
    cm = M.confusion([0, 0, 1, 8], [0, 1, 1, 7])
    assert cm.shape == (9, 9) and cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 1 and cm[8, 7] == 1 and cm.sum() == 4


def test_accuracy_by_bib_swap_rates():
    df = pd.DataFrame({"bib": ["green"] * 4 + ["yellow"] * 2, "jersey": ["6", "6", "9", "9", "10", "11"]})
    pred = [I["6"], I["9"], I["6"], I["6"], I["10"], I["11"]]
    allowed = np.zeros((6, 9), bool)
    allowed[:, [I["6"], I["9"], I["10"], I["11"]]] = True
    out = M.accuracy_by_bib(df, onehot(pred), allowed).set_index(["bib", "number"])
    assert set(out.index) == {("green", "6"), ("green", "9"), ("yellow", "10")}
    assert out.loc[("green", "6"), "acc_9way"] == .5 and out.loc[("green", "6"), "swap_rate"] == .5
    assert out.loc[("green", "9"), "acc_9way"] == 0 and out.loc[("green", "9"), "swap_rate"] == 1.0
    assert out.loc[("yellow", "10"), "acc_team"] == 1.0


def votes_fixture():
    # one track (truth 6) in windows of 1 s at fps 10: frames 0-9 -> window 0, 10-19 -> window 1
    df = pd.DataFrame({"video": "V", "track_id": 1, "frame": [0, 1, 2, 10, 11, 12],
                       "number_idx": I["6"]})
    pred = [I["6"], I["6"], I["9"], I["9"], I["9"], I["6"]]
    probs = onehot(pred, .8)
    return df, probs


def test_vote_windows_and_abstention():
    df, probs = votes_fixture()
    whole = M.tracklet_vote(df, probs, None, np.inf, fps=10)
    assert len(whole) == 1 and whole.n_crops.iloc[0] == 6
    w1 = M.tracklet_vote(df, probs, None, 1.0, fps=10)
    assert w1.window.tolist() == [0, 1] and w1.correct.tolist() == [True, False]
    wt = np.array([1, 1, 0, 0, 0, 0.])  # window 1 has no eligible crops -> abstains
    v = M.tracklet_vote(df, probs, wt, 1.0, fps=10)
    assert v.abstained.tolist() == [False, True] and v.pred_idx.tolist() == [I["6"], -1]
    s = M.summarize_votes(v)
    assert s["abstain_rate"] == .5 and s["accuracy_voted"] == 1.0 and s["accuracy_all"] == .5


def test_vote_roster_mask_changes_outcome():
    df = pd.DataFrame({"video": "V", "track_id": 1, "frame": [0, 1], "number_idx": I["6"]})
    probs = np.zeros((2, 9)); probs[:, I["1"]] = .5; probs[:, I["6"]] = .3
    allowed = np.zeros((2, 9), bool); allowed[:, [I["6"], I["9"], I["11"]]] = True
    assert not M.tracklet_vote(df, probs, None, np.inf).correct.iloc[0]
    assert M.tracklet_vote(df, probs, None, np.inf, allowed).correct.iloc[0]


def test_coverage_curve_monotone_and_values():
    y = np.array([0, 1, 2, 3])
    probs = onehot([0, 1, 0, 0])               # correct, correct, wrong, wrong
    scores = np.array([.9, .6, .4, .1])
    c = M.coverage_curve(y, probs, scores, [0, .5, .8, 1.0])
    assert c.coverage.tolist() == [1.0, .5, .25, 0.0]
    assert c.accuracy.iloc[:3].tolist() == [.5, 1.0, 1.0] and np.isnan(c.accuracy.iloc[3])
    assert (np.diff(c.coverage) <= 0).all()


def test_auc():
    assert M.auc([0, 0, 1, 1], [.1, .2, .3, .4]) == 1.0
    assert M.auc([1, 1, 0, 0], [.1, .2, .3, .4]) == 0.0
    assert M.auc([0, 1, 0, 1], [.5, .5, .5, .5]) == .5
    assert M.auc([0, 1, 0, 1], [.1, .35, .4, .8]) == .75
    assert np.isnan(M.auc([1, 1], [.1, .2]))


def test_accuracy_by_track_and_free_reads():
    df = pd.DataFrame({"crop_id": list("abcd"), "video": "V", "track_id": [1, 1, 2, 2], "jersey": ["6", "6", "9", "9"],
                       "number_idx": [I["6"]] * 2 + [I["9"]] * 2, "facing": ["front", "side", "front", "front"]})
    t = M.accuracy_by_track(df, [I["6"], I["9"], I["9"], I["9"]]).set_index("track_id")
    assert t.accuracy.to_dict() == {1: .5, 2: 1.0}
    allowed = np.zeros((4, 9), bool); allowed[:, [I["6"], I["9"], I["11"]]] = True
    reads = pd.DataFrame({"crop_id": ["a", "b", "c", "x"], "text": ["6", "12", "", "9"]})
    s = M.free_read_stats(df, reads, allowed)
    assert s["n"] == 3 and s["exact_rate"] == pytest.approx(1 / 3) and s["empty_rate"] == pytest.approx(1 / 3)
    assert s["off_roster_rate"] == pytest.approx(1 / 3) and s["off_roster_of_nonempty"] == .5
    assert M.free_read_stats(df, pd.DataFrame({"crop_id": ["zz"], "text": ["1"]}), allowed) == {"n": 0}
