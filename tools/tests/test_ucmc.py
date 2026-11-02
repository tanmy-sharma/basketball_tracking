import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tracker.evaluate import score  # noqa: E402
from tracker.kalman import TrackStatus  # noqa: E402
from tracker.mapper import HomographyMapper  # noqa: E402
from tracker.run_ucmc import Det, Params, in_court, track  # noqa: E402
from tracker.ucmc import UCMCTrack  # noqa: E402
from trackid import io  # noqa: E402

VIDEO, N = "IMG_0106", 600
H = io.load_homographies(io.TRAIN / "court_registration.csv")[0]


def gt_dets(n=N):
    b = io.load_bboxes(VIDEO)
    return b[b.frame <= n].drop(columns="track_id").assign(conf=0.95)


def test_mapper_gives_metres_and_matches_gt():
    b = io.load_bboxes(VIDEO).iloc[0]
    g = io.apply_h(H, [[b.x + b.w / 2, b.y + b.h]])[0]
    y, R = HomographyMapper(H, 0.05, 0.05).mapto(np.array([b.x, b.y, b.w, b.h]))
    assert y[:, 0] * 100 == pytest.approx(g, abs=1e-6)
    assert R.shape == (2, 2) and np.all(np.linalg.eigvalsh(R) > 0)


def test_rows_outside_court_are_never_written():
    d = gt_dets()
    far = d[d.frame % 2 == 0].assign(x=100.0, y=1800.0, w=60.0, h=200.0)  # foot point well off the court
    out = track(pd.concat([d, far]).sort_values("frame", kind="stable"), H, Params(input_margin=5000.0), N)
    assert in_court(out.x.to_numpy(), out.y.to_numpy(), 0).all()


def test_clean_detections_do_not_shatter_into_many_tracks():
    # The defaults favour pure pieces over long ones: 6 players gave 9 tracks here when this was last checked
    out = track(gt_dets(), H, Params(), N)
    r = out[out.kind == "det"]
    assert r.groupby("frame").size().median() >= 5
    assert r.id.nunique() <= 12


def test_cdt_counts_frames_since_last_observation():
    # A track unobserved for cdt frames is still alive; on the next miss it is deleted
    out = track(gt_dets(100), H, Params(cdt=5, emit_coast=999), 150)
    late = out[out.frame > 100]
    assert (late.kind == "coast").all() and late.frame.max() == 105


def test_retention_is_separate_from_what_is_written():
    out = track(gt_dets(100), H, Params(cdt=5, emit_coast=0), 150)
    assert (out.kind == "det").all() and out.frame.max() <= 100


def two_tracks_and_a_box_between():
    t = UCMCTrack(20.0, 20.0, 25.0, 25.0, 10.0, 8, 30.0, 0.7, ambiguity=1.0)
    R = np.eye(2) * 0.01
    mk = lambda x: Det(np.array([[x], [5.0]]), R, 0.9, 50.0, 200.0, np.zeros(4))
    for f in range(1, 6):  # two tracks, 1 m apart, both confirmed
        t.update([mk(5.0), mk(6.0)], f)
    assert len(t.confirmed_idx) == 2
    return t, mk


def test_ambiguous_box_is_not_committed_and_starts_no_track():
    t, mk = two_tracks_and_a_box_between()
    before = [trk.kf.x.copy() for trk in t.trackers]
    t.update([mk(5.5)], 6)  # exactly between them
    assert t.stats["abstained"] == 1 and len(t.trackers) == 2
    assert all(trk.detidx == -1 and trk.status == TrackStatus.Coasted for trk in t.trackers)
    # Neither track moved towards the box beyond its own prediction
    assert all(abs(trk.kf.x[0, 0] - x[0, 0]) < 0.05 for trk, x in zip(t.trackers, before))


def test_clear_box_is_still_committed():
    t, mk = two_tracks_and_a_box_between()
    t.update([mk(5.0)], 6)
    assert t.stats["abstained"] == 0 and t.trackers[0].detidx == 0


def test_gate_rejects_a_far_box_that_the_ranking_limit_would_accept():
    R = np.eye(2) * 0.01
    far = Det(np.array([[7.5], [5.0]]), R, 0.9, 50.0, 200.0, np.zeros(4))
    near = lambda: Det(np.array([[5.0], [5.0]]), R, 0.9, 50.0, 200.0, np.zeros(4))
    for gate, matched in ((0.0, True), (9.21, False)):
        t = UCMCTrack(1000.0, 1000.0, 25.0, 25.0, 10.0, 8, 30.0, 0.7, gate=gate)
        for f in range(1, 6):
            t.update([near()], f)
        t.update([far], 6)
        assert (t.trackers[0].detidx == 0) is matched


def test_noise_falls_with_confidence():
    m = HomographyMapper(H, 0.02, 0.02, 1.0)
    box = np.array([2000.0, 1300.0, 100.0, 270.0])
    low, high = np.linalg.det(m.mapto(box, 0.3)[1]), np.linalg.det(m.mapto(box, 0.95)[1])
    assert low > 10 * high


def test_perfect_tracks_score_near_one():
    b = io.load_bboxes(VIDEO)
    p = io.apply_h(H, np.c_[b.x + b.w / 2, b.y + b.h])
    t = pd.DataFrame(dict(frame=b.frame, id=b.track_id, x=p[:, 0], y=p[:, 1], raw_x=p[:, 0], raw_y=p[:, 1], kind="det",
                          conf=1.0, bx=b.x, by=b.y, bw=b.w, bh=b.h))
    r = score(VIDEO, t)
    assert r["TI"] > 0.999 and r["tracks"] == 6 and r["spurious_tracks"] == 0


# Team labels -------------------------------------------------------------------------------------------------------

def team_det(x, team, evidence=7.0):
    return Det(np.array([[x], [5.0]]), np.eye(2) * 0.01, 0.9, 50.0, 200.0, np.zeros(4), team=team, evidence=evidence)


def test_no_labels_gives_the_same_tracks_with_team_off_and_on():
    d = gt_dets(300)
    a = track(d, H, Params(), 300)
    b = track(d, H, Params(team_scale=2.0, team_veto=3.0, team_birth=1), 300)
    assert a.equals(b)


def test_team_belief_adds_evidence_and_is_capped():
    t = UCMCTrack(20.0, 20.0, 25.0, 25.0, 10.0, 8, 30.0, 0.7, team_scale=1.0, team_cap=10.0)
    for f in range(1, 4):
        t.update([team_det(5.0, 1)], f)
    assert t.trackers[0].team_belief == pytest.approx(10.0)  # 3 x 7 capped at 10
    t.update([team_det(5.0, -1, 2.0)], 4)
    assert t.trackers[0].team_belief == pytest.approx(8.0)


def test_team_lock_fixes_a_track_sure_for_enough_frames_in_a_row():
    t = UCMCTrack(20.0, 20.0, 25.0, 25.0, 10.0, 8, 30.0, 0.7, team_scale=0.0, team_cap=10.0, team_lock_frames=5)
    for f in range(1, 5):  # 4 sure frames: not yet locked, so contrary calls still count
        t.update([team_det(5.0, 1)], f)
    t.update([team_det(5.0, -1, 8.0)], 5)  # belief 10 -> 2: no longer sure, so the run restarts
    assert t.trackers[0].team_belief == pytest.approx(2.0)
    assert not getattr(t.trackers[0], "team_locked", False)
    for f in range(6, 10):  # sure again from frame 6: 4 in a row, still unlocked
        t.update([team_det(5.0, 1)], f)
    assert not getattr(t.trackers[0], "team_locked", False)
    t.update([team_det(5.0, 1)], 10)  # 5 in a row -> locked at the cap
    assert t.trackers[0].team_locked and t.trackers[0].team_belief == pytest.approx(10.0)
    for f in range(11, 16):
        t.update([team_det(5.0, -1, 7.0)], f)
    assert t.trackers[0].team_belief == pytest.approx(10.0)


def test_penalty_is_zero_without_a_call_or_belief_and_grows_with_disagreement():
    p = UCMCTrack.team_penalty
    assert p(7.0, 0.0) == 0.0  # track has no belief
    assert p(7.0, 10.0) == 0.0  # agrees: no reward either
    assert p(7.0, -10.0) == pytest.approx(2 * 7.0 - 2 * np.log(2), abs=0.2)  # about 2 logit(q); the track is 99.995% sure, not certain
    assert p(2.0, -10.0) < p(7.0, -10.0)  # a less reliable call costs less


def test_box_of_the_other_team_goes_to_the_matching_track():
    # Two sure tracks 1 m apart; a box nearer the wrong-team track joins the right one once the penalty is on
    for scale, expect in ((0.0, 0), (1.0, 1)):
        t = UCMCTrack(20.0, 20.0, 25.0, 25.0, 10.0, 8, 30.0, 0.7, team_scale=scale)
        for f in range(1, 6):
            t.update([team_det(5.0, 1), team_det(5.3, -1)], f)
        t.update([team_det(5.12, -1)], 6)
        assert [trk.detidx for trk in t.trackers].index(0) == expect


def test_veto_blocks_a_clear_box_of_the_other_team():
    t = UCMCTrack(20.0, 20.0, 25.0, 25.0, 10.0, 8, 30.0, 0.7, team_veto=5.0)
    for f in range(1, 6):
        t.update([team_det(5.0, 1)], f)
    t.update([team_det(5.0, -1)], 6)
    assert t.trackers[0].detidx == -1  # the clear wrong-team box starts a new track instead
    assert len(t.trackers) == 2


def test_birth_is_blocked_when_the_team_already_has_three_matched_tracks():
    for birth, n in ((False, 4), (True, 3)):
        t = UCMCTrack(20.0, 20.0, 25.0, 25.0, 10.0, 8, 30.0, 0.7, team_birth=birth)
        for f in range(1, 6):
            t.update([team_det(x, 1) for x in (2.0, 5.0, 8.0)], f)
        t.update([team_det(x, 1) for x in (2.0, 5.0, 8.0, 11.0)], 6)
        assert len(t.trackers) == n
