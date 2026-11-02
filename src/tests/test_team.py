"""Team colours: calibration, classifying a crop, rejecting other people, and the checks around GT."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from src.team import run
from src.team.colour import BIB_REFERENCE, TeamColours, calibrate, crop_colour


def lab_bgr(a: float, b: float, L: int = 180) -> tuple[int, int, int]:
    pixel = np.array([[[L, a + 128, b + 128]]], np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0])


def player(bib_ab: tuple[float, float] | None, size=(90, 40)) -> np.ndarray:
    """Court-blue background with a bib, or a black shirt, across the chest."""
    h, w = size
    crop = np.full((h, w, 3), (170, 60, 30), np.uint8)
    chest = lab_bgr(*bib_ab) if bib_ab else (15, 15, 15)
    crop[int(.1 * h):int(.55 * h), int(.2 * w):int(.8 * w)] = chest
    return crop


TEAMS = {"yellow": "A", "green": "B"}


def fitted() -> TeamColours:
    rng = np.random.default_rng(0)
    samples = np.r_[rng.normal((-7, 30), 2, (30, 2)), rng.normal((-30, 25), 2, (30, 2))]
    return calibrate("V", samples, TEAMS, [5, 9])


def test_calibration_measures_this_videos_colours_and_names_them():
    tc = fitted()
    assert np.linalg.norm(np.array(tc.centroids["green"]) - (-30, 25)) < 2
    assert np.linalg.norm(np.array(tc.centroids["yellow"]) - (-7, 30)) < 2
    assert tc.teams == TEAMS and tc.warnings == []


def test_calibration_ignores_bibs_of_teams_not_playing_and_skin():
    rng = np.random.default_rng(1)
    clean = np.r_[rng.normal((-7, 30), 2, (30, 2)), rng.normal((-18, 22), 2, (30, 2))]
    with_others = np.r_[clean, rng.normal((15, -21), 2, (40, 2)), rng.normal((13, 18), 2, (40, 2))]
    a, b = calibrate("V", clean, TEAMS), calibrate("V", with_others, TEAMS)
    assert np.linalg.norm(np.array(a.centroids["yellow"]) - b.centroids["yellow"]) < 1
    assert np.linalg.norm(np.array(a.centroids["green"]) - b.centroids["green"]) < 1


def test_calibration_warns_when_one_colour_is_missing_or_colours_are_too_close():
    rng = np.random.default_rng(2)
    assert any("green" in w for w in calibrate("V", rng.normal((-7, 30), 2, (30, 2)), TEAMS).warnings)
    assert any("apart" in w for w in calibrate("V", rng.normal((-12, 26), 1, (60, 2)), TEAMS).warnings)


def test_run_returns_colour_and_team_from_a_crop():
    tc = fitted()
    got = tc.run(player((-7, 30)))
    assert (got.colour, got.team) == ("yellow", "A")
    got = tc.run(player((-30, 25)))
    assert (got.colour, got.team) == ("green", "B")


def test_background_removes_grass_behind_a_black_shirt():
    grass = lab_bgr(-20, 30, 120)
    crop = np.full((90, 40, 3), grass, np.uint8)
    crop[:, 16:24] = (15, 15, 15)
    plain = np.full((90, 40, 3), grass, np.uint8)
    assert crop_colour(crop)[0] == "ok"  # the failure this exists for
    assert crop_colour(crop, True, plain)[0] == "no_colour"
    assert fitted().run(crop, True, plain).team is None


def test_background_keeps_a_bib_against_grass():
    grass = lab_bgr(-20, 30, 120)
    crop = np.full((90, 40, 3), grass, np.uint8)
    crop[:, 12:28] = lab_bgr(15, -21, 120)
    status, ab = crop_colour(crop, True, np.full((90, 40, 3), grass, np.uint8))
    assert status == "ok" and np.linalg.norm(ab - (15, -21)) < 4


def test_background_image_removes_a_moving_player():
    frames = [np.full((36, 64, 3), 100, np.uint8) for _ in range(5)]
    for i, img in enumerate(frames):
        img[10:20, i * 10:i * 10 + 8] = 255
    assert np.median(np.stack(frames), axis=0).max() == 100


def test_court_blue_is_not_taken_for_a_bib():
    crop = np.full((90, 40, 3), (170, 60, 30), np.uint8)
    assert crop_colour(crop)[0] == "no_colour"


def test_black_shirt_and_other_bibs_are_not_a_team():
    tc = fitted()
    got = tc.run(player(None))
    assert (got.colour, got.team, got.status) == ("none", None, "no_colour")
    got = tc.run(player(BIB_REFERENCE["purple"]))
    assert (got.colour, got.team) == ("none", None)


def test_between_the_two_colours_is_unsure():
    tc = fitted()
    middle = (np.array(tc.centroids["yellow"]) + tc.centroids["green"]) / 2
    got = tc.classify(middle)
    assert (got.colour, got.team) == ("unsure", None)


def test_tiny_crop_is_unsure_not_a_crash():
    tc = fitted()
    got = tc.run(np.zeros((3, 3, 3), np.uint8))
    assert (got.colour, got.status) == ("unsure", "degenerate")
    assert tc.run(np.zeros((0, 0, 3), np.uint8)).status == "degenerate"


def test_torso_crop_uses_all_of_the_crop():
    tc = fitted()
    torso = np.full((40, 30, 3), lab_bgr(-7, 30), np.uint8)
    assert tc.run(torso, whole_player=False).colour == "yellow"


def test_save_and_load_round_trip(tmp_path):
    tc = fitted()
    tc.save(tmp_path / "c.json")
    again = TeamColours.load(tmp_path / "c.json")
    assert again == tc and again.run(player((-7, 30))).team == "A"


def test_calibration_frames_are_random_inside_the_video_and_repeatable():
    frames = run.calibration_frames(1000, 3)
    assert len(frames) == len(set(frames)) == 10
    assert min(frames) > 50 and max(frames) <= 950
    assert frames == run.calibration_frames(1000, 3) and frames != run.calibration_frames(1000, 4)


def test_gt_is_refused_on_challenge_videos():
    with pytest.raises(ValueError):
        run.load_boxes("IMG_0114", "gt")


@pytest.mark.skipif(not (run.io.FINAL / "games_roster.csv").exists() or not (run.io.TRAIN / "games_roster.csv").exists(),
                    reason="competition data isn't in git")
def test_rosters_give_the_expected_bibs():
    assert run.bibs_of("IMG_0114") == {"yellow": "A", "green": "B"}
    assert run.bibs_of("IMG_0115") == {"purple": "A", "pink": "B"}
    assert run.bibs_of("IMG_0108") == {"green": "A", "yellow": "B"}


def test_labels_follow_the_agreed_format():
    tc = fitted()
    table = pd.DataFrame({"frame": [1, 1, 1], "det_row": [0, 1, 2], "x": 0., "y": 0., "w": 1., "h": 1., "conf": .9,
                          "status": ["ok", "no_colour", "degenerate"], "a": [-7, np.nan, np.nan], "b": [30, np.nan, np.nan]})
    out = run.classify_table(table, tc)
    assert list(out.team) == ["A", "", ""] and list(out.colour) == ["yellow", "none", "unsure"]
    assert out.p_pink.isna().all() and out.p_purple.isna().all()
    assert out.p_none[1] == 1 and np.isnan(out.p_none[2])


def test_score_counts_precision_and_rejected_non_players():
    table = pd.DataFrame({"frame": [1, 1, 2, 2, 3], "x": [0., 100, 0, 100, 0], "y": 0., "w": 10., "h": 10.,
                          "overlap": 0., "team": ["A", "B", "B", "", ""], "colour": ["yellow", "green", "green", "unsure", "none"],
                          "true_team": ["A", "B", "A", "A", ""], "player": [1, 2, 1, 1, -1], "conf": .9, "gt_iou": [1., 1., 1., 1., 0.]})
    s = run.score(table)
    assert s["precision"] == pytest.approx(2 / 3) and s["coverage"] == pytest.approx(3 / 4)
    assert s["non_player_rejected"] == 1 and s["player_vote_correct"] == 2


def test_decode_with_many_uneven_frames_keeps_numbering(tmp_path):
    from src.tests.fixtures import frame_index, make_video
    path = make_video(tmp_path / "v.mkv", 300, 64, 36)
    wanted = sorted(set(range(1, 301)) - {5, 77})
    got = list(run.decode(path, wanted, 36))
    assert [f for f, _ in got] == wanted
    assert all(frame_index(img[..., ::-1]) == f - 1 for f, img in got[::40])


def test_decode_picks_exact_frames(tmp_path):
    from src.tests.fixtures import frame_index, make_video
    path = make_video(tmp_path / "v.mkv", 30, 64, 36)
    got = list(run.decode(path, [3, 7, 20], 36))
    assert [f for f, _ in got] == [3, 7, 20]
    assert [frame_index(img[..., ::-1]) for _, img in got] == [2, 6, 19]
