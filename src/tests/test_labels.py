from __future__ import annotations

import logging

import numpy as np
import pytest

from src.dataio.labels import load_labelled_boxes, roster_mask
from src.tests.fixtures import make_training_tree, patch_trackid_paths


@pytest.fixture
def tree(tmp_path, monkeypatch):
    def build(**kw):
        root = make_training_tree(tmp_path / "Training", ["IMG_0106", "IMG_0107"], n_frames=10, **kw)
        patch_trackid_paths(monkeypatch, root)
        return root
    return build


def test_identity_bib_and_columns(tree):
    tree()
    df = load_labelled_boxes("IMG_0106")
    assert len(df) == 60 and df.in_game.all()
    t1, t4 = df[df.track_id == 1].iloc[0], df[df.track_id == 4].iloc[0]
    assert (t1.team, t1.jersey, t1.bib, t1.number_idx) == ("A", "6", "green", 4)
    assert (t4.team, t4.jersey, t4.bib, t4.number_idx) == ("B", "1", "pink", 0)
    assert df[df.track_id == 1].x.iloc[0] == 302
    assert load_labelled_boxes("IMG_0107").query("track_id == 1").bib.iloc[0] == "yellow"


def test_stray_track_dropped_and_logged(tree, caplog):
    tree(stray_track=True)
    with caplog.at_level(logging.WARNING):
        df = load_labelled_boxes("IMG_0106")
    assert 0 not in set(df.track_id) and len(df) == 60
    assert "off-roster" in caplog.text and "[0]" in caplog.text


def test_duplicates_dropped_and_logged(tree, caplog):
    tree(duplicate_row=True)
    with caplog.at_level(logging.WARNING):
        df = load_labelled_boxes("IMG_0106")
    assert not df.duplicated(["frame", "track_id"]).any() and len(df) == 60
    assert "duplicate" in caplog.text


def test_roster_mask(tree):
    tree()
    m = roster_mask("IMG_0106", "A")
    assert m.dtype == bool and m.tolist() == [False, False, False, False, True, True, False, True, False]
    assert roster_mask("IMG_0106", "B").sum() == 3
    with pytest.raises(KeyError):
        roster_mask("IMG_0106", "C")


def test_challenge_refused(tree):
    tree()
    with pytest.raises(ValueError):
        load_labelled_boxes("IMG_0114")
    with pytest.raises(ValueError):
        roster_mask("IMG_0115", "A")
