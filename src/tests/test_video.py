from __future__ import annotations

import numpy as np
import pytest

from src.dataio.video import frame_count, iter_frames, probe, read_frames, scaled_size, video_path
from src.tests.fixtures import frame_index, make_video, make_training_tree, patch_trackid_paths


@pytest.fixture(scope="module")
def vid(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("v") / "clip.mkv", 23, w=128, h=72)


def test_probe_and_count(vid):
    assert probe(vid) == (128, 72)
    assert frame_count(vid) == 23


def test_frame_numbers_match_colour(vid):
    got = [(n, frame_index(img)) for n, img in iter_frames(vid, 5, height=72)]
    assert got == [(1, 0), (6, 5), (11, 10), (16, 15), (21, 20)]


def test_rgb_and_downscale(vid):
    n, img = next(iter_frames(vid, 1, height=36))
    assert img.shape == (36, 64, 3) and img.dtype == np.uint8
    assert tuple(img[10, 10]) == (0, 0, 77)  # R, G, B order
    assert scaled_size(vid, 36) == (64, 36, 2.0)


def test_limit(vid):
    assert [n for n, _ in iter_frames(vid, 2, height=36, limit=3)] == [1, 3, 5]
    assert list(iter_frames(vid, 2, height=36, limit=0)) == []


def test_read_frames(vid):
    got = read_frames(vid, [12, 3, 3, 100], height=36)
    assert sorted(got) == [3, 12] and frame_index(got[12]) == 11


def test_not_16_9(tmp_path):
    p = make_video(tmp_path / "sq.mkv", 2, w=64, h=64)
    with pytest.raises(ValueError):
        scaled_size(p, 32)


def test_video_path(tmp_path, monkeypatch):
    root = make_training_tree(tmp_path / "T", ["IMG_0106"], n_frames=2)
    patch_trackid_paths(monkeypatch, root)
    p = make_video(root / "vidoes" / "img_0106.MOV", 2, w=64, h=36)
    assert video_path("IMG_0106") == p
    with pytest.raises(FileNotFoundError):
        video_path("IMG_0107")
    with pytest.raises(ValueError):
        video_path("IMG_0114")
