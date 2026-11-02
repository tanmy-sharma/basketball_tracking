from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from src.core.config import CropCfg
from src.crops.extract import extract_game, write_overlays
from src.crops.pose import ViTPose
from src.dataio.schema import CROP_COLUMNS
from src.dataio.store import CropStore, crops_root, load_manifest
from src.tests.fixtures import frame_index, make_training_tree, make_video, patch_trackid_paths

W, H = 1280, 720  # 4K / 3


class FakePose:
    """Keypoints at fixed fractions of each box; box 0 gets low confidence when `low_first` is set."""

    def __init__(self, low_first: bool = False) -> None:
        self.low_first, self.calls = low_first, 0

    def __call__(self, frame_rgb, boxes_xywh):
        self.calls += 1
        assert frame_rgb.shape == (H, W, 3)
        out = np.zeros((len(boxes_xywh), 17, 3), np.float32)
        for i, (x, y, w, h) in enumerate(boxes_xywh):
            for k, (fx, fy) in {5: (.75, .25), 6: (.25, .25), 11: (.7, .55), 12: (.3, .55)}.items():
                out[i, k] = (x + fx * w, y + fy * h, .1 if (self.low_first and i == 0) else .9)
        return out


def setup(tmp_path, monkeypatch, label_frames=60, video_frames=60, **tree_kw):
    root = make_training_tree(tmp_path / "Training", ["IMG_0106"], n_frames=label_frames, **tree_kw)
    patch_trackid_paths(monkeypatch, root)
    make_video(root / "vidoes" / "IMG_0106.MOV", video_frames, w=W, h=H)
    return root


CFG = CropCfg(every=5, min_side_px=4)


def test_extract_rows_offsets_and_pixels(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, stray_track=True)
    d = extract_game("IMG_0106", CFG, FakePose(low_first=True))
    assert d == crops_root(CFG) / "IMG_0106" and (d / "done.json").exists()
    assert not list(d.parent.glob("*.tmp"))
    df = load_manifest(crops_root(CFG), ["IMG_0106"])
    assert list(df.columns) == CROP_COLUMNS
    assert sorted(df.frame.unique()) == list(range(1, 61, 5)) and len(df) == 12 * 6
    assert set(df.track_id) == {1, 2, 3, 4, 5, 6}  # stray track 0 dropped
    assert set(df[df.status == "ok"].facing) == {"front"} and set(df[df.status == "low_pose"].facing) == {"unknown"}
    assert (df.status == "low_pose").sum() == 12 and (df.status == "ok").sum() == 60
    assert df.pose_min_conf.min() == pytest.approx(.1, abs=1e-6)
    # box 0 of each frame is track 1 at x=300+2f (4K) -> /3 at 720p
    r = df[(df.frame == 6) & (df.track_id == 1)].iloc[0]
    assert (r.box_x, r.box_y, r.box_w, r.box_h) == pytest.approx((312 / 3, 200.0, 140 / 3, 400 / 3), abs=1e-6)
    store = CropStore(crops_root(CFG))
    for _, r in df.iterrows():
        px = store.read("IMG_0106", r.offset, r.crop_w, r.crop_h)
        assert (r.crop_x1 - r.crop_x0, r.crop_y1 - r.crop_y0) == (px.shape[1], px.shape[0])
        assert px.shape[1] >= 4 and (px[..., 0] == (r.frame - 1) % 256).all()  # cut from the right frame
    offs = np.sort(df.offset.to_numpy())
    sizes = (df.crop_w * df.crop_h * 3).to_numpy()[np.argsort(df.offset.to_numpy())]
    assert (offs[1:] == (offs[:-1] + sizes[:-1])).all()  # contiguous
    kps = np.load(d / "keypoints.npy")
    assert kps.shape == (len(df), 17, 3) and kps.dtype == np.float32
    meta = json.loads((d / "meta.json").read_text())
    assert meta["decoded_sampled_frames"] == 12 and meta["unlabelled_tail_frames"] == 0


def test_extract_labels_end_before_video(tmp_path, monkeypatch, caplog):
    setup(tmp_path, monkeypatch, label_frames=50, video_frames=60)
    with caplog.at_level(logging.WARNING):
        d = extract_game("IMG_0106", CFG, FakePose())
    assert "labels end at frame 50" in caplog.text and "10 unlabelled" in caplog.text
    meta = json.loads((d / "meta.json").read_text())
    assert meta["unlabelled_tail_frames"] == 10 and meta["decoded_sampled_frames"] == 12
    assert load_manifest(crops_root(CFG), ["IMG_0106"]).frame.max() == 46


def test_undecodable_tail_after_the_labels_is_accepted(tmp_path, monkeypatch, caplog):
    """IMG_0104 on Colab: the container counts more frames than ffmpeg decodes, all of them after the last label."""
    import src.crops.extract as ex
    setup(tmp_path, monkeypatch, label_frames=50, video_frames=60)
    monkeypatch.setattr(ex, "frame_count", lambda p: 80)
    with caplog.at_level(logging.WARNING):
        d = extract_game("IMG_0106", CFG, FakePose())
    assert "decoded 12 of 16 sampled frames" in caplog.text
    assert load_manifest(crops_root(CFG), ["IMG_0106"]).frame.max() == 46


def test_labelled_frames_that_do_not_decode_still_raise(tmp_path, monkeypatch):
    import src.crops.extract as ex
    setup(tmp_path, monkeypatch, label_frames=60, video_frames=40)
    monkeypatch.setattr(ex, "frame_count", lambda p: 60)
    with pytest.raises(RuntimeError, match="decoded 8 sampled frames, expected 12-12"):
        extract_game("IMG_0106", CFG, FakePose())
    assert not (crops_root(CFG) / "IMG_0106" / "done.json").exists()


def test_extract_video_shorter_than_labels_raises(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, label_frames=60, video_frames=40)
    with pytest.raises(RuntimeError, match="40 frames but labels reach frame 60"):
        extract_game("IMG_0106", CFG, FakePose())
    assert not (crops_root(CFG) / "IMG_0106" / "done.json").exists()


def test_limit_frames_skips_check_and_stops_early(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, label_frames=60, video_frames=40)
    cfg = CropCfg(every=5, min_side_px=4, limit_frames=20)
    d = extract_game("IMG_0106", cfg, FakePose())
    assert sorted(load_manifest(crops_root(cfg), ["IMG_0106"]).frame.unique()) == [1, 6, 11, 16]
    assert json.loads((d / "meta.json").read_text())["limit_frames"] == 20


def test_degenerate_box_has_no_pixels(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    cfg = CropCfg(every=5, min_side_px=4000)  # nothing can pass min_side
    extract_game("IMG_0106", cfg, FakePose())
    df = load_manifest(crops_root(cfg), ["IMG_0106"])
    assert (df.status == "degenerate").all() and (df.offset == -1).all() and (df.crop_w == 0).all()


def test_challenge_refused(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        extract_game("IMG_0114", CFG, FakePose())


def test_overlays(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, label_frames=50, video_frames=60)
    out = write_overlays("IMG_0106", tmp_path / "ov", height=H)
    assert [p.name for p in out] == ["IMG_0106_f000001.png", "IMG_0106_f000026.png", "IMG_0106_f000050.png"]
    import cv2
    img = cv2.imread(str(out[0]))
    assert img.shape == (H, W, 3) and (img[..., 1] == 255).any()  # green box lines drawn


def test_vitpose_wrapper_tiny_random():
    pytest.importorskip("transformers")
    from transformers import VitPoseBackboneConfig, VitPoseConfig, VitPoseForPoseEstimation, VitPoseImageProcessor
    bb = VitPoseBackboneConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, mlp_ratio=2,
                               image_size=[256, 192], patch_size=[16, 16], out_indices=[2])
    model = VitPoseForPoseEstimation(VitPoseConfig(backbone_config=bb, num_labels=17))
    pose = ViTPose("unused", None, "cpu", model=model, processor=VitPoseImageProcessor())
    frame = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    out = pose(frame, np.array([[10, 20, 40, 100], [100, 30, 50, 120], [200, 40, 30, 90]], float))
    assert out.shape == (3, 17, 3) and out.dtype == np.float32 and np.isfinite(out).all()
    assert pose(frame, np.zeros((0, 4))).shape == (0, 17, 3)
