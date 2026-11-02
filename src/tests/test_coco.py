from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.core.config import Config, GamesCfg, load_config
from src.core.paths import ROOT
from src.detector import coco
from src.tests.fixtures import make_training_tree, make_video, patch_trackid_paths

GAMES = ["IMG_0106", "IMG_0107"]


def detector_cfg(games: list[str] = GAMES, **det) -> Config:
    return Config.model_validate({
        "run_name": "det", "plugin": "detector", "device": "cpu",
        "games": {"train": games, "test": ["IMG_0108"]}, "split": {"guard_frames": 0},
        "detector": {"sample_every": 5, "num_workers": 0, "predict_batch": 8, **det}})


def make_world(tmp_path: Path, monkeypatch, games: list[str] = GAMES, label_frames: int = 60,
               video_frames: int = 60, **tree) -> Path:
    """Synthetic Training tree with tiny lossless videos; probe() reports a 4K source."""
    root = make_training_tree(tmp_path / "Training", games, n_frames=label_frames, **tree)
    patch_trackid_paths(monkeypatch, root)
    for g in games:
        make_video(root / "vidoes" / f"{g}.mkv", video_frames, 160, 90)
    for mod in ("src.detector.coco", "src.detector.predict"):
        monkeypatch.setattr(f"{mod}.probe", lambda p: (3840, 2160))
    return root


def load(ds: Path, split: str) -> dict:
    return json.loads((ds / split / "_annotations.coco.json").read_text())


def test_export_layout_boxes_and_split(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch)
    ds = coco.build_dataset(detector_cfg())
    tr, va = load(ds, "train"), load(ds, "valid")
    assert ds.parent.name == "detector_dataset"
    assert [c["id"] for c in tr["categories"]] == [1] and tr["categories"][0]["supercategory"] == "player"
    assert (len(tr["images"]), len(va["images"])) == (18, 6)  # 12 sampled frames per game, last 3 are valid
    assert not (ds / "test").exists()
    train_names = {i["file_name"] for i in tr["images"]}
    assert not train_names & {i["file_name"] for i in va["images"]}
    assert all((ds / "train" / n).exists() for n in train_names)
    assert all(i["width"] == 1280 and i["height"] == 720 for i in tr["images"])
    first = next(i for i in tr["images"] if i["file_name"] == "IMG_0106_000001.jpg")
    box = next(a for a in tr["annotations"] if a["image_id"] == first["id"])["bbox"]
    assert box == pytest.approx([302 / 3, 200.0, 140 / 3, 400 / 3], abs=0.01)  # 4K -> 720p is /3
    assert len(tr["annotations"]) == 18 * 6 and all(a["category_id"] == 1 and a["iscrowd"] == 0 for a in tr["annotations"])
    assert all(a["area"] == pytest.approx(a["bbox"][2] * a["bbox"][3], abs=0.01) for a in tr["annotations"])


def test_image_content_matches_frame_number(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch, games=["IMG_0106"])
    ds = coco.build_dataset(detector_cfg(["IMG_0106"]))
    img = cv2.imread(str(ds / "train" / "IMG_0106_000006.jpg"))[:, :, ::-1]
    assert abs(int(img[360, 640, 0]) - 5) <= 3  # frame number 6 is zero-based index 5


def test_unlabelled_tail_is_not_exported(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch, games=["IMG_0106"], label_frames=40, video_frames=60)
    ds = coco.build_dataset(detector_cfg(["IMG_0106"]))
    frames = [int(i["file_name"][9:15]) for s in ("train", "valid") for i in load(ds, s)["images"]]
    assert frames and max(frames) <= 40


def test_stray_track_dropped(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch, games=["IMG_0106"], stray_track=True, duplicate_row=True)
    ds = coco.build_dataset(detector_cfg(["IMG_0106"]))
    c = load(ds, "train")
    per_image = pd.Series([a["image_id"] for a in c["annotations"]]).value_counts()
    assert (per_image == 6).all()


def test_scale_boxes_clamps_and_drops_tiny():
    df = pd.DataFrame({"x": [-30.0, 100.0, 3800.0, 3000.0], "y": [10.0, 10.0, 2000.0, 100.0],
                       "w": [90.0, 3.0, 400.0, 300.0], "h": [300.0, 300.0, 400.0, 3.0]})
    out = coco.scale_boxes(df, 1 / 3, 1 / 3, 1280, 720)
    assert out == [[0.0, 3.33, 20.0, 100.0], [1266.67, 666.67, 13.33, 53.33]]


def test_unit_scale_box_inside_image_is_untouched():
    df = pd.DataFrame({"x": [10.0], "y": [20.0], "w": [30.0], "h": [40.0]})
    assert coco.scale_boxes(df, 1.0, 1.0, 100, 100) == [[10.0, 20.0, 30.0, 40.0]]


def test_test_game_refused(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch)
    cfg = detector_cfg()
    bad = cfg.model_copy(update={"games": GamesCfg.model_construct(train=GAMES + ["IMG_0108"], test=["IMG_0108"],
                                                                    label_pool=[])})
    with pytest.raises(ValueError, match="IMG_0108"):
        coco.build_dataset(bad)
    with pytest.raises(ValueError, match="overlap"):
        Config.model_validate({**cfg.model_dump(mode="json"),
                               "games": {"train": GAMES + ["IMG_0108"], "test": ["IMG_0108"]}})


def test_rebuild_reuses_dataset(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch, games=["IMG_0106"])
    cfg = detector_cfg(["IMG_0106"])
    ds = coco.build_dataset(cfg)
    monkeypatch.setattr(coco, "iter_frames", lambda *a, **k: (_ for _ in ()).throw(AssertionError("decoded again")))
    assert coco.build_dataset(cfg) == ds
    assert coco.dataset_hash(cfg) != coco.dataset_hash(detector_cfg(["IMG_0106"], sample_every=10))


def test_loads_in_pycocotools(tmp_path, monkeypatch):
    pytest.importorskip("pycocotools.coco")
    from pycocotools.coco import COCO
    make_world(tmp_path, monkeypatch, games=["IMG_0106"])
    ds = coco.build_dataset(detector_cfg(["IMG_0106"]))
    api = COCO(str(ds / "train" / "_annotations.coco.json"))
    assert len(api.getCatIds()) == 1 and len(api.getImgIds()) == 9


def test_shipped_config_trains_on_seven_games_and_keeps_0108_out():
    cfg = load_config(ROOT / "src/configs/detector/rfdetr_small.yaml")
    assert cfg.plugin == "detector"
    assert cfg.games.train == ["IMG_0104", "IMG_0105", "IMG_0106", "IMG_0107", "IMG_0109", "IMG_0110", "IMG_0111"]
    assert cfg.games.test == ["IMG_0108"]
    assert cfg.detector.resolution == 512 and cfg.detector.sample_every == 15
    assert np.isclose(cfg.detector.lr_encoder, 1.5e-4)
