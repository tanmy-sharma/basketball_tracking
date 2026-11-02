from __future__ import annotations

import pickle
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.core.config import Config, SplitCfg
from src.core.constants import NUMBERS
from src.dataio.schema import CROP_COLUMNS, crop_id, validate_manifest
from src.dataio.select import training_rows
from src.dataio.split import assign_split
from src.dataio.store import CropStore, CropWriter, crops_root, load_manifest, save_manifest
from src.core.config import CropCfg
from src.tests.fixtures import (frame_index, make_crop_store, make_training_tree, make_video,
                                patch_trackid_paths, tiny_dino_overrides, toy_config_dict)

SIX = ["6", "9", "11", "1", "3", "5"]


def cfg(**over) -> Config:
    d = toy_config_dict()
    d["split"] = {"val_start_frac": 0.8, "guard_frames": 20}
    d.update(over)
    return Config.model_validate(d)


@pytest.fixture
def df(tmp_path) -> pd.DataFrame:
    root = tmp_path / "crops"
    make_crop_store(root, "IMG_0106", 600, SIX, np.random.default_rng(0))  # frames 1..496
    return load_manifest(root, ["IMG_0106"])


def test_writer_store_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for h, w in [(5, 3), (7, 4), (2, 9)]]
    with CropWriter(tmp_path / "V") as wr:
        offs = [wr.add(i) for i in imgs]
    assert offs == [0, 45, 129]
    store = pickle.loads(pickle.dumps(CropStore(tmp_path)))  # DataLoader workers pickle it
    for img, off in zip(imgs, offs):
        assert np.array_equal(store.read("V", off, img.shape[1], img.shape[0]), img)
    with pytest.raises(IndexError):
        store.read("V", 129, 50, 50)
    with pytest.raises(ValueError):
        CropWriter(tmp_path / "W").add(np.zeros((2, 2), np.uint8))


def test_manifest_roundtrip_and_validation(tmp_path, df):
    assert list(df.columns) == CROP_COLUMNS and len(df) == 600
    assert pd.api.types.is_string_dtype(df.jersey) and set(df.jersey) == set(SIX)
    assert df.in_game.dtype == bool and df.crop_id[0] == crop_id("IMG_0106", 1, 1)
    store = CropStore(tmp_path / "crops")
    r = df.iloc[7]
    assert store.read(r.video, r.offset, r.crop_w, r.crop_h).shape == (r.crop_h, r.crop_w, 3)
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "crops", ["IMG_0107"])
    bad = df.copy()
    bad.loc[0, "status"] = "weird"
    with pytest.raises(ValueError, match="status"):
        validate_manifest(bad)
    bad = df.copy()
    bad.loc[0, "frame"] = 999
    with pytest.raises(ValueError, match="crop_id"):
        validate_manifest(bad)
    with pytest.raises(ValueError, match="missing"):
        validate_manifest(df.drop(columns=["bib"]))
    d = tmp_path / "x"
    d.mkdir()
    save_manifest(d, df)
    assert not list(d.glob("*.tmp"))


def test_crops_root_follows_crop_cfg():
    assert crops_root(CropCfg()) == crops_root(CropCfg())
    assert crops_root(CropCfg()) != crops_root(CropCfg(every=10))


def test_split_block_and_guard(df):
    s = assign_split(df, SplitCfg(val_start_frac=0.8, val_end_frac=1.0, guard_frames=20))
    fmax = df.frame.max()
    lo = 0.8 * fmax
    assert (s[df.frame < lo - 20] == "train").all()
    assert (s[(df.frame >= lo - 20) & (df.frame < lo + 20)] == "guard").all()
    assert (s[df.frame >= lo + 20] == "val").all()
    assert set(s) == {"train", "guard", "val"}


def test_split_interior_block_guards_both_edges(df):
    s = assign_split(df, SplitCfg(val_start_frac=0.4, val_end_frac=0.6, guard_frames=10))
    fmax = df.frame.max()
    lo, hi = 0.4 * fmax, 0.6 * fmax
    assert (s[(df.frame >= lo + 10) & (df.frame < hi - 10)] == "val").all()
    assert (s[df.frame < lo - 10] == "train").all() and (s[df.frame >= hi + 10] == "train").all()
    assert (s[(df.frame >= hi - 10) & (df.frame < hi + 10)] == "guard").all()


def test_split_is_per_video(tmp_path):
    root = tmp_path / "c"
    rng = np.random.default_rng(0)
    make_crop_store(root, "IMG_0106", 60, SIX, rng)
    make_crop_store(root, "IMG_0107", 300, SIX, rng)
    d = load_manifest(root, ["IMG_0106", "IMG_0107"])
    s = assign_split(d, SplitCfg(guard_frames=0))
    for v, g in d.groupby("video"):
        assert (s[g.index][g.frame >= 0.8 * g.frame.max()] == "val").all()
    assert (s[d.video == "IMG_0106"] == "val").sum() < (s[d.video == "IMG_0107"] == "val").sum()


def test_training_rows_basic(df):
    out = training_rows(df, cfg())
    assert set(out.split) == {"train", "val"} and len(out) < len(df)
    assert out.index.equals(pd.RangeIndex(len(out)))
    assert out[out.split == "val"].frame.min() > out[out.split == "train"].frame.max()


def test_training_rows_refuses_test_and_foreign_games(tmp_path, df):
    make_crop_store(tmp_path / "crops", "IMG_0108", 60, SIX, np.random.default_rng(0))
    both = load_manifest(tmp_path / "crops", ["IMG_0106", "IMG_0108"])
    with pytest.raises(ValueError, match="test"):
        training_rows(both, cfg())
    with pytest.raises(ValueError, match="not in games.train"):
        training_rows(both, cfg(games={"train": ["IMG_0106"], "test": []}))
    with pytest.raises(ValueError, match="challenge"):
        training_rows(df.assign(video="IMG_0114"), cfg())


def test_training_rows_quality_filters(df):
    d = df.copy()
    d.loc[d.index[:50], "status"] = "low_pose"
    d.loc[d.index[50:100], "pose_min_conf"] = 0.2
    d.loc[d.index[100:150], "max_iou"] = 0.5
    out = training_rows(d, cfg())
    assert not out.crop_id.isin(d.crop_id[:150]).any()
    out = training_rows(d, cfg(filters={"max_iou": None}))
    assert out.crop_id.isin(d.crop_id[100:150]).any()


def test_training_rows_exclude(df):
    c = cfg(filters={"exclude": [{"bib": "green", "numbers": ["6", "9"]}]})
    out = training_rows(df, c)
    assert not out.jersey.isin(["6", "9"]).any() and out.jersey.isin(["11", "1"]).any()


def test_training_rows_include_csv(tmp_path, df):
    keep = df[df.jersey == "3"].crop_id
    p = tmp_path / "inc.csv"
    pd.DataFrame({"crop_id": keep}).to_csv(p, index=False)
    out = training_rows(df, cfg(filters={"include_csv": str(p)}))
    assert set(out.jersey) == {"3"} and out.crop_id.isin(keep).all()


def test_subsample_is_by_frame_not_row(df):
    out = training_rows(df, cfg(filters={"subsample_every": 2}))
    full = training_rows(df, cfg())
    frames = sorted(full.frame.unique())
    assert set(out.frame) <= set(frames[::2]) | set()
    # all tracks of a kept frame survive together
    assert (out.groupby("frame").size() == full[full.frame.isin(out.frame)].groupby("frame").size()).all()
    assert 0.4 < out.frame.nunique() / full.frame.nunique() < 0.6


def test_training_rows_raises_when_split_empty(df):
    with pytest.raises(ValueError, match="no val"):
        training_rows(df, cfg(split={"val_start_frac": 0.8, "val_end_frac": 0.8, "guard_frames": 0}))
    with pytest.raises(ValueError, match="no train"):
        training_rows(df, cfg(split={"val_start_frac": 0.0, "guard_frames": 0}))


# fixtures used by every later package ---------------------------------------------------------------

def test_make_training_tree_resolves_identities(tmp_path, monkeypatch):
    root = make_training_tree(tmp_path / "Training", ["IMG_0106", "IMG_0107", "IMG_0109"], n_frames=30,
                              stray_track=True, duplicate_row=True)
    patch_trackid_paths(monkeypatch, root)
    from trackid import io
    ros = io.load_roster(root / "games_roster.csv")
    assert set(ros) == {"IMG_0106", "IMG_0107", "IMG_0109"}
    ids = io.track_identities("IMG_0107", ros["IMG_0107"])
    assert ids[1] == ("A", "6") and ids[4] == ("B", "1") and len(ids) == 6
    bb = io.load_bboxes("IMG_0106")
    assert (bb.track_id == 0).sum() == 1 and bb.duplicated(["frame", "track_id"]).sum() == 1
    assert bb.frame.max() == 30
    gt = io.load_court_gt("IMG_0106")
    assert set(gt.jersey) == set(SIX) and pd.api.types.is_string_dtype(gt.jersey)


def test_make_crop_store_content(tmp_path):
    root = tmp_path / "c"
    d = make_crop_store(root, "IMG_0106", 18, ["6", "9", "11"], np.random.default_rng(0))
    assert (root / "IMG_0106" / "done.json").exists() and len(d) == 18
    assert d.number_idx.tolist()[:3] == [NUMBERS.index(n) for n in ("6", "9", "11")]
    assert (d.status == "ok").all() and d.frame.max() == 1 + 5 * 5
    a = CropStore(root).read("IMG_0106", int(d.offset[0]), int(d.crop_w[0]), int(d.crop_h[0]))
    b = CropStore(root).read("IMG_0106", int(d.offset[1]), int(d.crop_w[1]), int(d.crop_h[1]))
    assert a.std() > 5 and a.shape != () and a.mean() != b.mean()
    r1 = make_crop_store(tmp_path / "r1", "IMG_0106", 6, SIX, np.random.default_rng(3))
    r2 = make_crop_store(tmp_path / "r2", "IMG_0106", 6, SIX, np.random.default_rng(3))
    assert r1.equals(r2)


def test_make_video_frames_decode_in_order(tmp_path):
    p = make_video(tmp_path / "v", 300, w=32, h=18)
    assert p.suffix == ".mkv" and p.exists()
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 18, 32, 3)
    assert len(frames) == 300
    assert [frame_index(f) for f in frames[[0, 5, 255, 256, 299]]] == [0, 5, 255, 256, 299]


def test_tiny_dino_overrides_run():
    import torch
    from transformers import Dinov2Config, Dinov2Model
    o = tiny_dino_overrides()
    m = Dinov2Model(Dinov2Config(**o)).eval()
    with torch.no_grad():
        out = m(pixel_values=torch.zeros(2, 3, o["image_size"], o["image_size"]))
    assert out.last_hidden_state.shape[-1] == o["hidden_size"] and len(m.encoder.layer) == 4
