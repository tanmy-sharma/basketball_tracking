from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.classifier.dataset import CropDataset
from src.classifier.predict import predict_crops
from src.classifier.transforms import CropTransform
from src.core.checkpoint import load_checkpoint
from src.core.config import Config, load_config
from src.core.constants import NUMBERS
from src.core.paths import ROOT
from src.dataio.store import CropStore, crops_root
from src.tests.fixtures import make_crop_store, tiny_dino_overrides
from src.train import run_training

CONFIG_DIR = ROOT / "src" / "configs" / "classifier"
ALL_CONFIGS = ["base", "number_all", "eligibility", "number_eligible", "number_eligible_resnet34",
               "xcolor_no_yellow", "xcolor_no_green", "number_submit"]


def small_cfg(tmp_path: Path, head: str = "number", backbone: str = "dino", epochs: tuple[int, int] = (2, 6),
              **extra) -> Config:
    bb: dict = {"name": backbone, "pretrained": False}
    if backbone == "dino":
        bb["config_overrides"] = tiny_dino_overrides()
    d = {"run_name": f"t_{head}_{backbone}", "plugin": "classifier", "device": "cpu", "seed": 0,
         "games": {"train": ["IMG_0106"], "test": ["IMG_0108"]},
         "split": {"guard_frames": 0},
         "model": {"backbone": bb, "head": head, "head_hidden": 64, "input_size": 56, "dropout": 0.0},
         "aug": {"max_rotate_deg": 3, "grey_p": 0.0, "hue_deg": 5, "subcrop_frac": 0.04},
         "train": {"batch_size": 32, "num_workers": 0, "patience": 20,
                   "phases": [{"name": "head", "epochs": epochs[0], "unfreeze_blocks": 0, "head_lr": 3e-3},
                              {"name": "ft", "epochs": epochs[1], "unfreeze_blocks": 4, "head_lr": 3e-3,
                               "backbone_lr": 1e-3, "llrd": 0.9}]}}
    d.update(extra)
    return Config.model_validate(d)


def build_store(cfg: Config, n: int = 450, numbers: list[str] | None = None) -> pd.DataFrame:
    return make_crop_store(crops_root(cfg.crops), "IMG_0106", n, numbers or list(NUMBERS),
                           np.random.default_rng(0))


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_shipped_configs_load(name):
    cfg = load_config(CONFIG_DIR / f"{name}.yaml")
    assert cfg.plugin == "classifier"
    if name != "number_submit":
        assert cfg.games.test == ["IMG_0108"] and "IMG_0108" not in cfg.games.train
        assert len(cfg.games.train) == 7
    else:
        assert cfg.games.test == [] and len(cfg.games.train) == 8


def test_config_differences():
    base = load_config(CONFIG_DIR / "number_all.yaml")
    assert load_config(CONFIG_DIR / "eligibility.yaml").model.head == "eligibility"
    assert load_config(CONFIG_DIR / "number_eligible.yaml").filters.include_csv.endswith("eligible_final.csv")
    r = load_config(CONFIG_DIR / "number_eligible_resnet34.yaml")
    assert r.model.backbone.name == "resnet34" and r.train.phases[-1].unfreeze_blocks == 2
    assert r.model.backbone.model_copy(update={"name": "dino"}) == base.model.backbone
    assert load_config(CONFIG_DIR / "xcolor_no_yellow.yaml").filters.exclude[0].bib == "yellow"
    assert load_config(CONFIG_DIR / "xcolor_no_green.yaml").filters.exclude[0].numbers == ["6", "9"]


def test_number_training_beats_chance_and_predict_preserves_order(tmp_path):
    cfg = small_cfg(tmp_path, epochs=(2, 30))
    df = build_store(cfg, n=900)
    t0 = time.time()
    res = run_training(cfg, [])
    assert time.time() - t0 < 60
    assert res.metric_name == "val_balanced_acc" and res.metric > 2 / 9
    ck = load_checkpoint(res.best_path)
    assert ck["head"] == "number" and ck["classes"] == list(NUMBERS) and ck["backbone"] == "dino"

    store = CropStore(crops_root(cfg.crops))
    sub = df.iloc[::-1].iloc[:50].reset_index(drop=True)  # reordered subset
    p = predict_crops(res.best_path, sub, store, "cpu", batch_size=16)
    assert p.shape == (50, 9) and np.allclose(p.sum(1), 1, atol=1e-5)
    p_all = predict_crops(res.best_path, df, store, "cpu", batch_size=64)
    np.testing.assert_allclose(p, p_all[::-1][:50], atol=1e-4)
    assert (p_all.argmax(1) == df.number_idx.to_numpy()).mean() > 2 / 9


def test_val_metrics_reported(tmp_path):
    cfg = small_cfg(tmp_path, epochs=(1, 0))
    build_store(cfg, n=270)
    run_training(cfg, [])
    from src.core.run import resolve_run_dir
    m = pd.read_csv(resolve_run_dir(cfg) / "metrics.csv")
    assert {"val_loss", "val_acc", "val_balanced_acc", "val_within_team_acc"} <= set(m.columns)
    assert (m.val_within_team_acc >= m.val_acc - 1e-9).all()


def test_eligibility_head_from_label_csv(tmp_path):
    cfg0 = small_cfg(tmp_path)
    df = build_store(cfg0, n=360, numbers=["1", "3", "5"])
    labels = pd.DataFrame({"crop_id": df.crop_id, "label": (df.number_idx == df.number_idx.min()).astype(int)})
    labels.to_csv(tmp_path / "labels.csv", index=False)
    cfg = small_cfg(tmp_path, head="eligibility", data={"label_csv": str(tmp_path / "labels.csv")})
    res = run_training(cfg, [])
    ck = load_checkpoint(res.best_path)
    assert ck["head"] == "eligibility" and ck["classes"] == ["ineligible", "eligible"]
    p = predict_crops(res.best_path, df, CropStore(crops_root(cfg.crops)), "cpu")
    assert p.shape == (len(df), 2)
    assert res.metric > 0.5


def test_resnet34_runs_through_same_plugin(tmp_path):
    cfg = small_cfg(tmp_path, backbone="resnet34", epochs=(1, 1))
    build_store(cfg, n=180)
    res = run_training(cfg, [])
    assert load_checkpoint(res.best_path)["backbone"] == "resnet34"


def test_dataset_label_override_and_inputs(tmp_path):
    cfg = small_cfg(tmp_path)
    df = build_store(cfg, n=18)

    class Sub(CropDataset):
        def label(self, i):
            return torch.tensor([int(self.labels[i]), 99])

    ds = Sub(df, CropStore(crops_root(cfg.crops)), CropTransform(56), "number_idx")
    x, y = ds[3]
    assert x.shape == (3, 56, 56) and y.tolist() == [int(df.number_idx[3]), 99]
    assert CropDataset(df, CropStore(crops_root(cfg.crops)), CropTransform(56), None)[0][1] == -1
