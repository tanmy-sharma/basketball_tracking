from __future__ import annotations

import json
import sys

import pytest
import torch
import yaml
from pydantic import ValidationError

from src.core import constants
from src.core.checkpoint import load_checkpoint, save_checkpoint
from src.core.config import Config, PhaseCfg, RecognizerCfg, WeakCfg, config_hash, crop_hash, load_config, CropCfg
from src.core.plugin import PLUGINS, FitResult, ModelPlugin, get_plugin, register_plugin
from src.core.run import open_run, resolve_run_dir
from src.tests.fixtures import toy_config, toy_config_dict

PHASE = {"name": "p", "epochs": 1, "unfreeze_blocks": 0, "head_lr": 1e-3}


def cfg_with(**over) -> dict:
    d = toy_config_dict()
    d.update(over)
    return d


def write(p, d):
    p.write_text(yaml.safe_dump(d))
    return p


def test_constants():
    assert len(constants.NUMBERS) == 9 and constants.NUMBER_TO_IDX["12"] == 8
    assert {"strhub", "doctr", "mmcv", "ultralytics"} <= set(constants.FORBIDDEN_MODULES)
    assert constants.CHALLENGE_VIDEOS == {"IMG_0114", "IMG_0115"}


def test_base_merge_and_set(tmp_path):
    write(tmp_path / "base.yaml", cfg_with(seed=3, crops={"every": 10}))
    write(tmp_path / "child.yaml", {"_base": "base.yaml", "run_name": "child", "crops": {"pad_frac": 0.2}})
    c = load_config(tmp_path / "child.yaml", ["train.batch_size=4", "games.train=[IMG_0106, IMG_0107]"])
    assert (c.run_name, c.seed, c.crops.every, c.crops.pad_frac) == ("child", 3, 10, 0.2)
    assert c.train.batch_size == 4 and c.games.train == ["IMG_0106", "IMG_0107"]
    with pytest.raises(ValueError):
        load_config(tmp_path / "child.yaml", ["nokey"])


def test_base_cycle(tmp_path):
    write(tmp_path / "a.yaml", {"_base": "b.yaml"})
    write(tmp_path / "b.yaml", {"_base": "a.yaml"})
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("bad", [
    {"bogus": 1},
    {"aug": {"hflip": True}},
    {"aug": {"max_rotate_deg": 16}},
    {"games": {"train": ["IMG_0114"]}},
    {"games": {"train": ["IMG_0106"], "test": ["IMG_0115"]}},
    {"games": {"train": ["IMG_0106"], "label_pool": ["IMG_0107"], "test": ["IMG_0107"]}},
    {"games": {"train": ["IMG_0106", "IMG_0108"], "test": ["IMG_0108"]}},
    {"games": {"train": ["video1"]}},
    {"plugin": "classifier"},
    {"plugin": "detector"},
    {"plugin": "recognizer"},
    {"model": {"head": "eligibility"}},
])
def test_rejects(bad):
    with pytest.raises(ValidationError):
        Config.model_validate(cfg_with(**bad))


def test_valid_plugin_configs():
    ok = cfg_with(plugin="detector", detector={})
    assert Config.model_validate(ok).detector.resolution == 512
    ok = cfg_with(plugin="recognizer", recognizer={"config_overrides": {"depth": 1}})
    assert Config.model_validate(ok).recognizer.seqlen == 4
    ok = cfg_with(model={"head": "eligibility"}, data={"label_csv": "x.csv"})
    assert Config.model_validate(ok).model.head == "eligibility"


def test_recognizer_defaults():
    r = RecognizerCfg()
    assert r.input_size == 224 and r.seqlen == 4 and r.resize == "letterbox" and r.pretrained
    assert r.sha256 == "3ff5c0490dabcb99b8540e66feb61d60e57821b55dabaacecbf31920824e42aa"
    assert r.url.endswith("vitstr_small_patch16_224_aug.pth")
    with pytest.raises(ValidationError):
        RecognizerCfg(resize="crop")


def test_weak_folds():
    ph = [PHASE]
    w = WeakCfg(name="w", folds=[["IMG_0104", "IMG_0109"], ["IMG_0105"]], fold_phases=ph)
    assert w.fold_plugin == "recognizer"
    assert WeakCfg(name="w", folds=w.folds, fold_plugin="classifier", fold_phases=ph).fold_plugin == "classifier"
    with pytest.raises(ValidationError):
        WeakCfg(name="w", pool=["IMG_0106"], fold_phases=ph)
    with pytest.raises(ValidationError):
        WeakCfg(name="w", folds=[["IMG_0104"], ["IMG_0104"]], fold_phases=ph)
    with pytest.raises(ValidationError):
        WeakCfg(name="w", folds=[["IMG_0104"], ["IMG_0114"]], fold_phases=ph)
    with pytest.raises(ValidationError):
        WeakCfg(name="w", folds=[["IMG_0104"]], fold_phases=ph)


def test_config_is_frozen():
    c = toy_config()
    with pytest.raises(ValidationError):
        c.seed = 5


def test_hash_stable_and_sensitive(tmp_path):
    a, b = toy_config(), toy_config()
    assert config_hash(a) == config_hash(b) and len(config_hash(a)) == 8
    assert config_hash(Config.model_validate(cfg_with(device="cuda"))) == config_hash(a)
    nw = cfg_with()
    nw["train"]["num_workers"] = 2
    assert config_hash(Config.model_validate(nw)) == config_hash(a)
    assert config_hash(Config.model_validate(cfg_with(seed=1))) != config_hash(a)
    csv = tmp_path / "inc.csv"
    csv.write_text("crop_id\nx\n")
    c = Config.model_validate(cfg_with(filters={"include_csv": str(csv)}))
    h1 = config_hash(c)
    csv.write_text("crop_id\ny\n")
    assert config_hash(c) != h1
    assert crop_hash(CropCfg()) == crop_hash(CropCfg()) != crop_hash(CropCfg(every=10))


def test_run_dir_manifest_and_resume(tmp_path):
    cfg = toy_config()
    run = open_run(cfg, ["x"])
    assert run.dir == resolve_run_dir(cfg) and run.dir.name == f"toy__{run.hash}"
    m = json.loads((run.dir / "manifest.json").read_text())
    for k in ("git_hash", "dirty", "bundle_sha256", "seed", "argv", "config", "config_hash", "versions",
              "weight_sources", "input_files", "frame_counts", "started"):
        assert k in m
    assert json.loads((run.dir / "config.json").read_text())["run_name"] == "toy"
    run.log_metrics({"epoch": 1, "a": 1.0})
    run.log_metrics({"epoch": 2, "a": 2.0})
    assert (run.dir / "metrics.csv").read_text().count("\n") == 3
    assert not run.is_done
    run.finish(FitResult(run.dir / "best.pt", "m", 0.5, {"k": 1}))
    assert run.is_done and not list(run.dir.glob("*.tmp"))
    again = open_run(cfg, ["y"])
    assert again.is_done and again.manifest()["argv"] == ["y"] and "resumed" in again.manifest()
    r = again.load_result()
    assert (r.metric_name, r.metric, r.extra) == ("m", 0.5, {"k": 1})


def test_checkpoint_roundtrip(tmp_path):
    p = tmp_path / "sub" / "best.pt"
    payload = {"format_version": 1, "state_dict": {"w": torch.ones(2)}, "classes": ["a"], "epoch": 3}
    save_checkpoint(p, payload)
    assert not list(p.parent.glob("*.tmp"))
    got = load_checkpoint(p)
    assert got["epoch"] == 3 and torch.equal(got["state_dict"]["w"], torch.ones(2))


def test_checkpoint_atomic_on_failure(tmp_path):
    p = tmp_path / "best.pt"
    save_checkpoint(p, {"v": 1})
    with pytest.raises(Exception):
        save_checkpoint(p, {"v": lambda: 0})  # not picklable
    assert load_checkpoint(p) == {"v": 1}


def test_plugin_registry():
    assert PLUGINS["recognizer"] == "src.recognizer.plugin:RecognizerPlugin"
    assert get_plugin("src.tests.fixtures:ToyPlugin").__name__ == "ToyPlugin"
    with pytest.raises(KeyError):
        get_plugin("nope")

    class Mine(ModelPlugin):
        def fit(self, run):
            raise NotImplementedError

    try:
        register_plugin("mine", Mine)
        assert get_plugin("mine") is Mine
    finally:
        PLUGINS.pop("mine", None)


def test_rfdetr_not_imported_by_core():
    import src.core.plugin  # noqa: F401
    assert "rfdetr" not in sys.modules
