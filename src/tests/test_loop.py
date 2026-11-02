from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from src.core.checkpoint import load_checkpoint
from src.core.config import Config
from src.core.paths import ROOT
from src.core.run import resolve_run_dir
from src.tests.fixtures import ToyPlugin, toy_config, toy_config_dict
from src.train import run_training


@pytest.fixture(autouse=True)
def _reset_crash():
    ToyPlugin.crash_at = None
    yield
    ToyPlugin.crash_at = None


def metrics(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(resolve_run_dir(cfg) / "metrics.csv")


def test_two_phases_and_artifacts():
    cfg = toy_config(patience=99)
    res = run_training(cfg, [])
    d = resolve_run_dir(cfg)
    m = metrics(cfg)
    assert list(m.epoch) == [1, 2, 3, 4, 5]
    assert list(m.phase) == ["head"] * 2 + ["ft"] * 3
    assert res.extra["epochs"] == 5 and not res.extra["stopped_early"]
    assert res.best_path == d / "best.pt" and res.metric_name == "val_balanced_acc"
    assert (d / "done.json").exists() and (d / "summary.json").exists()
    assert res.metric == pytest.approx(m.val_balanced_acc.max())
    best = load_checkpoint(d / "best.pt")  # weights_only=True
    assert best["format_version"] == 1 and best["plugin"] == cfg.plugin
    assert set(best) >= {"head", "classes", "backbone", "state_dict", "config", "epoch", "phase",
                         "metric_name", "metric"}
    last = load_checkpoint(d / "last.pt", full=True)
    assert last["phase_index"] == 1 and last["epoch_in_phase"] == 3 and "optimizer" in last


def test_frozen_head_phase_only_trains_last_layer():
    cfg = toy_config(patience=99, phases=[{"name": "head", "epochs": 1, "unfreeze_blocks": 0, "head_lr": 1e-2}])
    plugin = ToyPlugin(cfg)
    model = plugin.build_model()
    before = [p.clone() for p in model.parameters()]
    groups = plugin.configure_phase(model, cfg.train.phases[0])
    assert len(groups[0]["params"]) == 2
    assert [p.requires_grad for p in model.parameters()] == [False, False, True, True]
    assert all((a == b).all() for a, b in zip(before, model.parameters()))


def test_early_stop_only_in_unfrozen_phase():
    phases = [{"name": "head", "epochs": 3, "unfreeze_blocks": 0, "head_lr": 0.0},
              {"name": "ft", "epochs": 8, "unfreeze_blocks": 1, "head_lr": 0.0}]
    cfg = toy_config(run_name="es", phases=phases, patience=2)
    res = run_training(cfg, [])
    # lr 0 never improves: head phase runs fully (3), then 2 non-improving ft epochs stop the run
    assert res.extra["stopped_early"] and res.extra["epochs"] == 5
    assert list(metrics(cfg).phase) == ["head"] * 3 + ["ft"] * 2


def test_resume_after_crash_matches_uninterrupted():
    clean = toy_config(run_name="clean", patience=99)
    run_training(clean, [])
    cfg = toy_config(run_name="crashy", patience=99)
    ToyPlugin.crash_at = 3
    with pytest.raises(RuntimeError, match="simulated"):
        run_training(cfg, [])
    d = resolve_run_dir(cfg)
    assert (d / "last.pt").exists() and not (d / "done.json").exists()
    assert load_checkpoint(d / "last.pt", full=True)["epoch"] == 2
    res = run_training(cfg, [])
    assert res.extra["epochs"] == 5
    a, b = metrics(clean), metrics(cfg)
    assert list(b.epoch) == [1, 2, 3, 4, 5]
    assert b.val_loss.tolist() == pytest.approx(a.val_loss.tolist())
    assert "resuming" in (d / "log.txt").read_text()


def test_done_run_is_skipped():
    cfg = toy_config(patience=99)
    first = run_training(cfg, [])
    n = len(metrics(cfg))
    again = run_training(cfg, [])
    assert again.metric == first.metric and len(metrics(cfg)) == n
    assert "already done" in (resolve_run_dir(cfg) / "log.txt").read_text()


def test_missing_monitor_fails_fast():
    d = toy_config_dict()
    d["train"]["monitor"] = "nope"
    with pytest.raises(KeyError, match="nope"):
        run_training(Config.model_validate(d), [])


def test_manifest_gets_weight_sources():
    cfg = toy_config(patience=99)
    run_training(cfg, [])
    m = json.loads((resolve_run_dir(cfg) / "manifest.json").read_text())
    assert m["weight_sources"] == [] and "finished" in m


def test_cli_end_to_end(tmp_path):
    p = tmp_path / "toy.yaml"
    p.write_text(yaml.safe_dump(toy_config_dict(run_name="cli")))
    env = {**os.environ, "SRC_OUTPUTS": str(tmp_path / "out")}
    r = subprocess.run([sys.executable, "-m", "src.train", "--config", str(p), "--set", "seed=1",
                        "train.patience=9"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    runs = list((tmp_path / "out" / "runs").glob("cli__*"))
    assert len(runs) == 1 and (runs[0] / "done.json").exists()
