from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import pytest

from src.core.run import resolve_run_dir
from src.tests.test_coco import detector_cfg, make_world
from src.train import run_training

ALLOWED = {"dataset_dir", "output_dir", "epochs", "batch_size", "grad_accum_steps", "lr", "lr_encoder",
           "resolution", "seed", "num_workers", "device", "tensorboard", "early_stopping",
           "early_stopping_patience", "checkpoint_interval", "run_test", "progress_bar", "notes", "resume"}
CALLS: list[dict] = []


def install_fake_rfdetr(monkeypatch, metrics: str | None = "epoch,val/mAP_50_95,val/ema_mAP_50_95\n0,0.2,0.3\n1,0.5,0.45\n",
                        write_best: bool = True) -> None:
    class RFDETRSmall:
        def train(self, **kw):
            CALLS.append(kw)
            out = Path(kw["output_dir"])
            if write_best:
                (out / "checkpoint_best_total.pth").write_bytes(b"best-weights")
            if metrics is not None:
                (out / "metrics.csv").write_text(metrics)

    CALLS.clear()
    monkeypatch.setitem(sys.modules, "rfdetr", types.SimpleNamespace(RFDETRSmall=RFDETRSmall))


@pytest.fixture
def world(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch)
    return detector_cfg(epochs=3, patience=2, checkpoint_interval=5)


def test_module_import_does_not_import_rfdetr():
    import subprocess
    code = ("import sys, src.detector.plugin, src.detector.predict, src.detector.coco, src.detector.evaluate;"
            "assert 'rfdetr' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_fit_passes_only_allowed_kwargs_and_copies_best(world, monkeypatch):
    install_fake_rfdetr(monkeypatch)
    res = run_training(world, argv=["test"])
    kw = CALLS[0]
    assert set(kw) == ALLOWED
    assert (kw["epochs"], kw["batch_size"], kw["resolution"], kw["seed"]) == (3, 16, 512, 0)
    assert (kw["device"], kw["tensorboard"], kw["run_test"], kw["progress_bar"]) == ("cpu", False, False, None)
    assert kw["early_stopping"] is True and kw["early_stopping_patience"] == 2 and kw["checkpoint_interval"] == 5
    assert kw["resume"] is None and "num_classes" not in kw
    run_dir = resolve_run_dir(world)
    assert Path(kw["output_dir"]) == run_dir / "rfdetr" and kw["notes"] == run_dir.name.split("__")[1]
    assert (Path(kw["dataset_dir"]) / "train" / "_annotations.coco.json").exists()
    assert res.best_path == run_dir / "best.pth" and res.best_path.read_bytes() == b"best-weights"
    assert res.metric_name == "val_mAP_50_95" and res.metric == pytest.approx(0.5)
    assert res.extra["best_epoch"] == 1
    assert (run_dir / "done.json").exists()


def test_manifest_records_weights_dataset_and_best_hash(world, monkeypatch):
    install_fake_rfdetr(monkeypatch)
    run_training(world, argv=["test"])
    m = json.loads((resolve_run_dir(world) / "manifest.json").read_text())
    assert m["weight_sources"][0]["name"] == "rf-detr-small.pth"
    assert m["weight_sources"][0]["url"].endswith("small_coco/checkpoint_best_regular.pth")
    assert "fb37061c1af7bace359c91b723a8d5c1" in m["weight_sources"][0]["trained_on"]
    assert len(m["input_files"]["best.pth"]) == 64
    assert "detector_dataset" in m["detector_dataset"]


def test_resumes_from_last_ckpt_when_present(world, monkeypatch):
    install_fake_rfdetr(monkeypatch)
    out = resolve_run_dir(world) / "rfdetr"
    out.mkdir(parents=True)
    (out / "last.ckpt").write_bytes(b"x")
    run_training(world, argv=["test"])
    assert CALLS[0]["resume"] == str(out / "last.ckpt")


def test_missing_metric_columns_give_nan_not_failure(world, monkeypatch):
    install_fake_rfdetr(monkeypatch, metrics="epoch,val/loss\n0,1.0\n")
    res = run_training(world, argv=["test"])
    assert math.isnan(res.metric) and res.best_path.exists()
    assert "columns" in (resolve_run_dir(world) / "log.txt").read_text()


def test_missing_best_checkpoint_fails_without_done(world, monkeypatch):
    install_fake_rfdetr(monkeypatch, write_best=False)
    with pytest.raises(RuntimeError, match="checkpoint_best_total"):
        run_training(world, argv=["test"])
    assert not (resolve_run_dir(world) / "done.json").exists()


def test_finished_run_is_not_retrained(world, monkeypatch):
    install_fake_rfdetr(monkeypatch)
    run_training(world, argv=["test"])
    run_training(world, argv=["test"])
    assert len(CALLS) == 1


def test_kwargs_accepted_by_real_trainconfig(world, monkeypatch):
    pytest.importorskip("rfdetr.config")
    from rfdetr.config import TrainConfig
    install_fake_rfdetr(monkeypatch)
    run_training(world, argv=["test"])
    kw = {k: v for k, v in CALLS[0].items() if k not in ("device", "resolution")}
    TrainConfig(**kw)
