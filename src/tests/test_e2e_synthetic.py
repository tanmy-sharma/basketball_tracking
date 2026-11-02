"""Tier-1 end to end on synthetic data: extract -> weak folds -> eligibility -> finalize -> recogniser -> evaluate.

Three training folds of one game each plus one test game, a fake pose model and tiny random networks.
No GPU, downloads or real video. A rerun of every stage must skip all the work.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.core.config import load_config
from src.core.paths import ROOT, outputs_dir
from src.core.run import resolve_run_dir
from src.crops import extract
from src.dataio.store import crops_root, load_manifest
from src.evaluate import classifier as eval_cli
from src.recognizer import predict as rec_predict
from src.tests.fixtures import make_training_tree, make_video, patch_trackid_paths, tiny_dino_overrides
from src.train import main as train_main
from src.weak import finalize, folds, qa

TRAIN = ["IMG_0104", "IMG_0105", "IMG_0106"]
TEST = "IMG_0108"
FOLDS = [[g] for g in TRAIN]
FRAMES = 60
TINY_REC = {"input_size": 32, "pretrained": False,
            "config_overrides": {"img_size": 32, "patch_size": 16, "embed_dim": 48, "depth": 4, "num_heads": 2}}
TINY_DINO = {"name": "dino", "pretrained": False, "config_overrides": tiny_dino_overrides()}
PHASES = [{"name": "head", "epochs": 1, "unfreeze_blocks": 0, "head_lr": 1e-2},
          {"name": "ft", "epochs": 2, "unfreeze_blocks": 2, "head_lr": 1e-2, "backbone_lr": 1e-3, "llrd": .9}]


class FakePose:
    """Torso keypoints at fixed fractions of every box."""

    def __call__(self, frame_rgb, boxes_xywh):
        out = np.zeros((len(boxes_xywh), 17, 3), np.float32)
        for i, (x, y, w, h) in enumerate(boxes_xywh):
            for k, (fx, fy) in {5: (.75, .25), 6: (.25, .25), 11: (.7, .55), 12: (.3, .55)}.items():
                out[i, k] = (x + fx * w, y + fy * h, .9)
        return out


def write_cfg(path: Path, name: str, **fields) -> Path:
    d = {"run_name": name, "device": "cpu", "seed": 0, "games": {"train": TRAIN, "test": [TEST]},
         "split": {"val_start_frac": .8, "guard_frames": 0},
         "crops": {"every": 2, "min_side_px": 4},
         "aug": {"max_rotate_deg": 3, "grey_p": 0.0, "hue_deg": 5, "subcrop_frac": .04},
         "train": {"batch_size": 32, "num_workers": 0, "patience": 20, "phases": PHASES}}
    d.update(fields)
    path.write_text(yaml.safe_dump(d))
    return path


def snapshot(root: Path) -> dict[str, int]:
    """mtime of every file that a retrain or re-extraction would rewrite."""
    keep = ("best.pt", "last.pt", "done.json", "metrics.csv", "summary.json", "crops.bin", "manifest.csv")
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in sorted(root.rglob("*"))
            if p.is_file() and (p.name in keep or "_fold_cache" in p.parts)}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    games = TRAIN + [TEST]
    root = make_training_tree(tmp_path / "Training", games, n_frames=FRAMES)
    patch_trackid_paths(monkeypatch, root)
    for g in games:
        make_video(root / "vidoes" / f"{g}.MOV", FRAMES, w=640, h=360)
    return root


def run_pipeline(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    out = outputs_dir()
    weak_dir = out / "weak" / "e2e"
    rec = {"recognizer": TINY_REC}
    cfgs = {
        "weak": write_cfg(tmp_path / "weak.yaml", "weak_e2e", plugin="classifier", model={"backbone": TINY_DINO, "input_size": 56},
                          **rec, weak={"name": "e2e", "folds": FOLDS, "fold_plugin": "recognizer", "tau_pos": 0.0,
                                       "tau_neg": 0.12, "head_threshold": 0.0, "fill_min_per_class": 10,
                                       "fold_subsample_every": 2, "fold_phases": PHASES}),
        "elig": write_cfg(tmp_path / "elig.yaml", "eligibility", plugin="classifier", data={"label_csv": str(weak_dir / "eligibility_labels.csv")},
                          model={"backbone": TINY_DINO, "head": "eligibility", "head_hidden": 32, "input_size": 56, "dropout": 0.0}),
        "all": write_cfg(tmp_path / "all.yaml", "vitstr_all", plugin="recognizer", **rec),
        "eligible": write_cfg(tmp_path / "eligible.yaml", "vitstr_eligible", plugin="recognizer", **rec,
                              filters={"include_csv": str(weak_dir / "eligible_final.csv")}),
    }
    # 1. crops
    cfg = load_config(cfgs["all"])
    for g in TRAIN + [TEST]:
        extract.extract_game(g, cfg.crops, FakePose())
    # 2. weak folds (fold models are tiny random ViTSTRs), then QA
    folds.main(["--config", str(cfgs["weak"]), "--device", "cpu"])
    qa.main(["--config", str(cfgs["weak"])])
    # 3. eligibility head on the weak labels, then finalize
    train_main(["--config", str(cfgs["elig"])])
    finalize.main(["--config", str(cfgs["weak"]), "--eligibility", str(cfgs["elig"]), "--device", "cpu"])
    # 4. recogniser on all crops and on the eligible ones, free reads, evaluation on the test game
    for k in ("all", "eligible"):
        train_main(["--config", str(cfgs[k])])
    reads = out / "reads"
    rec_predict.main(["--config", str(cfgs["all"]), "--best", str(resolve_run_dir(load_config(cfgs["all"])) / "best.pt"),
                      "--game", TEST, "--out", str(reads), "--mode", "free", "--device", "cpu"])
    free_csv = next(reads.glob("free_*.csv"))
    eval_cli.main(["--game", TEST, "--number", f"all={cfgs['all']}", "--number", f"eligible={cfgs['eligible']}",
                   "--eligibility", str(cfgs["elig"]), "--free-reads", f"all={free_csv}", "--out", str(out / "eval"),
                   "--device", "cpu"])
    return cfgs


def test_pipeline_end_to_end_and_rerun_skips(tree, tmp_path, monkeypatch):
    cfgs = run_pipeline(tmp_path, monkeypatch)
    out = outputs_dir()

    for g in TRAIN + [TEST]:
        d = crops_root(load_config(cfgs["all"]).crops) / g
        assert (d / "done.json").exists() and (d / "keypoints.npy").exists() and (d / "meta.json").exists()
    weak = out / "weak" / "e2e"
    wl = pd.read_csv(weak / "weak_labels.csv")
    assert set(wl.video) == set(TRAIN) and wl.fold.nunique() == 3
    assert {1, 0} <= set(wl.label) and (wl.label == -2).any(), "need eligible, ineligible and uncovered crops"
    assert {"eligibility_labels.csv", "coverage.md", "qa.md", "eligible_final.csv"} <= {p.name for p in weak.iterdir()}
    assert list((weak / "sheets").glob("*.png"))
    assert "WARNING" in (weak / "coverage.md").read_text()
    fin = pd.read_csv(weak / "eligible_final.csv")
    assert {"weak", "head"} <= set(fin.source) and fin.crop_id.is_unique

    runs = {k: resolve_run_dir(load_config(cfgs[k])) for k in ("elig", "all", "eligible")}
    folds_dirs = sorted((out / "runs").glob("fold_e2e_*"))
    assert len(folds_dirs) == 3
    for d in [*runs.values(), *folds_dirs]:
        assert (d / "done.json").exists() and (d / "best.pt").exists() and (d / "config.json").exists()
        m = json.loads((d / "manifest.json").read_text())
        assert m["config_hash"] == d.name.split("__")[1] and m["finished"] and "versions" in m
        assert m["bundle_sha256"] is None and "git_hash" in m
    assert json.loads((runs["all"] / "manifest.json").read_text())["weight_sources"] == []  # tiny random models: no pretrained weights
    assert load_manifest(crops_root(load_config(cfgs["all"]).crops), [TEST]).shape[0] > 0

    ev = out / "eval"
    metrics = json.loads((ev / "metrics.json").read_text())
    assert set(metrics["models"]) == {"all", "eligible"} and metrics["game"] == TEST
    for m in metrics["models"].values():
        for sub in ("all", "clean"):
            assert {"acc_9way", "acc_team", "per_track_team", "confusion", "votes", "coverage", "by_bib",
                    "elig_auc_for_correct", "eligible_only"} <= set(m[sub])
    assert "exact_rate" in metrics["models"]["all"]["free_reads"] and "free_reads" not in metrics["models"]["eligible"]
    assert (ev / "report.md").exists() and (ev / "predictions_all.csv.gz").exists() and (ev / "confusion_eligible.csv").exists()

    before = snapshot(out)
    assert any(k.endswith("best.pt") for k in before) and any("_fold_cache" in k for k in before)
    run_pipeline_again(cfgs)
    assert snapshot(out) == before, "a rerun changed or recreated finished work"


def run_pipeline_again(cfgs: dict[str, Path]) -> None:
    """Every entry point again; none may redo training, extraction or fold prediction."""
    cfg = load_config(cfgs["all"])
    argv = ["extract", "--config", str(cfgs["all"]), "--games", *TRAIN, TEST]
    old = sys.argv
    sys.argv = argv
    try:
        extract.main()
    finally:
        sys.argv = old
    folds.main(["--config", str(cfgs["weak"]), "--device", "cpu"])
    train_main(["--config", str(cfgs["elig"])])
    finalize.main(["--config", str(cfgs["weak"]), "--eligibility", str(cfgs["elig"]), "--device", "cpu"])
    for k in ("all", "eligible"):
        train_main(["--config", str(cfgs[k])])
    assert (crops_root(cfg.crops) / TEST / "done.json").exists()


def load_builder():
    spec = importlib.util.spec_from_file_location("build_src_notebooks", ROOT / "colab" / "build_src_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stock_baseline_script_scores_the_test_game(tree, tmp_path, monkeypatch):
    """The notebook's inline stock-ViTSTR scorer, fed by the real stock predictor (tiny random weights stand in)."""
    cfg_path = write_cfg(tmp_path / "all.yaml", "vitstr_all", plugin="recognizer", recognizer=TINY_REC)
    cfg = load_config(cfg_path)
    for g in TRAIN + [TEST]:
        extract.extract_game(g, cfg.crops, FakePose())
    d = outputs_dir() / "stock"
    rec_predict.main(["--config", str(cfg_path), "--stock", "--game", TEST, "--out", str(d), "--mode", "both", "--device", "cpu"])
    monkeypatch.setattr(sys, "argv", ["-c", str(cfg_path), str(d)])
    exec(compile(load_builder().STOCK_EVAL, "stock_eval", "exec"), {"__name__": "__main__"})
    res = json.loads((d / "stock_metrics.json").read_text())
    assert res["n"] > 0 and 0 <= res["scores"]["acc_team"] <= 1 and "exact_rate" in res["free_reads"]
