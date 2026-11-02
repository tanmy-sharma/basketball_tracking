"""Weak eligibility labels from 3 fixed game folds.  `python -m src.weak.folds --config weak_folds3.yaml`

Each fold's model trains on the other groups and reads the held-out group, so a crop's label says whether a
model that never saw that game read the number correctly. Fold runs are resumable and cached.
"""
from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import Config, RecognizerCfg, load_config
from src.core.paths import outputs_dir
from src.core.run import resolve_run_dir
from src.dataio.select import training_rows
from src.dataio.store import CropStore, crops_root, load_manifest
from src.weak.assemble import assemble_labels, coverage_markdown, eligibility_labels


def predict_fn(plugin: str) -> Callable[..., np.ndarray]:
    """Lazy `predict_crops` of the `classifier` or `recognizer` package."""
    if plugin not in ("classifier", "recognizer"):
        raise ValueError(f"no predict_crops for plugin {plugin!r}")
    return importlib.import_module(f"src.{plugin}.predict").predict_crops


def fold_config(cfg: Config, held_out: list[str]) -> Config:
    """Fold config: the whole held-out group moves to `games.test`; the rest of the folds train."""
    weak = cfg.weak
    if weak is None:
        raise ValueError("config has no `weak` block")
    if sorted(held_out) not in [sorted(f) for f in weak.folds]:
        raise ValueError(f"{held_out} is not one of the configured folds {weak.folds}")
    train = [g for f in weak.folds for g in f if g not in held_out]
    d = cfg.model_dump(mode="json")
    d["games"] = {"train": train, "test": list(cfg.games.test) + list(held_out), "label_pool": []}
    d["plugin"] = weak.fold_plugin
    d["run_name"] = f"fold_{weak.name}_" + "_".join(g[-4:] for g in held_out)
    d["train"]["phases"] = [p.model_dump(mode="json") for p in weak.fold_phases]
    d["filters"].update(subsample_every=weak.fold_subsample_every, include_csv=None)
    d["data"] = {"label_csv": None}
    d["weak"] = None
    if d.get("model"):
        d["model"]["head"] = "number"
    if weak.fold_plugin == "recognizer" and d.get("recognizer") is None:
        d["recognizer"] = RecognizerCfg().model_dump(mode="json")
    return Config.model_validate(d)


def run_fold(fcfg: Config, held: list[str], device: str) -> tuple[pd.DataFrame, np.ndarray, set[int], set[str]]:
    """Train (or reuse) one fold and score its held-out crops. Returns (rows, probs, covered, train_bibs)."""
    from src.train import run_training
    root = crops_root(fcfg.crops)
    train_rows = training_rows(load_manifest(root, fcfg.games.train), fcfg)
    man = load_manifest(root, held)
    man = man[(man.status == "ok") & (man.pose_min_conf >= fcfg.filters.min_pose_conf)
              & (man.offset >= 0)].reset_index(drop=True)
    cache = outputs_dir() / "weak" / "_fold_cache" / f"{resolve_run_dir(fcfg).name}.npz"
    result = run_training(fcfg, argv=["weak.folds"])
    if cache.exists() and (z := np.load(cache, allow_pickle=False))["crop_id"].tolist() == man.crop_id.tolist():
        probs = z["probs"]
    else:
        probs = predict_fn(fcfg.plugin)(result.best_path, man, CropStore(root), device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, crop_id=man.crop_id.to_numpy(dtype=str), probs=probs)
    return man, probs, {int(i) for i in train_rows.number_idx.unique()}, set(train_rows.bib)


def build_weak_labels(cfg: Config, device: str = "auto") -> pd.DataFrame:
    from src.core.loop import pick_device
    dev = str(pick_device(device))
    parts = []
    for held in cfg.weak.folds:
        fcfg = fold_config(cfg, held)
        man, probs, covered, bibs = run_fold(fcfg, held, dev)
        parts.append(assemble_labels(man, probs, covered, bibs, cfg.weak, cfg.filters.max_iou,
                                     fold="+".join(held)))
    return pd.concat(parts, ignore_index=True)


def write_outputs(weak_df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    weak_df.to_csv(out / "weak_labels.csv", index=False)
    eligibility_labels(weak_df).to_csv(out / "eligibility_labels.csv", index=False)
    (out / "coverage.md").write_text(coverage_markdown(weak_df))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="a.b=c")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    if cfg.weak is None:
        raise SystemExit("config has no `weak` block")
    out = outputs_dir() / "weak" / cfg.weak.name
    write_outputs(build_weak_labels(cfg, a.device), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
