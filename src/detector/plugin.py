"""RF-DETR Small fine-tune: wraps rfdetr's own trainer (imported lazily, it is an optional dependency)."""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import pandas as pd

from src.core.plugin import FitResult, ModelPlugin, WeightSource
from src.core.run import Run, file_sha256
from src.detector.coco import build_dataset

STOCK_URL = "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth"
METRIC_COLUMNS = ("val/mAP_50_95", "val/ema_mAP_50_95")  # rfdetr's best_total picks the better of the two


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def best_metric(metrics_csv: Path, run: Run) -> tuple[float, int | None]:
    """Best validation mAP over epochs, or (nan, None) with the column list logged if it is not there."""
    if not metrics_csv.exists():
        run.log(f"no {metrics_csv}; metric stored as NaN")
        return math.nan, None
    df = pd.read_csv(metrics_csv)
    cols = [c for c in METRIC_COLUMNS if c in df.columns and df[c].notna().any()]
    if not cols:
        run.log(f"none of {METRIC_COLUMNS} in metrics.csv; columns: {list(df.columns)}")
        return math.nan, None
    best = df[cols].max(axis=1)
    i = best.idxmax()
    epoch = int(df.loc[i, "epoch"]) if "epoch" in df.columns and pd.notna(df.loc[i, "epoch"]) else None
    return float(best.max()), epoch


class DetectorPlugin(ModelPlugin):
    def weight_sources(self) -> list[WeightSource]:
        return [WeightSource("rf-detr-small.pth", STOCK_URL, trained_on="COCO; Apache-2.0; md5 fb37061c1af7bace359c91b723a8d5c1")]

    def fit(self, run: Run) -> FitResult:
        cfg, d = self.cfg, self.cfg.detector
        ds = build_dataset(cfg)
        run.log(f"dataset: {ds}")
        out = run.dir / "rfdetr"
        out.mkdir(exist_ok=True)
        resume = out / "last.ckpt"
        from rfdetr import RFDETRSmall  # lazy: not installed in the shared venv
        model = RFDETRSmall()
        model.train(dataset_dir=str(ds), output_dir=str(out), epochs=d.epochs, batch_size=d.batch_size,
                    grad_accum_steps=d.grad_accum_steps, lr=d.lr, lr_encoder=d.lr_encoder,
                    resolution=d.resolution, seed=cfg.seed, num_workers=d.num_workers,
                    device=resolve_device(cfg.device), tensorboard=False, early_stopping=True,
                    early_stopping_patience=d.patience, checkpoint_interval=d.checkpoint_interval,
                    run_test=False, progress_bar=None, notes=run.hash,
                    resume=str(resume) if resume.exists() else None)
        src = out / "checkpoint_best_total.pth"
        if not src.exists():
            raise RuntimeError(f"training left no {src.name}; found {sorted(p.name for p in out.iterdir())}")
        best = run.dir / "best.pth"
        shutil.copyfile(src, best)
        metric, epoch = best_metric(out / "metrics.csv", run)
        info = (ds / "done.json").read_text() if (ds / "done.json").exists() else ""
        files = dict(run.manifest().get("input_files") or {})
        files["best.pth"] = file_sha256(best)
        run.update_manifest(input_files=files, detector_dataset=str(ds), detector_dataset_info=info)
        return FitResult(best, "val_mAP_50_95", metric, {"best_epoch": epoch, "dataset": str(ds)})
