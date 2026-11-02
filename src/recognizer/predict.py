"""Inference: roster scoring (N, 9), free decoding, stock (no fine-tune) baseline, and a CLI.

python -m src.recognizer.predict --config CFG (--best best.pt | --stock) --game IMG_0108 --out DIR [--mode free|score|both]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.core.config import Config, RecognizerCfg, load_config
from src.core.loop import pick_device
from src.dataio.store import CropStore, crops_root, load_manifest
from src.recognizer.model import ViTSTR, build_vitstr
from src.recognizer.scoring import score_numbers
from src.recognizer.tokens import decode_free
from src.recognizer.transforms import eval_transform


def load_model(best_pt: Path | None, rec: RecognizerCfg, device: str | torch.device = "cpu") -> tuple[ViTSTR, RecognizerCfg]:
    """Fine-tuned model from best.pt (config taken from the checkpoint), or the stock weights with remapped head."""
    if best_pt is None:
        model = build_vitstr(rec)
    else:
        ck = torch.load(best_pt, map_location="cpu", weights_only=True)
        rec = Config.model_validate(ck["config"]).recognizer
        rec = rec.model_copy(update={"pretrained": False})
        model = build_vitstr(rec)
        model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), rec


@torch.no_grad()
def _logits(model: ViTSTR, rec: RecognizerCfg, df: pd.DataFrame, store: CropStore, device, batch_size: int):
    tf = eval_transform(rec)
    dev = torch.device(device)
    for i in range(0, len(df), batch_size):
        rows = df.iloc[i:i + batch_size]
        x = np.stack([tf(store.read(r.video, int(r.offset), int(r.crop_w), int(r.crop_h)), None)
                      for r in rows.itertuples()])
        yield model(torch.from_numpy(x).to(dev)).float().cpu()


def _run(best_pt, df, store, device, batch_size, rec=None):
    model, rec = load_model(best_pt, rec or RecognizerCfg(), device)
    return list(_logits(model, rec, df, store, device, batch_size))


def predict_crops(best_pt: Path, df: pd.DataFrame, store: CropStore, device: str, batch_size: int = 256) -> np.ndarray:
    """(N, 9) probabilities in NUMBERS order, in df row order (log-prob sum per number, softmax across 9)."""
    out = _run(best_pt, df, store, device, batch_size)
    return torch.cat([score_numbers(lg).softmax(-1) for lg in out]).numpy() if out else np.zeros((0, 9), np.float32)


def predict_free(best_pt: Path | None, df: pd.DataFrame, store: CropStore, device: str, batch_size: int = 256,
                 rec: RecognizerCfg | None = None) -> list[str]:
    """Unconstrained greedy read of each crop. `best_pt=None` is the stock baseline (needs `rec`)."""
    out: list[str] = []
    for lg in _run(best_pt, df, store, device, batch_size, rec):
        out += decode_free(lg.argmax(-1).numpy())
    return out


def predict_stock(df: pd.DataFrame, store: CropStore, device: str, rec: RecognizerCfg | None = None,
                  batch_size: int = 256) -> tuple[np.ndarray, list[str]]:
    """Untrained baseline: pretrained weights, head remapped to digits. Returns ((N, 9) probs, free text)."""
    out = _run(None, df, store, device, batch_size, rec)
    probs = torch.cat([score_numbers(lg).softmax(-1) for lg in out]).numpy()
    free = [t for lg in out for t in decode_free(lg.argmax(-1).numpy())]
    return probs, free


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="supplies crops and recognizer settings")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--best", type=Path)
    g.add_argument("--stock", action="store_true", help="pretrained weights, no fine-tuning")
    ap.add_argument("--game", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["free", "score", "both"], default="both")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    device = str(pick_device(a.device))
    df = load_manifest(crops_root(cfg.crops), a.game)
    df = df[(df.status == "ok") & (df.offset >= 0)].reset_index(drop=True)
    store = CropStore(crops_root(cfg.crops))
    model, rec = load_model(None if a.stock else a.best, cfg.recognizer or RecognizerCfg(), device)
    out = list(_logits(model, rec, df, store, device, a.batch_size))
    a.out.mkdir(parents=True, exist_ok=True)
    tag = "stock" if a.stock else a.best.parent.name
    if a.mode in ("free", "both"):
        text = [t for lg in out for t in decode_free(lg.argmax(-1).numpy())]
        pd.DataFrame({"crop_id": df.crop_id, "text": text}).to_csv(a.out / f"free_{tag}.csv", index=False)
    if a.mode in ("score", "both"):
        probs = torch.cat([score_numbers(lg).softmax(-1) for lg in out]).numpy()
        np.save(a.out / f"probs_{tag}.npy", probs)
        df[["crop_id"]].to_csv(a.out / f"probs_{tag}_ids.csv", index=False)
    print(f"{len(df)} crops -> {a.out}")


if __name__ == "__main__":
    main()
