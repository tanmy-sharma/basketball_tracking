"""Contamination checks: eligibility rate by facing and by whether the fold saw the bib, plus contact sheets.

If `side` crops are eligible as often as `front` ones, or rates are far higher when the fold saw the bib,
the labels reflect person memory rather than digit reading.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.core.config import load_config
from src.core.constants import NUMBERS
from src.core.paths import outputs_dir
from src.dataio.store import CropStore, crops_root, load_manifest


def eligibility_rates(weak_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Share of label 1 among labelled (0/1) crops, by facing and by fold_saw_bib."""
    lab = weak_df[weak_df.label.isin([0, 1])]
    def rate(by: str) -> pd.DataFrame:
        g = lab.groupby(by).label
        return pd.DataFrame({"n": g.size(), "eligible_rate": g.mean()}).reset_index()
    return {"facing": rate("facing"), "fold_saw_bib": rate("fold_saw_bib")}


def contact_sheet(crops: list[np.ndarray], cell: tuple[int, int] = (48, 72), cols: int = 12) -> np.ndarray:
    w, h = cell
    rows = (len(crops) + cols - 1) // cols or 1
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        sheet[r * h:(r + 1) * h, k * w:(k + 1) * w] = cv2.resize(c, cell, interpolation=cv2.INTER_AREA)
    return sheet


def write_sheets(weak_df: pd.DataFrame, man: pd.DataFrame, store: CropStore, out: Path, per: int = 60,
                 seed: int = 0) -> list[Path]:
    """One PNG per (number, label in {1, 0}) with a random sample of crops."""
    out.mkdir(parents=True, exist_ok=True)
    mi = man.set_index("crop_id")
    paths = []
    for (idx, lab), g in weak_df[weak_df.label.isin([0, 1])].groupby(["number_idx", "label"]):
        pick = g.sample(min(per, len(g)), random_state=seed).crop_id
        imgs = [store.read(r.video, int(r.offset), int(r.crop_w), int(r.crop_h)) for r in mi.loc[pick].itertuples()]
        p = out / f"sheet_{NUMBERS[idx]}_{'eligible' if lab == 1 else 'ineligible'}.png"
        cv2.imwrite(str(p), cv2.cvtColor(contact_sheet(imgs), cv2.COLOR_RGB2BGR))
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = outputs_dir() / "weak" / cfg.weak.name
    weak_df = pd.read_csv(out / "weak_labels.csv", dtype={"crop_id": str, "video": str, "bib": str})
    lines = ["# Weak-label QA", ""]
    for k, t in eligibility_rates(weak_df).items():
        lines += [f"## by {k}", "", t.to_string(index=False), ""]
    (out / "qa.md").write_text("\n".join(lines))
    root = crops_root(cfg.crops)
    man = load_manifest(root, sorted(set(weak_df.video)))
    write_sheets(weak_df, man, CropStore(root), out / "sheets")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
