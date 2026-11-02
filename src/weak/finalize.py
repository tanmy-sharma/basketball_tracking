"""Final eligible-crop list for number training.  `python -m src.weak.finalize --config weak_folds3.yaml`"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.core.config import WeakCfg, load_config
from src.core.paths import outputs_dir
from src.core.run import resolve_run_dir


def finalize_eligible(weak_df: pd.DataFrame, elig_scores: pd.Series | np.ndarray, weak: WeakCfg) -> pd.DataFrame:
    """Label 1 -> `weak`; uncovered (-2) with head score >= head_threshold -> `head`; per-number top-up -> `fill`.

    Labels 0, -1 and -3 are never admitted. Scores are a Series indexed by crop_id or an array in weak_df order.
    """
    scores = (elig_scores if isinstance(elig_scores, pd.Series)
              else pd.Series(np.asarray(elig_scores, dtype=float), index=weak_df.crop_id.to_numpy()))
    sc = weak_df.crop_id.map(scores).to_numpy(dtype=float)
    df = weak_df.assign(elig_score=sc)
    out = [df[df.label == 1].assign(source="weak")]
    unc = df[(df.label == -2) & df.elig_score.notna()]
    head = unc[unc.elig_score >= weak.head_threshold]
    out.append(head.assign(source="head"))
    taken = set(pd.concat(out).crop_id)
    counts = pd.concat(out).groupby("number_idx").size()
    pool = unc[~unc.crop_id.isin(taken)]
    for idx in sorted(df.number_idx.unique()):
        need = weak.fill_min_per_class - int(counts.get(idx, 0))
        if need > 0:
            out.append(pool[pool.number_idx == idx].nlargest(need, "elig_score").assign(source="fill"))
    res = pd.concat(out)[["crop_id", "source", "elig_score"]]
    return res.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="config with a `weak` block")
    ap.add_argument("--eligibility", help="eligibility-head config; needed only if uncovered crops exist")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="a.b=c")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    weak = cfg.weak
    if weak is None:
        raise SystemExit("config has no `weak` block")
    wdir = outputs_dir() / "weak" / weak.name
    weak_df = pd.read_csv(wdir / "weak_labels.csv", dtype={"crop_id": str, "video": str, "bib": str})
    scores = pd.Series(dtype=float)
    need = weak_df[weak_df.label == -2]
    if len(need):
        if not a.eligibility:
            raise SystemExit(f"{len(need)} uncovered crops need head scores: pass --eligibility CONFIG")
        scores = _head_scores(load_config(a.eligibility), need, a.device)
    res = finalize_eligible(weak_df, scores, weak)
    res.to_csv(wdir / "eligible_final.csv", index=False)
    print(res.groupby(["source"]).size().to_string())
    print(f"wrote {wdir / 'eligible_final.csv'} ({len(res)} rows)")


def _head_scores(ecfg, need: pd.DataFrame, device: str) -> pd.Series:
    from src.core.loop import pick_device
    from src.dataio.store import CropStore, crops_root, load_manifest
    root = crops_root(ecfg.crops)
    man = load_manifest(root, sorted(set(need.video))).set_index("crop_id").loc[need.crop_id].reset_index()
    from src.weak.folds import predict_fn
    probs = predict_fn(ecfg.plugin)(resolve_run_dir(ecfg) / "best.pt", man, CropStore(root),
                                    str(pick_device(device)))
    return pd.Series(probs[:, 1], index=need.crop_id.to_numpy())


if __name__ == "__main__":
    main()
