"""Held-out report for number readers on the test game.

python -m src.evaluate.classifier --game IMG_0108 --number dino=CFG --number vitstr=CFG \
    [--eligibility CFG] [--free-reads vitstr=reads.csv] --out DIR [--allow-nontest]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import Config, load_config
from src.core.constants import BIB_OF_ROSTER, NUMBER_TO_IDX, NUMBERS
from src.core.run import resolve_run_dir
from src.dataio.store import CropStore, crops_root, load_manifest
from src.evaluate import metrics as M
from src.weak.folds import predict_fn

WINDOWS = {"whole": float("inf"), "60s": 60.0, "10s": 10.0, "2s": 2.0}
THRESHOLDS = [0, .1, .3, .5, .7, .9]
ELIG_CUT = .5
LIMITS = """## Limits

- **Leakage.** The test game shares the competition's teams, bibs and venue with the training games, and only
  near-duplicate frames were removed from the validation block. Validation is for early stopping and is never a
  generalisation score. Models may partly remember people and bibs; the cross-colour runs are the only result that
  isolates digit reading from that.
- **Camera shift.** Training and test games come from the same sideline camera, but the challenge videos use a
  different camera. Scores here do not transfer to them and are an upper bound on what to expect.
- **Correlated crops.** Crops of one player on neighbouring frames are almost identical, so per-crop accuracy
  overstates certainty. Read the per-track table (6 players) as the honest variance estimate.
- **Roster limit.** Within-team numbers use the known 3-number roster of the player's team, as the pipeline does.
"""


def roster_mask(bibs) -> np.ndarray:
    roster_of = {b: r for r, b in BIB_OF_ROSTER.items()}
    unknown = set(bibs) - set(roster_of)
    if unknown:
        raise ValueError(f"unknown bib colours: {sorted(unknown)}")
    return np.array([[n in roster_of[b] for n in NUMBERS] for b in bibs], dtype=bool)


def _check_game(cfg: Config, game: str, allow_nontest: bool, what: str) -> None:
    if game not in cfg.games.test and not allow_nontest:
        raise ValueError(f"{game} is not in games.test of {what} ({cfg.games.test}); pass --allow-nontest to override")


def _best(cfg: Config) -> Path:
    p = resolve_run_dir(cfg) / "best.pt"
    if not p.exists():
        raise FileNotFoundError(f"no trained checkpoint at {p}")
    return p


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) or math.isinf(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def evaluate_subset(df: pd.DataFrame, probs: np.ndarray, allowed: np.ndarray, elig: np.ndarray | None) -> dict:
    y = df.number_idx.to_numpy()
    p9, pt = probs.argmax(1), M.masked_pred(probs, allowed)
    tracks = M.accuracy_by_track(df, pt)
    cm = M.confusion(y, p9)
    i = NUMBER_TO_IDX
    out = {"n": len(df), "acc_9way": M.per_crop_accuracy(y, probs), "acc_team": M.within_team_accuracy(y, probs, allowed),
           "per_track_team": tracks.to_dict("records"),
           "per_track_team_mean": float(tracks.accuracy.mean()), "per_track_team_std": float(tracks.accuracy.std(ddof=0)),
           "per_track_9way": M.accuracy_by_track(df, p9).to_dict("records"),
           "confusion": cm, "cells": {"1_as_11": int(cm[i["1"], i["11"]]), "11_as_1": int(cm[i["11"], i["1"]]),
                                      "10_as_12": int(cm[i["10"], i["12"]]), "12_as_10": int(cm[i["12"], i["10"]])},
           "by_bib": M.accuracy_by_bib(df, probs, allowed).to_dict("records"),
           "by_facing": M.accuracy_by(df, pt, "facing").to_dict("records"), "votes": {}, "coverage": {}}
    weight_sets = {"all_crops": None}
    if elig is not None:
        weight_sets["eligible_only"] = (elig >= ELIG_CUT).astype(float)
    for wname, wt in weight_sets.items():
        for mask_name, al in (("unmasked", None), ("roster_masked", allowed)):
            for win, secs in WINDOWS.items():
                out["votes"][f"{wname}/{mask_name}/{win}"] = M.summarize_votes(
                    M.tracklet_vote(df, probs, wt, secs, al))
    out["coverage"]["max_softmax"] = M.coverage_curve(y, probs, np.where(allowed, probs, 0).max(1), THRESHOLDS,
                                                      allowed).to_dict("records")
    if elig is not None:
        out["coverage"]["eligibility"] = M.coverage_curve(y, probs, elig, THRESHOLDS, allowed).to_dict("records")
        out["elig_auc_for_correct"] = M.auc(pt == y, elig)
        out["elig_auc_for_correct_max_softmax"] = M.auc(pt == y, np.where(allowed, probs, 0).max(1))
        keep = elig >= ELIG_CUT
        out["eligible_only"] = {"coverage": float(keep.mean()),
                                "acc_team": float((pt[keep] == y[keep]).mean()) if keep.any() else None}
    return out


def md(rows, cols: list[str] | None = None) -> str:
    df = pd.DataFrame(rows)
    if df.empty:
        return "(none)\n"
    df = df[cols] if cols else df
    f = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def render_report(game: str, res: dict, banner: str | None) -> str:
    L = [f"# Held-out report: {game}", ""]
    if banner:
        L += [f"> **{banner}**", ""]
    L += [LIMITS]
    for name, m in res["models"].items():
        for sub in ("all", "clean"):
            s = m[sub]
            L += [f"## {name} ({sub} crops, n={s['n']})", "",
                  f"- 9-way accuracy: {s['acc_9way']:.3f}", f"- within-team 3-way accuracy: {s['acc_team']:.3f}",
                  f"- per-track within-team accuracy: mean {s['per_track_team_mean']:.3f}, "
                  f"std {s['per_track_team_std']:.3f} over {len(s['per_track_team'])} tracks",
                  f"- look-alike cells (1/11, 10/12): {s['cells']}", ""]
            L += ["### Accuracy per track (within-team)", "", md(s["per_track_team"]),
                  "### 6 vs 9 vs 10 by bib colour", "", md(s["by_bib"]),
                  "### By facing (within-team)", "", md(s["by_facing"]),
                  "### Tracklet votes (sum of probabilities)", "",
                  md([{"setting": k, **v} for k, v in s["votes"].items()]), "### Coverage vs accuracy", ""]
            for cname, rows in s["coverage"].items():
                L += [f"{cname}:", "", md(rows)]
            if "elig_auc_for_correct" in s:
                L += [f"Eligibility AUC for correct reads: {_f(s['elig_auc_for_correct'])} "
                      f"(max-softmax baseline {_f(s['elig_auc_for_correct_max_softmax'])}). "
                      f"Eligible-only (score >= {ELIG_CUT}): {s['eligible_only']}", ""]
        if "free_reads" in m:
            L += ["### Free decoding (no roster limit)", "", md([m["free_reads"]])]
    L += ["Confusion matrices are in `confusion_<name>.csv` (rows true, columns predicted, NUMBERS order).", ""]
    return "\n".join(L)


def run_eval(game: str, numbers: dict[str, str], out: Path, eligibility: str | None = None,
             free_reads: dict[str, str] | None = None, allow_nontest: bool = False, device: str = "auto",
             overrides: list[str] = ()) -> dict:
    from src.core.loop import pick_device
    cfgs = {n: load_config(c, list(overrides)) for n, c in numbers.items()}
    if not cfgs:
        raise ValueError("give at least one --number NAME=CONFIG")
    ecfg = load_config(eligibility, list(overrides)) if eligibility else None
    for n, c in {**cfgs, **({"eligibility": ecfg} if ecfg else {})}.items():
        _check_game(c, game, allow_nontest, n)
    banner = None
    if allow_nontest:
        banner = f"{game} may be a training game: results are NOT held-out"
        print("=" * 70 + f"\n{banner}\n" + "=" * 70)
    first = next(iter(cfgs.values()))
    root = crops_root(first.crops)
    man = load_manifest(root, [game])
    man = man[(man.status == "ok") & (man.pose_min_conf >= first.filters.min_pose_conf) & (man.offset >= 0)]
    man = man.reset_index(drop=True)
    store, dev, allowed = CropStore(root), str(pick_device(device)), roster_mask(man.bib)
    elig = None
    if ecfg:
        elig = predict_fn(ecfg.plugin)(_best(ecfg), man, store, dev)[:, 1]
    thr = first.filters.max_iou
    clean = (man.max_iou <= thr).to_numpy() if thr is not None else np.ones(len(man), bool)
    reads = {n: pd.read_csv(p, dtype=str, keep_default_na=False) for n, p in (free_reads or {}).items()}
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"game": game, "banner": banner, "n_all": len(man), "n_clean": int(clean.sum()), "models": {}}
    for name, cfg in cfgs.items():
        probs = predict_fn(cfg.plugin)(_best(cfg), man, store, dev)
        m = {sub: evaluate_subset(man[mask].reset_index(drop=True), probs[mask], allowed[mask],
                                  None if elig is None else elig[mask])
             for sub, mask in (("all", np.ones(len(man), bool)), ("clean", clean))}
        if name in reads:
            m["free_reads"] = M.free_read_stats(man, reads[name], allowed)
        res["models"][name] = m
        pd.DataFrame(m["all"]["confusion"], index=NUMBERS, columns=NUMBERS).to_csv(out / f"confusion_{name}.csv")
        pt = M.masked_pred(probs, allowed)
        pd.DataFrame({"crop_id": man.crop_id, "video": man.video, "frame": man.frame, "track_id": man.track_id,
                      "jersey": man.jersey, "number_idx": man.number_idx, "pred_9way": probs.argmax(1),
                      "pred_team": pt, "p_true": probs[np.arange(len(man)), man.number_idx],
                      "p_max": probs.max(1), "clean": clean,
                      "elig": np.nan if elig is None else elig}).to_csv(out / f"predictions_{name}.csv.gz", index=False)
    (out / "metrics.json").write_text(json.dumps(_clean(res), indent=2))
    (out / "report.md").write_text(render_report(game, _clean(res), banner))
    return res


def _kv(items: list[str] | None) -> dict[str, str]:
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep or not k or not v:
            raise SystemExit(f"expected NAME=VALUE, got {it!r}")
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True)
    ap.add_argument("--number", action="append", default=[], metavar="NAME=CONFIG",
                    help="classifier or recognizer config (repeatable)")
    ap.add_argument("--eligibility", help="classifier config with head: eligibility")
    ap.add_argument("--free-reads", action="append", default=[], metavar="NAME=CSV",
                    help="free-decoding output with columns crop_id,text for the model NAME")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-nontest", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    run_eval(a.game, _kv(a.number), Path(a.out), a.eligibility, _kv(a.free_reads), a.allow_nontest, a.device)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
