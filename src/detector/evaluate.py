"""Stock vs fine-tuned detections against the ground-truth player boxes, via tools/eval_detections.py.

  python -m src.detector.evaluate --game IMG_0108 --stock A.txt --finetuned B.txt [--dets name=path ...]
"""
from __future__ import annotations

import argparse
import contextlib
import io as stdio
from pathlib import Path

import numpy as np

from src.core.paths import add_tools_to_path, outputs_dir

HEADER = """\
# Detector evaluation: {game}

> **Camera-shift risk.** The fine-tuned detector is trained on the end-line camera of the training games, where
> only the 6 players are labelled, so officials and waiting players act as negatives. The held-out game below uses
> the same camera, so these scores cannot show how the detector behaves on the oblique challenge camera, where
> those same people stand near the court. Treat a fine-tuned win here as unproven for the challenge videos and keep
> stock RF-DETR as the fallback.

Scoring is on-court detections only (margin {margin:g} cm), at the confidence with the best on-court F1 unless fixed.
"""


def _ed():
    add_tools_to_path()
    import eval_detections
    return eval_detections


def compare(game: str, name: str, path: Path | str, margin: float = 150.0, conf: float | None = None) -> dict:
    """Recall, precision, F1 and foot-point error for one detections file (same logic as eval_detections.evaluate)."""
    ed = _ed()
    io = ed.io
    H, _ = io.load_homographies(io.TRAIN / "court_registration.csv")
    dets = ed.load_dets(path)
    frames = np.arange(dets.frame.min(), dets.frame.max() + 1)
    every = int(np.median(np.diff(np.sort(dets.frame.unique())))) if dets.frame.nunique() > 1 else 1
    frames = frames[(frames - frames[0]) % every == 0]
    bb = io.load_bboxes(game)
    bb = bb[bb.frame.isin(frames)]
    gt = {f: g[["x", "y", "w", "h"]].to_numpy() for f, g in bb.groupby("frame")}
    frames = np.array([f for f in frames if f in gt])
    court = dets[ed.on_court(dets, H, margin)]

    def prf(t: float) -> tuple[float, float, float]:
        tp, fn, fp = ed.score(gt, court[court.conf >= t], frames)
        r, p = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
        return r, p, 2 * r * p / max(r + p, 1e-9)

    t = conf if conf is not None else max(ed.THRESHOLDS, key=lambda c: prf(c)[2])
    recall, precision, f1 = prf(t)
    by = {f: d[["x", "y", "w", "h"]].to_numpy() for f, d in court[court.conf >= t].groupby("frame")}
    errs, misses = [], 0
    for f in frames:
        g, d = gt[f], by.get(f, np.empty((0, 4)))
        for i, j in enumerate(ed.match(g, d)[0]):
            if j < 0:
                misses += 1
                continue
            a = io.apply_h(H, [[g[i, 0] + g[i, 2] / 2, g[i, 1] + g[i, 3]]])[0]
            b = io.apply_h(H, [[d[j, 0] + d[j, 2] / 2, d[j, 1] + d[j, 3]]])[0]
            errs.append(float(np.hypot(*(a - b))))
    e = np.array(errs)
    q = (lambda p: float(np.quantile(e, p))) if len(e) else (lambda p: float("nan"))
    return {"name": name, "conf": float(t), "frames": int(len(frames)), "recall": recall, "precision": precision,
            "f1": f1, "err_median": q(.5), "err_p90": q(.9),
            **{f"within_{c}": float((e <= c).mean()) if len(e) else float("nan") for c in (13, 30, 100)}}


def render(game: str, rows: list[dict], logs: dict[str, str], margin: float = 150.0) -> str:
    out = [HEADER.format(game=game, margin=margin),
           "| detector | conf | frames | recall | precision | F1 | foot err median (cm) | p90 (cm) | <=13 cm | <=30 cm | <=100 cm |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['name']} | {r['conf']:.2f} | {r['frames']} | {r['recall']:.3f} | {r['precision']:.3f} | "
                   f"{r['f1']:.3f} | {r['err_median']:.1f} | {r['err_p90']:.1f} | {r['within_13']:.3f} | "
                   f"{r['within_30']:.3f} | {r['within_100']:.3f} |")
    for name, text in logs.items():
        out += ["", f"## eval_detections output: {name}", "", "```", text.rstrip(), "```"]
    return "\n".join(out) + "\n"


def run(game: str, dets: dict[str, Path | str], margin: float = 150.0, conf: float | None = None,
        out: Path | None = None) -> tuple[Path, list[dict]]:
    ed = _ed()
    rows, logs = [], {}
    for name, path in dets.items():
        rows.append(compare(game, name, path, margin, conf))
        buf = stdio.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                ed.evaluate(game, str(path), margin, conf)
        except Exception as ex:  # the long-form report is a bonus; keep the table
            buf.write(f"\n[eval_detections.evaluate stopped: {type(ex).__name__}: {ex}]\n")
        logs[name] = buf.getvalue()
    out = out or outputs_dir() / "reports" / "detector" / f"{game}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(game, rows, logs, margin))
    return out, rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True)
    ap.add_argument("--stock", type=Path)
    ap.add_argument("--finetuned", type=Path)
    ap.add_argument("--dets", nargs="*", default=[], metavar="name=path")
    ap.add_argument("--margin", type=float, default=150.0)
    ap.add_argument("--conf", type=float)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    dets: dict[str, Path | str] = {}
    if a.stock:
        dets["stock"] = a.stock
    if a.finetuned:
        dets["finetuned"] = a.finetuned
    for item in a.dets:
        name, sep, path = item.partition("=")
        if not sep:
            ap.error(f"--dets expects name=path, got {item!r}")
        dets[name] = path
    if not dets:
        ap.error("give at least one of --stock, --finetuned, --dets")
    out, rows = run(a.game, dets, a.margin, a.conf, a.out)
    print(f"report: {out}")


if __name__ == "__main__":
    main()
