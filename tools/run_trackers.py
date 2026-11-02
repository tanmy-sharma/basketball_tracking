#!/usr/bin/env python
"""Run the Roboflow trackers (all but McByte) on cached detections, score them, and report where misses come from.

  python run_trackers.py IMG_0106 ../outputs/detections/IMG_0106__rfdetr-small__720p.txt [more.txt ...] [--sync-dir DIR]

Each GT box per frame gets one label, using image IoU >= 0.5:
  tracked        matched to a track, same track as last time
  idsw           matched, but to a different track than last time
  lost_assoc     no track here, yet a detection this tracker was given covers the box: the tracker failed
  lost_low_conf  no track here; a detection covers the box, but its score was under the tracker's input cutoff
  lost_no_det    no track here, and no detection covers the box: the detector failed

Writes MOT tracks and per-run scores to ../outputs/tracks/, and tracker_report.md plus tracker_summary.csv to
../outputs/reports/. The report is rewritten after every run. Runs with saved scores are skipped, so a rerun resumes, and
a run that fails doesn't stop the others. --sync-dir copies each result there (mirroring ../outputs) as soon as it exists.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import supervision as sv
import trackers as rf

from eval_detections import iou_matrix, load_dets, match, on_court
from trackid import eval_mot, io
from trackid.stratified import clear_ledger

ROOT = Path(__file__).resolve().parents[1]
FPS = 30000 / 1001
LABELS = ["tracked", "idsw", "lost_assoc", "lost_low_conf", "lost_no_det"]
MARGIN = 150.0  # cm around the court, as in eval_detections

# (class, library kwargs, input cutoff). BoT-SORT's camera-motion step is off because the camera is static.
# "lowthr" is there because RF-DETR recall falls quickly above 0.6, while several defaults start tracks at 0.7.
CONFIGS = {
    "sort__default": (rf.SORTTracker, {}, 0.25),
    "sort__lowthr": (rf.SORTTracker, {"track_activation_threshold": 0.4}, 0.4),
    "bytetrack__default": (rf.ByteTrackTracker, {}, 0.1),
    "bytetrack__lowthr": (rf.ByteTrackTracker, {"track_activation_threshold": 0.4, "high_conf_det_threshold": 0.4}, 0.1),
    "ocsort__default": (rf.OCSORTTracker, {}, 0.1),
    "ocsort__lowthr": (rf.OCSORTTracker, {"high_conf_det_threshold": 0.4}, 0.1),
    "botsort__default": (rf.BoTSORTTracker, {"enable_cmc": False}, 0.1),
    "botsort__lowthr": (rf.BoTSORTTracker, {"enable_cmc": False, "track_activation_threshold": 0.4,
                                            "high_conf_det_threshold": 0.4}, 0.1),
    "cbiou__default": (rf.CBIoUTracker, {}, 0.1),
    "cbiou__lowthr": (rf.CBIoUTracker, {"track_activation_threshold": 0.4, "high_conf_det_threshold": 0.4}, 0.1),
}


def run_tracker(cfg: str, dets: pd.DataFrame, n_frames: int, min_conf: float) -> np.ndarray:
    """Returns MOT rows: frame, id, x, y, w, h, 1, 1."""
    cls, kwargs, _ = CONFIGS[cfg]
    tracker = cls(frame_rate=FPS, **kwargs)
    by = {f: d for f, d in dets[dets.conf >= min_conf].groupby("frame")}
    rows = []
    for f in range(1, n_frames + 1):
        d = by.get(f)
        if d is None:
            det = sv.Detections.empty()
        else:
            xywh = d[["x", "y", "w", "h"]].to_numpy()
            det = sv.Detections(xyxy=sv.xywh_to_xyxy(xywh), confidence=d.conf.to_numpy(), class_id=np.zeros(len(d), int))
        out = tracker.update(det)
        for (x1, y1, x2, y2), tid in zip(out.xyxy, out.tracker_id):
            if tid >= 0:
                rows.append((f, tid, x1, y1, x2 - x1, y2 - y1, 1, 1))
    return np.array(rows, float).reshape(-1, 8)


def label_misses(gt: np.ndarray, tracks: np.ndarray, dets: pd.DataFrame, min_conf: float) -> tuple[dict, pd.DataFrame]:
    """Metrics plus one row per GT box: frame, GT id, label, overlap with other players."""
    data = eval_mot.build_sequence_data(seq_name="seq", gt=gt, tracker=tracks, eval_class_id=None, distractor_ids=set(),
                                        pred_class_id=None)
    bundle = eval_mot.eval_bundle(data, 0.5)
    summary = eval_mot.summarize_bundle(bundle)
    ledger = clear_ledger(data)
    if sum(int(x.sum()) for x in ledger["is_idsw"]) != summary["IDSW"]:
        raise AssertionError("per-box ID switches do not sum to the CLEAR IDSW total")

    all_by = {f: d for f, d in dets.groupby("frame")}
    rows = []
    for t in range(data.num_timesteps):
        g = eval_mot._rows_for_frame(gt, t + 1)
        if not len(g):
            continue
        box = g[:, 2:6]
        d = all_by.get(t + 1)
        d_box = np.empty((0, 4)) if d is None else d[["x", "y", "w", "h"]].to_numpy()
        used = match(box, d_box[(d.conf >= min_conf).to_numpy()] if d is not None else d_box)[0] >= 0
        any_det = match(box, d_box)[0] >= 0
        overlap = (iou_matrix(box, box) - np.eye(len(box))).max(axis=1)
        for i in range(len(box)):
            if ledger["assoc_tr_id"][t][i] >= 0:
                lab = "idsw" if ledger["is_idsw"][t][i] else "tracked"
            elif used[i]:
                lab = "lost_assoc"
            else:
                lab = "lost_low_conf" if any_det[i] else "lost_no_det"
            rows.append((t + 1, int(g[i, 1]), lab, overlap[i]))
    labels = pd.DataFrame(rows, columns=["frame", "gt_id", "label", "overlap"])
    if len(labels) != data.num_gt_dets:
        raise AssertionError(f"{len(labels)} labels for {data.num_gt_dets} GT boxes")
    return summary, labels


def one_run(job: tuple[str, str, str]) -> dict:
    video, det_path, cfg = job
    H, _ = io.load_homographies(io.TRAIN / "court_registration.csv")
    gt = eval_mot._read_mot_txt(io.GT_BBOX / f"{video}.txt")
    dets = load_dets(det_path)
    dets = dets[on_court(dets, H, MARGIN)]
    min_conf = CONFIGS[cfg][2]
    tracks = run_tracker(cfg, dets, int(gt[:, 0].max()), min_conf)

    stem = Path(det_path).stem
    out = ROOT / "outputs" / "tracks" / stem
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / f"{cfg}.txt", tracks, fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d")

    summary, labels = label_misses(gt, tracks, dets, min_conf)
    labels["label"] = labels.label.astype(str)
    share = labels.label.value_counts(normalize=True).reindex(LABELS, fill_value=0.0)
    res = dict(detector=stem, config=cfg, HOTA=summary["HOTA"], DetA=summary["DetA"], AssA=summary["AssA"],
               IDF1=summary["IDF1"], MOTA=summary["MOTA"], IDSW=summary["IDSW"], Frag=summary["Frag"],
               track_ids=len(np.unique(tracks[:, 1])) if len(tracks) else 0, **share.to_dict())
    bins = pd.cut(labels.overlap, [-1, 0, .1, .3, 1], labels=["none", "0-0.1", "0.1-0.3", ">0.3"])
    by = pd.crosstab(bins, labels.label, normalize="index").reindex(columns=LABELS, fill_value=0.0)
    res["by_overlap"] = {str(k): v for k, v in by.to_dict("index").items()}
    res["by_overlap_n"] = {str(k): int(v) for k, v in bins.value_counts().items()}
    # written last and atomically: a saved score means this run is complete
    path = out / f"{cfg}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, default=float))
    os.replace(tmp, path)
    return res


def safe_run(job: tuple[str, str, str]) -> tuple[tuple[str, str, str], dict | None, str | None]:
    """Keep one failed run from taking the others down."""
    try:
        return job, one_run(job), None
    except Exception:
        return job, None, traceback.format_exc()


def track_paths(det_path: str, cfg: str) -> tuple[Path, Path]:
    d = ROOT / "outputs" / "tracks" / Path(det_path).stem
    return d / f"{cfg}.txt", d / f"{cfg}.json"


def sync(sync_dir: Path | None, *paths: Path) -> None:
    """Copy finished files to the same place under sync_dir. The rename means a partial copy never looks complete."""
    if sync_dir is None:
        return
    for path in paths:
        dest = sync_dir / path.relative_to(ROOT / "outputs")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".partial")
        shutil.copyfile(path, tmp)
        os.replace(tmp, dest)


def to_md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) and abs(v) <= 1 else
                                       f"{v:.1f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_report(video: str, results: list[dict], failed: list[str], total: int) -> tuple[Path, Path]:
    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("by_overlap")} for r in results])
    table = table.sort_values("HOTA", ascending=False)
    rep = ROOT / "outputs" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    csv = rep / "tracker_summary.csv"
    table.to_csv(csv, index=False)

    best = max(results, key=lambda r: r["HOTA"])
    overlap = pd.DataFrame.from_dict(best["by_overlap"], orient="index").rename_axis("overlap bin").reset_index()
    text = [f"# Tracker comparison on {video}", "", f"{len(results)} of {total} runs finished." +
            (f" Failed: {', '.join(failed)}." if failed else ""), "",
            "Scores are image-box metrics (IoU), not TI-HOTA: nothing here assigns team or jersey.",
            "HOTA, DetA, AssA, IDF1 and MOTA are percentages. The label columns are shares of all GT boxes.", "",
            to_md(table), "", f"## Misses by overlap for the best run: {best['detector']} {best['config']}", "",
            "Each row is a share of the GT boxes in that overlap bin (max IoU with another player).", "",
            to_md(overlap), "", f"Boxes per bin: {best['by_overlap_n']}", ""]
    md = rep / "tracker_report.md"
    md.write_text("\n".join(text))
    return md, csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("dets", nargs="+")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--sync-dir", type=Path, help="copy each finished result here as soon as it exists")
    a = ap.parse_args()

    results, jobs = [], []
    for d in a.dets:
        for c in a.configs:
            saved = track_paths(d, c)[1]
            if saved.exists():
                print(f"skip {Path(d).stem} {c}: already done", flush=True)
                results.append(json.loads(saved.read_text()))
            else:
                jobs.append((a.video, d, c))

    failed = []
    with Pool(a.workers) as pool:
        for job, r, err in pool.imap_unordered(safe_run, jobs):
            if r is None:
                print(f"FAILED {Path(job[1]).stem} {job[2]}:\n{err}", flush=True)
                failed.append(f"{Path(job[1]).stem} {job[2]}")
            else:
                print(f"{r['detector']} {r['config']}: HOTA {r['HOTA']:.1f} IDF1 {r['IDF1']:.1f} IDSW {r['IDSW']}", flush=True)
                results.append(r)
                sync(a.sync_dir, *track_paths(job[1], job[2]), *write_report(a.video, results, failed, len(a.dets) * len(a.configs)))

    if results:
        report = write_report(a.video, results, failed, len(a.dets) * len(a.configs))
        sync(a.sync_dir, *report)
        print(report[0].read_text())
    if failed:
        sys.exit(f"{len(failed)} run(s) failed: {failed}")


if __name__ == "__main__":
    main()
