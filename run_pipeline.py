#!/usr/bin/env python
"""Run the submitted pipeline end to end on the challenge videos. Inference only: nothing trains.

  uv run --no-sync python run_pipeline.py --config configs/submission.yaml \\
      --video-path /path/to/IMG_0114.MP4 /path/to/IMG_0115.MP4 --output-dir runs/fresh \\
      [--weights-dir weights] [--cache-dir results/challenge/intermediates] [--device auto|cuda|cpu] [--limit N] [--verify]

Stages, each an existing module. The settings in --config are passed explicitly; every other setting is that module's
source default at this commit, so the code version is part of the configuration.
  1. detect   src/infer/detect_video.py   RF-DETR Small (fine-tuned), every frame       -> detections/<V>__<model>__720p.txt
  2. jnr      src/infer/jnr_video.py      ViTPose torso crop, eligibility head, ViTSTR  -> jnr/<V>__<model>__720p__jnr.csv.gz
  3. colour   src/team/run.py             team colour labels per box                    -> labels/<V>__<model>__720p.csv.gz
  4. track    src/tracker/run_ucmc.py     team-aware UCMC tracker, ft_guard_lock30      -> tracks/<V>__ft_guard_lock30.txt
  5. stitch   src/tracker/jnr_tracks.py   JNR stitching, export, depth smoothing        -> submission/<V>.txt
  6. validate tools/validate_submission.py, and the upload ZIP
manifest.json then records, per stage output, its sha256 and whether it is byte-identical to the submitted run's
(reference: in --config), plus the TI-HOTA of the new submission scored against the submitted one. That score can be 1.0
without byte identity; only the hashes show identity.

The video id is the file name without extension (IMG_0114). Its roster and court registration are read from
data/Final challenge/. Weights are checked by sha256 before a model stage runs.

--cache-dir: a folder with any of detections/, jnr/ and labels/ (+ its .sha256) from an earlier run, under the names
above (the submission's own are in results/challenge/intermediates). A stage whose file is there is copied instead of run,
so the later stages can be replayed without a GPU. Jersey reads and labels point into the detection file, so a cached one
is accepted only if the detections also came from the cache, and every one, cached or computed, is checked row by row
against the detection file.
--output-dir: rerunning into the same folder resumes finished stages, but only with the same videos, config, cache,
device, --limit and code commit (recorded in run.json); anything else is refused. A finished stage is reused only if its
inputs and outputs still have the hashes recorded in done/<stage>.json; otherwise it is redone. The folder must not
overlap results/, data/, outputs/, the cache, the weights or the videos' folders; inside the repo it must be under runs/.
--limit N: smoke test on the first N >= 30 frames (no reference comparison).
--verify: strict replay. Exit with 1 unless every stage output is byte-identical to the submitted run. Without it, a
finished run exits with 0 and differences are only reported (a fresh run on other hardware is expected to differ).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
TRACKER_SETTING = "ft_guard_lock30"
FINAL = ROOT / "data" / "Final challenge"  # roster and court registration of the challenge videos
CHALLENGE = ["IMG_0114", "IMG_0115"]
SUBMITTED = ROOT / "results" / "challenge" / "submission"  # the uploaded files
PROTECTED = [ROOT / "results", ROOT / "data", ROOT / "outputs"]  # frozen references and inputs: never written


def sha256(path: Path) -> str:
    """Of the file's bytes; of the decompressed contents for .gz, whose header holds the time it was written."""
    h = hashlib.sha256()
    with (gzip.open if Path(path).suffix == ".gz" else open)(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decoded_frames(video: Path) -> int:
    """Frames that actually decode (docs/19): not metadata nb_frames, not the last frame with a detection."""
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True)
    return int(r.stdout.strip().split(",")[0])


class Runner:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.log = out / "logs"
        self.done = out / "done"
        # every stage writes under --output-dir (src/core/paths.py), never into the repo's outputs/
        self.env = {**os.environ, "SRC_OUTPUTS": str(out), "PYTHONUNBUFFERED": "1", "MPLBACKEND": "agg"}

    def __call__(self, name: str, cmd: list) -> None:
        cmd = [str(c) for c in cmd]
        print(f"\n=== {name}\n$ {' '.join(cmd)}", flush=True)
        t0 = time.time()
        self.log.mkdir(parents=True, exist_ok=True)
        with open(self.log / f"{name}.log", "w") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            p = subprocess.Popen(cmd, cwd=ROOT, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if p.wait():
                sys.exit(f"{name} failed with exit code {p.returncode}; see {self.log / f'{name}.log'}")
        print(f"--- {name}: {time.time() - t0:.0f}s", flush=True)

    def finished(self, stage: str, inputs: dict[str, str]) -> dict | None:
        """The stage's record if it finished earlier, was made from these inputs and its outputs are unchanged."""
        f = self.done / f"{stage}.json"
        if not f.is_file():
            return None
        rec = json.loads(f.read_text())
        why = None
        if rec.get("inputs") != inputs:
            why = "its inputs changed"
        else:
            for name, digest in rec.get("outputs", {}).items():
                path = self.out / name
                if not path.is_file():
                    why = f"{name} is missing"
                elif sha256(path) != digest:
                    why = f"{name} was changed"
                if why:
                    break
        if why:
            print(f"\n=== {stage}: finished earlier, but {why}; redoing it", flush=True)
            f.unlink()
            return None
        return rec

    def mark(self, stage: str, origin: str, inputs: dict[str, str], outputs: list[Path]) -> None:
        """Written only after a stage succeeded, so an interrupted stage is redone, never reused."""
        self.done.mkdir(parents=True, exist_ok=True)
        rec = {"origin": origin, "inputs": inputs,
               "outputs": {str(o.relative_to(self.out)): sha256(o) for o in outputs}}
        (self.done / f"{stage}.json").write_text(json.dumps(rec, indent=1))


def check_weights(cfg: dict, folder: Path) -> dict[str, Path]:
    found, bad = {}, []
    for key, w in cfg["weights"].items():
        p = folder / w["file"]
        if not p.is_file():
            bad.append(f"{key}: {p} is missing")
        elif (got := sha256(p)) != w["sha256"]:
            bad.append(f"{key}: {p} has sha256 {got[:8]}, expected {w['sha256'][:8]}")
        found[key] = p
    if bad:
        sys.exit("Weights do not match the submission (README, 'Weights'):\n  " + "\n  ".join(bad))
    return found


def overlaps(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def check_output_dir(out: Path, cache: Path | None, weights_dir: Path, videos: list[Path]) -> None:
    """Refuse, before anything is written, an output folder that could overwrite references, caches or inputs."""
    if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / "runs") or out == ROOT / "runs":
        sys.exit(f"--output-dir {out}: inside the repo it must be a folder under {ROOT / 'runs'}")
    guarded = [*PROTECTED, weights_dir, *([cache] if cache else []), *{v.parent for v in videos}]
    if hit := [str(g) for g in guarded if overlaps(out, g.resolve())]:
        sys.exit(f"--output-dir {out} overlaps protected folders {hit}; choose another folder")


def run_stamp(a: argparse.Namespace, cfg_text: str, videos: list[Path], cache: Path | None) -> dict:
    """Everything that decides a stage's output. A folder made with a different stamp is not reused."""
    cached = {}
    if cache:
        for kind in ("detections", "jnr", "labels"):
            for f in sorted((cache / kind).glob("*")) if (cache / kind).is_dir() else []:
                cached[f"{kind}/{f.name}"] = sha256(f)
    return {"videos": {v.stem: sha256(v) for v in videos}, "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
            "limit": a.limit, "device": a.device, "cache_files": cached, "commit": git_commit()}


def claim_output_dir(out: Path, stamp: dict) -> None:
    f = out / "run.json"
    if f.is_file():
        old = json.loads(f.read_text())
        if old != stamp:
            changed = sorted(k for k in stamp.keys() | old.keys() if stamp.get(k) != old.get(k))
            sys.exit(f"{out} was made with different {changed} (run.json). Use a new --output-dir.")
        print(f"resuming {out}: same videos, config, cache, device, limit and commit", flush=True)
    elif out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty and has no run.json. Use a new --output-dir.")
    else:
        out.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(stamp, indent=1))


def copy_from_cache(cache: Path, dests: list[Path]) -> None:
    for dest in dests:
        src = cache / dest.parent.name / dest.name
        if not src.is_file():
            sys.exit(f"cache {cache} has {dests[0].name} but not {src.name}, which must come with it")
        assert not overlaps(dest.resolve(), cache)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        print(f"\n=== {dest.parent.name}/{dest.name}: copied from cache {cache}", flush=True)


def check_rows(table: Path, dets: Path) -> None:
    """Jersey reads and colour labels point into the detection file by det_row (its 0-based line). Check that every
    row's frame and box are those of that line, so a file made from other detections is never used."""
    import numpy as np
    import pandas as pd
    d = pd.read_csv(dets, header=None, usecols=[0, 2, 3, 4, 5], names=["frame", "x", "y", "w", "h"])
    t = pd.read_csv(table, usecols=["det_row", "frame", "x", "y", "w", "h"])
    ok = bool(t.det_row.between(0, len(d) - 1).all())
    if ok:
        r = d.iloc[t.det_row.to_numpy()]
        # jersey files store boxes with 5 significant digits
        ok = bool((r.frame.to_numpy() == t.frame.to_numpy()).all()) and np.allclose(
            r[["x", "y", "w", "h"]].to_numpy(), t[["x", "y", "w", "h"]].to_numpy(), rtol=1e-4, atol=0.01)
    if not ok:
        sys.exit(f"{table.name} does not match {dets.name}: it was made from other detections")


def one_video(video: Path, cfg: dict, a: argparse.Namespace, run: Runner, weights) -> dict:
    out, vid, py = run.out, video.stem, sys.executable
    model = f"rfdetr-small-ft-{cfg['weights']['detector']['sha256'][:8]}"
    dets = out / "detections" / f"{vid}__{model}__720p.txt"
    jnr = out / "jnr" / f"{dets.stem}__jnr.csv.gz"
    labels = out / "labels" / f"{vid}__{model}__720p.csv.gz"
    tracks = out / "tracks" / f"{vid}__{TRACKER_SETTING}.txt"
    stitched = out / "tracks" / f"{vid}__{TRACKER_SETTING}__jnr.txt"
    sub = out / "submission" / f"{vid}.txt"
    for d in (tracks, sub):
        d.parent.mkdir(parents=True, exist_ok=True)
    limit = ["--limit", a.limit] if a.limit else []
    cache = a.cache_dir.resolve() if a.cache_dir else None

    print(f"\n##### {vid}: {video}", flush=True)
    n_frames = a.limit or decoded_frames(video)
    print(f"frames: {n_frames}", flush=True)
    j = cfg["jnr"]
    sidecar = Path(f"{labels}.sha256")  # sha256 of the detection file the labels were made from
    stages = [  # (name, outputs, upstream stages it was made from, command when it is computed)
        ("detections", f"{vid}_1_detect", [dets], [],
         lambda: [py, "-m", "src.infer.detect_video", "--video", video, "--weights", weights()["detector"],
                  "--previews", 0 if a.limit else cfg["detector"]["previews"], *limit]),
        ("jnr", f"{vid}_2_jnr", [jnr], ["detections"],
         lambda: [py, "-m", "src.infer.jnr_video", "--video", video, "--dets", dets, "--eligibility",
                  weights()["eligibility"], "--vitstr", weights()["vitstr"], "--every", j["every"],
                  "--min-conf", j["min_conf"], "--device", a.device, *limit]),
        ("labels", f"{vid}_3_colour", [labels, sidecar], ["detections"],
         lambda: [py, "-m", "src.team.run", vid, "--boxes", dets, "--video-path", video, "--seed", cfg["colour"]["seed"]]),
    ]
    origin: dict[str, str] = {}
    main_file = {kind: outs[0] for kind, _, outs, _, _ in stages}
    for kind, name, outs, deps, command in stages:
        inputs = {d: sha256(main_file[d]) for d in deps}
        if rec := run.finished(name, inputs):
            origin[kind] = f"reused_from_previous_run ({rec['origin']})"
            print(f"\n=== {name}: finished earlier ({rec['origin']}), outputs unchanged", flush=True)
            continue
        if cache and (cache / outs[0].parent.name / outs[0].name).is_file():
            # A cached file is only valid next to the cached files it was made from.
            if fresh := [d for d in deps if "copied_from_cache" not in origin[d]]:
                sys.exit(f"cache {cache} has {outs[0].name}, but its {fresh} were not taken from the cache. It was "
                         f"made from the cached ones, so it can't be used with others: remove it from the cache, or "
                         f"add the files it was made from")
            copy_from_cache(cache, outs)
            origin[kind] = "copied_from_cache"
        else:
            run(name, command())
            origin[kind] = "computed"
        if kind == "labels" and sidecar.read_text().split()[0] != sha256(dets):
            sys.exit(f"{labels.name} was made from other detections than {dets.name} ({sidecar.name})")
        if kind in ("jnr", "labels"):
            check_rows(outs[0], dets)
        run.mark(name, origin[kind], inputs, outs)
    t = cfg["tracker"]
    run(f"{vid}_4_track", [py, "src/tracker/run_ucmc.py", dets, FINAL / "court_registration.csv", tracks,
                           "--labels", labels, "--cdt", t["cdt"], "--ambiguity", t["ambiguity"],
                           "--team-scale", t["team_scale"], "--overlap-raw", t["overlap_raw"], "--q-bins", t["q_bins"],
                           "--guard-iou", t["guard_iou"], "--guard-front-px", t["guard_front_px"],
                           "--team-lock-frames", t["team_lock_frames"]])
    run(f"{vid}_5_stitch", [py, "src/tracker/jnr_tracks.py", tracks, jnr, "--video", vid, "--out", stitched,
                            "--roster", FINAL / "games_roster.csv", "--submission", sub, "--n-frames", n_frames,
                            "--smooth-sigma", cfg["stitching"]["smooth_sigma"]])

    files = {"detections": dets, "jnr": jnr, "labels": labels, "tracks": tracks, "submission": sub}
    m = {"video": str(video), "video_sha256": sha256(video), "n_frames": n_frames,
         "outputs": {k: sha256(p) for k, p in files.items()},
         "origin": {**origin, "tracks": "computed", "submission": "computed"}}
    ref = cfg.get("reference", {}).get(vid)
    if ref and not a.limit:
        m["reference"] = {k: ("same" if m["outputs"][k] == v else "DIFFERENT") for k, v in ref.items() if k in files}
        m["reference"]["n_frames"] = "same" if ref.get("n_frames") == n_frames else f"DIFFERENT: {ref.get('n_frames')}"
        submitted = SUBMITTED / f"{vid}.txt"
        if submitted.is_file():  # the submitted file scored as if it were the ground truth; 1.0 is not byte identity
            m["ti_hota_vs_submitted"] = round(agreement(submitted, sub), 6)
    return m


def agreement(reference: Path, new: Path) -> float:
    sys.path.insert(0, str(ROOT / "tools"))
    from trackid.metric import ti_hota_fast

    def rows(p: Path) -> list:
        out = []
        for line in p.read_text().split():
            f, x, y, team, jersey = line.split(",")
            out.append((int(f), float(x), float(y), team + jersey))
        return out
    return float(ti_hota_fast(rows(reference), rows(new))["TI-HOTA"])


def git_commit() -> str | None:
    r = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty", "--tags"], capture_output=True, text=True)
    return r.stdout.strip() or None


def cache_provenance(cache: Path | None) -> dict | None:
    """How the cached model outputs were made, as recorded when they were made (results/challenge/colab_reports)."""
    if cache is None:
        return None
    reports = cache.parent / "colab_reports"
    found = {str(f.relative_to(reports)): json.loads(f.read_text()) for f in sorted(reports.glob("*/weights_used.json"))}
    return {"weights_used": found, "note": "eligibility 'expected': false is explained in README, 'Weights'"} if found else {}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/submission.yaml")
    ap.add_argument("--video-path", type=Path, nargs="+", required=True, help="one or more videos")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    ap.add_argument("--cache-dir", type=Path, help="reuse detections/, jnr/ and labels/ files from this folder")
    ap.add_argument("--device", default="auto", help="for the jersey reader: auto, cuda, mps or cpu")
    ap.add_argument("--limit", type=int, help="smoke test: only the first N frames")
    ap.add_argument("--verify", action="store_true", help="strict replay: exit 1 unless every output matches the submission")
    a = ap.parse_args(argv)
    if a.cache_dir and a.limit:
        sys.exit("--cache-dir holds whole-video files; don't combine it with --limit")
    if a.limit is not None and a.limit < 30:  # the colour stage calibrates on 10 random frames from the middle 90%
        sys.exit("--limit must be at least 30 frames")
    if a.verify and a.limit:
        sys.exit("--verify compares whole videos; don't combine it with --limit")
    videos = [v.resolve() for v in a.video_path]
    if missing := [str(v) for v in videos if not v.is_file()]:
        sys.exit(f"not found: {missing}")
    if len({v.stem for v in videos}) != len(videos):
        sys.exit("two videos with the same name")
    cfg_text = a.config.read_text()
    cfg = yaml.safe_load(cfg_text)
    cache = a.cache_dir.resolve() if a.cache_dir else None
    if cache and not cache.is_dir():
        sys.exit(f"--cache-dir {cache} not found")
    out = a.output_dir.resolve()
    check_output_dir(out, cache, a.weights_dir.resolve(), videos)
    claim_output_dir(out, run_stamp(a, cfg_text, videos, cache))
    run = Runner(out)
    checked: dict[str, Path] = {}

    def weights() -> dict[str, Path]:  # checked only when a model stage actually runs
        if not checked:
            checked.update(check_weights(cfg, a.weights_dir.resolve()))
        return checked

    print(f"output {run.out}\npython {sys.version.split()[0]} on {platform.platform()}", flush=True)
    per_video = {v.stem: one_video(v, cfg, a, run, weights) for v in videos}
    manifest = {"config": str(a.config), "config_sha256": sha256(a.config), "limit": a.limit, "device": a.device,
                "python": sys.version.split()[0], "platform": platform.platform(), "commit": git_commit(),
                # loaded and hash-checked by this run; empty when every model output came from the cache
                "weights_verified_this_run": {k: sha256(p) for k, p in checked.items()},
                "cache_dir": str(cache) if cache else None, "cache_provenance": cache_provenance(cache),
                "videos": per_video}

    sub_dir = run.out / "submission"
    if sorted(per_video) == sorted(CHALLENGE):  # the validator checks the folder as the organisers receive it
        frames = [f"{v}={m['n_frames']}" for v, m in per_video.items()]
        zip_args = [] if a.limit else ["--make-zip", run.out / "submission.zip"]
        run("6_validate", [sys.executable, "tools/validate_submission.py", sub_dir, "--frames", *frames, *zip_args])
    else:
        print(f"\nNot validated: tools/validate_submission.py needs both {CHALLENGE} in one folder.")
    (run.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nsubmission files: {sub_dir}\nmanifest:         {run.out / 'manifest.json'}")
    diff = {v: {k: r for k, r in m["reference"].items() if r != "same"} for v, m in per_video.items() if "reference" in m}
    for v, m in per_video.items():
        if "reference" in m:
            print(f"{v} against the uploaded run: {json.dumps(m['reference'])}"
                  + (f", TI-HOTA vs submitted {m['ti_hota_vs_submitted']:.4f}" if "ti_hota_vs_submitted" in m else ""))
    if a.verify:
        if not all("reference" in m for m in per_video.values()):
            sys.exit("--verify: no reference for " + ", ".join(v for v, m in per_video.items() if "reference" not in m))
        if any(diff.values()):
            sys.exit(f"--verify: not identical to the submitted run: {json.dumps(diff)}")
        print("--verify: every stage output is byte-identical to the submitted run")
    elif any(diff.values()):
        print("Differences from the submitted run are reported above; expected for a fresh run on other hardware.")


if __name__ == "__main__":
    main()
