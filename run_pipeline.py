#!/usr/bin/env python
"""Run the submitted pipeline end to end on one challenge video. Inference only: nothing trains.

  uv run python run_pipeline.py --config configs/submission.yaml --video-path /path/to/IMG_0114.MP4 \\
      --output-dir runs/IMG_0114 [--weights-dir weights] [--cache-dir results/challenge/intermediates]
      [--device auto|cuda|cpu] [--limit N]

Stages, each an existing module called with the frozen values from --config (nothing relies on a stage's defaults):
  1. detect   src/infer/detect_video.py   RF-DETR Small (fine-tuned), every frame       -> detections/<V>__<model>__720p.txt
  2. jnr      src/infer/jnr_video.py      ViTPose torso crop, eligibility head, ViTSTR  -> jnr/<V>__<model>__720p__jnr.csv.gz
  3. colour   src/team/run.py             team colour labels per box                    -> labels/<V>__<model>__720p.csv.gz
  4. track    src/tracker/run_ucmc.py     team-aware UCMC tracker, ft_guard_lock30      -> tracks/<V>__ft_guard_lock30.txt
  5. stitch   src/tracker/jnr_tracks.py   JNR stitching, export, depth smoothing        -> submission/<V>.txt
  6. validate tools/validate_submission.py
Then every output's sha256 is compared with the uploaded run (reference: in --config) and written to manifest.json.

The video id is the file name without extension (IMG_0114). Its roster and court registration are read from
data/Final challenge/. Weights are checked by sha256 before anything runs.

--cache-dir: a folder with any of detections/, jnr/ and labels/ from an earlier run, under the names above (the
submission's own are in results/challenge/intermediates). A stage whose file is there is copied instead of run, so the
later stages can be checked without a GPU. Finished stage outputs already in --output-dir are reused, so a stopped run
resumes; delete --output-dir to start over.
--limit N: smoke test on the first N frames (stages 1-2 stop after N frames; the reference comparison is skipped).
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
        self.log.mkdir(parents=True, exist_ok=True)
        # every stage writes under --output-dir (src/core/paths.py), never into the repo's outputs/
        self.env = {**os.environ, "SRC_OUTPUTS": str(out), "PYTHONUNBUFFERED": "1", "MPLBACKEND": "agg"}

    def __call__(self, name: str, cmd: list) -> None:
        cmd = [str(c) for c in cmd]
        print(f"\n=== {name}\n$ {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with open(self.log / f"{name}.log", "w") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            p = subprocess.Popen(cmd, cwd=ROOT, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if p.wait():
                sys.exit(f"{name} failed with exit code {p.returncode}; see {self.log / f'{name}.log'}")
        print(f"--- {name}: {time.time() - t0:.0f}s", flush=True)


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


def from_cache(cache: Path | None, rel: str, dest: Path) -> bool:
    if cache is None or not (cache / rel).is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache / rel, dest)
    print(f"\n=== {rel}: copied from cache {cache}", flush=True)
    return True


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
    if not dets.is_file() and not from_cache(cache, f"detections/{dets.name}", dets):
        run(f"{vid}_1_detect", [py, "-m", "src.infer.detect_video", "--video", video, "--weights", weights()["detector"],
                                "--previews", 0 if a.limit else cfg["detector"]["previews"], *limit])
    if not jnr.is_file() and not from_cache(cache, f"jnr/{jnr.name}", jnr):
        j = cfg["jnr"]
        run(f"{vid}_2_jnr", [py, "-m", "src.infer.jnr_video", "--video", video, "--dets", dets,
                             "--eligibility", weights()["eligibility"], "--vitstr", weights()["vitstr"],
                             "--every", j["every"], "--min-conf", j["min_conf"], "--device", a.device, *limit])
    if not labels.is_file() and from_cache(cache, f"labels/{labels.name}", labels):  # with the hash of its detections
        from_cache(cache, f"labels/{labels.name}.sha256", Path(f"{labels}.sha256"))
    if not labels.is_file():
        run(f"{vid}_3_colour", [py, "-m", "src.team.run", vid, "--boxes", dets, "--video-path", video,
                                "--seed", cfg["colour"]["seed"]])
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
         "outputs": {k: sha256(p) for k, p in files.items()}}
    ref = cfg.get("reference", {}).get(vid)
    if ref and not a.limit:
        m["reference"] = {k: ("same" if m["outputs"][k] == v else "DIFFERENT") for k, v in ref.items() if k in files}
        m["reference"]["n_frames"] = "same" if ref.get("n_frames") == n_frames else f"DIFFERENT: {ref.get('n_frames')}"
        submitted = SUBMITTED / f"{vid}.txt"
        if submitted.is_file():  # how close: the submitted file scored as if it were the ground truth (1.0 = identical)
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/submission.yaml")
    ap.add_argument("--video-path", type=Path, nargs="+", required=True, help="one or more videos")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    ap.add_argument("--cache-dir", type=Path, help="reuse detections/ and jnr/ files from this folder")
    ap.add_argument("--device", default="auto", help="for the jersey reader: auto, cuda, mps or cpu")
    ap.add_argument("--limit", type=int, help="smoke test: only the first N frames")
    a = ap.parse_args(argv)
    if a.cache_dir and a.limit:
        sys.exit("--cache-dir holds whole-video files; don't combine it with --limit")
    videos = [v.resolve() for v in a.video_path]
    if missing := [str(v) for v in videos if not v.is_file()]:
        sys.exit(f"not found: {missing}")
    cfg = yaml.safe_load(a.config.read_text())
    run = Runner(a.output_dir.resolve())
    checked: dict[str, Path] = {}

    def weights() -> dict[str, Path]:  # checked only when a model stage actually runs
        if not checked:
            checked.update(check_weights(cfg, a.weights_dir.resolve()))
        return checked

    print(f"output {run.out}\npython {sys.version.split()[0]} on {platform.platform()}", flush=True)
    per_video = {v.stem: one_video(v, cfg, a, run, weights) for v in videos}
    manifest = {"config": str(a.config), "config_sha256": sha256(a.config), "limit": a.limit,
                "cache_dir": str(a.cache_dir) if a.cache_dir else None,
                "weights": {k: sha256(p) for k, p in checked.items()}, "videos": per_video}

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
    if any(diff.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
