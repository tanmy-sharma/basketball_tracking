#!/usr/bin/env python
"""Build colab/train_classifier.ipynb and colab/train_detector.ipynb, embedding the repo's code.

  python colab/build_src_notebooks.py

Only code, configs and lock files are embedded (see bundle_files). No videos, annotations or outputs.
Patterns (bundle, step/save/restore, Drive, download) follow colab/build_notebook.py, which is not changed.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

GAMES = ["IMG_0104", "IMG_0105", "IMG_0106", "IMG_0107", "IMG_0108", "IMG_0109", "IMG_0110", "IMG_0111"]
VIDEO_IDS = {
    "IMG_0104": "1e0r3kqSBbXVCwjDj4hh3_K7Oh_KKsTa7", "IMG_0105": "1yaAoPXn-aTLwmb-LasGMdKdyXff2TndK",
    "IMG_0106": "1vyVrNHIOJ9Fs3x7V5__XDndjJCDOR_17", "IMG_0107": "1rzzWnJyLt-ZAQvuYwQY7tWw7LalYLIiY",
    "IMG_0108": "1Ad2OB08gQsDHQEHleC7Wz_2VtmDtcyPe", "IMG_0109": "1rBMShAVMsnU38_1YQKcykIg1f2k52bdp",
    "IMG_0110": "1yNscs0JoERtgtt43oSGlO2SJebhtXMOJ", "IMG_0111": "1g6kidSDP4mo16dLsDWMeZXKNyIrqjCsD"}
BOXES_FOLDER = "1ONKvnUcfPG33mB9aYJBmj6aZMP_3QXvb"
COURT_FOLDER = "1JHXE8su46m9nRweCq0WonmPnNQhBbVwg"
REGISTRATION_ID = "1CDx1jVsYHs9CdvF3OC63gRk31WRvXEul"  # same ids as colab/build_notebook.py
ROSTER_ID = "1OL8kBVDOE3n6FK3swkc9Ic2DIzq9dzhH"
EXCLUDE_PACKAGES = ["python-doctr", "rf-groundingdino"]  # only reachable through trackers[detection]; src/ never imports them


def bundle_files() -> list[str]:
    """Repo-relative files that go into the notebook. Only the packages training needs, so work elsewhere in src/
    (tests, tracker, team) cannot change the code version, which names the Drive folder a run resumes from."""
    names = ["pyproject.toml", "uv.lock", "tools/eval_detections.py"]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "tools/trackid").glob("*.py")]
    packages = {"core", "dataio", "crops", "classifier", "recognizer", "weak", "evaluate", "detector", "configs"}
    for p in (ROOT / "src").rglob("*"):
        rel = p.relative_to(ROOT)
        if "__pycache__" in rel.parts or not p.is_file() or (len(rel.parts) > 2 and rel.parts[1] not in packages):
            continue
        if p.suffix == ".py" or (p.suffix == ".yaml" and rel.parts[:2] == ("src", "configs")):
            names.append(rel.as_posix())
    return sorted(set(names))


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def make_bundle() -> tuple[str, str, dict]:
    """(base64 tar.gz, code version, CODE_VERSION.json content). The version hashes names and contents."""
    names = bundle_files()
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode() + b"\0" + (ROOT / n).read_bytes() + b"\0")
    sha = h.hexdigest()
    status = _git("status", "--porcelain")
    meta = {"git_hash": _git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status),
            "bundle_sha256": sha}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for n in names:
            tar.add(ROOT / n, arcname=n)
        data = json.dumps(meta, indent=1).encode()
        info = tarfile.TarInfo("CODE_VERSION.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode(), sha[:8], meta


LICENCES = """**Public checkpoints and code used (all disclosed in the paper):**

| Component | Source | Licence |
|---|---|---|
| RF-DETR Small, COCO-pretrained (detector notebook; also the package `rfdetr`) | Roboflow | Apache 2.0 |
| DINOv2 small, `facebook/dinov2-small` (9-class and eligibility heads) | Meta | Apache 2.0 |
| ViTPose base, `usyd-community/vitpose-base-simple` (torso crops) | Hugging Face transformers port | Apache 2.0 |
| ViTSTR-small, `vitstr_small_patch16_224_aug.pth` (digit reader) | `roatienza/deep-text-recognition-benchmark` | Repo Apache 2.0; the weight files carry no separate licence and were trained on synthetic text |
| ResNet-34 ImageNet weights (classifier baseline) | torchvision | Code BSD-3; ImageNet data terms apply to the weights |

No other pretrained weights are loaded. All fine-tuning uses only the official training videos and annotations."""

COMMON_HOW = """**Setup:** Runtime > Change runtime type > **A100 GPU**, then Runtime > **Run all**. Keep this tab open.

1. Allow the Google Drive prompt. Each of the 8 training videos is taken from the runtime disk if present, then from
   `My Drive/tracking_videos` (any subfolder), and only then downloaded with `gdown`. If a video cannot be found,
   downloaded or read, the run prints a warning and **continues without it**, in a separate Drive folder whose name
   ends in `_without_<videos>`. Without IMG_0108 the models still train, but nothing can be scored.
2. Everything else runs unattended. Results are saved to
   `My Drive/tracking_results/src_@NOTEBOOK@/<code version>/` after every step, and training checkpoints are copied
   there every 10 minutes. If the runtime disconnects, choose Run all again: finished steps are skipped and an
   unfinished training run continues from its last copied checkpoint.
3. At the end, even if a step failed, two zips are built, copied to Drive and downloaded: `reports_<ver>.zip` first,
   then `weights_<ver>.zip`. **Allow multiple downloads** for this site in your browser, or the second file may be
   blocked. Both files are also in Drive, whatever the browser does.
4. The runtime check below warns if the GPU is not an A100. The run still works but is slower."""

INTRO_CLASSIFIER = f"""# Jersey-number pipeline: crops, weak labels, ViTSTR and DINOv2 readers (all 8 training videos)

{COMMON_HOW}

**What runs:** torso crops with ViTPose, 3-fold weak eligibility labels from fine-tuned ViTSTR-small, the eligibility
head, then ViTSTR fine-tunes (all crops, eligible crops, two cross-colour holdouts) and, if `RUN_BASELINES` is true, the
DINOv2 and ResNet-34 baselines. IMG_0108 is the only test game and is never trained on. The report is scored on it.
The stock (not fine-tuned) ViTSTR is scored too, as the untrained baseline. Fold-model weights stay in Drive and are
left out of `weights_<ver>.zip`.

{LICENCES}"""

INTRO_DETECTOR = f"""# RF-DETR Small fine-tune on the training games (all 8 videos)

{COMMON_HOW}

**What runs:** RF-DETR Small is fine-tuned on the 7 training games (player boxes only). Stock and fine-tuned
detections on IMG_0108 are then scored against its ground truth. IMG_0108 is never trained on. The fine-tuned detector
is tuned to the end-line camera, so keep stock RF-DETR as the fallback for the challenge videos.

{LICENCES}"""

SETUP = '''import os, subprocess, sys, time
from pathlib import Path

NOTEBOOK = "@NOTEBOOK@"
CODE_VERSION = "@CODE_VERSION@"
DETECTOR = @DETECTOR@
@FLAGS@GAMES = @GAMES@
REPO = Path("/content/tracking_challenge")
PY = REPO / ".venv/bin/python"
OUTPUTS = REPO / "outputs"
REPORTS = OUTPUTS / "reports"
ZIP_DIR = Path("/content")

def sh(cmd, cwd=REPO, log=None):
    """Run a command, stream its output, optionally keep a copy, and stop the notebook if it fails."""
    print("$", " ".join(map(str, cmd)), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "agg"}  # Colab's inline backend does not exist in our env
    p = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = open(log, "a") if log else None
    for line in p.stdout:
        print(line, end="", flush=True)
        if out:
            out.write(line)
    if out:
        out.close()
    if p.wait():
        raise RuntimeError(f"command failed with exit code {p.returncode}: {cmd}")

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
assert gpu.returncode == 0, "No GPU. Set Runtime > Change runtime type > A100 GPU, then run again."
print(gpu.stdout.strip())
if "A100" not in gpu.stdout:
    print("WARNING: this is not an A100. The run still works but will be slower, and the time estimates do not apply.")'''

DRIVE = '''# Drive holds the results and, optionally, the videos
from google.colab import drive

drive.mount("/content/drive")
DRIVE_VIDEOS = Path("/content/drive/MyDrive/tracking_videos")
RESULTS = Path(f"/content/drive/MyDrive/tracking_results/src_{NOTEBOOK}/{CODE_VERSION}")
RESULTS.mkdir(parents=True, exist_ok=True)
print("Results will be saved to", RESULTS)'''

UNPACK = '''# Code bundled by colab/build_src_notebooks.py: pyproject, uv.lock, src/ (without tests and tracker), configs and eval helpers
import base64, io, tarfile
BUNDLE = "@BUNDLE@"
REPO.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE))) as tar:
    tar.extractall(REPO, filter="data")
print("Unpacked", sum(1 for _ in REPO.rglob("*") if _.is_file()), "files; code version", CODE_VERSION)'''

INSTALL = '''# Same locked versions as the local repo, in a separate env so Colab's own packages don't interfere.
# python-doctr and rf-groundingdino are only pulled in by trackers[detection]; src/ never imports them, so they are skipped.
EXCLUDE = @EXCLUDE@
sh([sys.executable, "-m", "pip", "install", "-q", "uv", "gdown"])
sh([sys.executable, "-m", "uv", "sync", "--frozen", "--no-dev", "--python", "3.12",
    *(["--group", "rfdetr"] if DETECTOR else []),
    *[a for p in EXCLUDE for a in ("--no-install-package", p)]])
check = ("import torch, timm, transformers, cv2; from importlib.metadata import version; "
         + ("import rfdetr; " if DETECTOR else "")
         + "print('torch', torch.__version__, 'timm', timm.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available()); "
         "assert torch.cuda.is_available(), 'torch cannot see the GPU'")
sh([PY, "-c", check])
sh([PY, "-c", "import src.core.config, src.train; print('src imports OK')"])'''

DATA = '''# Annotations from the official challenge Drive; videos from the runtime disk, your Drive, or gdown (in that order).
import gdown, shutil

TRAIN = REPO / "data/Training"
BOX_DIR = TRAIN / "full player annotations/Image-space bounding boxes"
COURT_DIR = TRAIN / "full player annotations/court position"
VIDEO_DIR = TRAIN / "vidoes"
BOXES_FOLDER, COURT_FOLDER = "@BOXES@", "@COURT@"
FILES = {"@REG@": TRAIN / "court_registration.csv", "@ROSTER@": TRAIN / "games_roster.csv"}
VIDEO_IDS = @VIDEO_IDS@

def fetch_folder(folder_id, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if all((dest / f"{g}.txt").exists() for g in GAMES):
        return
    try:
        gdown.download_folder(id=folder_id, output=str(dest), quiet=True, remaining_ok=True)
    except Exception as e:  # whatever did arrive is still used; games without labels are dropped below
        print(f"WARNING: {dest.name} download failed: {e!r}", flush=True)
    for f in dest.rglob("*.txt"):  # a nested folder in the download is flattened
        if f.parent != dest and not (dest / f.name).exists():
            shutil.move(str(f), dest / f.name)

fetch_folder(BOXES_FOLDER, BOX_DIR)
fetch_folder(COURT_FOLDER, COURT_DIR)
for file_id, path in FILES.items():
    if not path.exists() and not gdown.download(id=file_id, output=str(path), quiet=True):
        raise RuntimeError(f"Could not download {path.name} from Drive (https://drive.google.com/file/d/{file_id})")

def label_last_frame(game):
    with open(BOX_DIR / f"{game}.txt") as f:
        return max(int(line.split(",", 1)[0]) for line in f if line.strip())

def frame_count(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return int(r.stdout.strip().strip(","))

def in_drive(game):
    if not DRIVE_VIDEOS.is_dir():
        return None
    return next((p for p in sorted(DRIVE_VIDEOS.rglob("*")) if p.is_file() and p.name.lower() == f"{game}.mov".lower()), None)

def get_game(game):
    """Returns None when the game is ready, otherwise why it cannot be used. Never raises: one bad video must not stop the run."""
    dest = VIDEO_DIR / f"{game}.MOV"
    partial = dest.with_name(dest.name + ".partial")
    try:
        if not (BOX_DIR / f"{game}.txt").exists() or not (COURT_DIR / f"{game}.txt").exists():
            return "its annotation files are missing"
        if not dest.exists():
            src = in_drive(game)
            if src is not None:
                print(f"{game}: copying from {src}", flush=True)
                shutil.copyfile(src, partial)
            else:
                print(f"{game}: not on the runtime disk or in {DRIVE_VIDEOS}; trying gdown", flush=True)
                if not gdown.download(id=VIDEO_IDS[game], output=str(partial), quiet=False):
                    raise RuntimeError("gdown got nothing, usually Google's download quota")
            os.replace(partial, dest)
        n, last = frame_count(dest), label_last_frame(game)
        if n < last:
            dest.unlink()
            return f"the video has {n} frames but its labels reach frame {last}, so it is incomplete or the wrong file"
        print(f"{game}: OK, {n} video frames, labels end at {last}" + (f" ({n - last} unlabelled at the end)" if n > last else ""))
    except Exception as e:
        partial.unlink(missing_ok=True)
        return repr(e)[:300]

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
why_missing = {g: why for g in GAMES if (why := get_game(g))}
MISSING = list(why_missing)
GAMES = [g for g in GAMES if g not in why_missing]
if not [g for g in GAMES if g != "IMG_0108"]:
    raise RuntimeError(f"No training video could be used: {why_missing}")
if MISSING:
    # A reduced run is a different experiment, so it gets its own Drive folder and config hashes and never mixes with a full run.
    RESULTS = RESULTS.parent / f"{CODE_VERSION}_without_{'_'.join(g[-4:] for g in MISSING)}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    sh([PY, "-m", "src.core.drop_games", "--games", *MISSING])
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "missing_videos.txt").write_text("".join(f"{g}: {w}\\n" for g, w in why_missing.items()))
    print("=" * 70)
    print(f"WARNING: CONTINUING WITHOUT {len(MISSING)} OF 8 VIDEOS. Results will be saved to {RESULTS}")
    for g, w in why_missing.items():
        print(f"  {g}: {w}\\n    To include it, add it to My Drive/tracking_videos and choose Run all (a new run starts): https://drive.google.com/file/d/{VIDEO_IDS[g]}")
    if "IMG_0108" in MISSING:
        print("  IMG_0108 is the test game: models will train, but nothing can be scored.")
    print("=" * 70)
print("Data ready:", GAMES)'''

HELPERS = '''# Save to Drive after every step, restore on restart, and package everything at the end.
import csv, re, shutil, threading, zipfile
from google.colab import drive, files

SYNC_MINUTES = 10
RESUME_FILES = ("last.pt", "last.ckpt")
_drive_lock = threading.RLock()  # the timer and the step-end save must not write the same .partial at once
_stop_sync = threading.Event()
_sync_thread = None
_synced = {}

def skip_file(f):
    """Never copied to Drive or zipped: temp files, resume state, and RF-DETR's dataset and intermediate checkpoints."""
    return (f.suffix in (".tmp", ".partial", ".ckpt") or f.name in ("last.pt",) or f.name.startswith("checkpoint")
            or "detector_dataset" in f.parts)

def _copy(f, dest):
    # Same size is not enough: best.pt keeps its size every epoch, so an older copy from the timer would stay in Drive.
    if dest.exists() and dest.stat().st_size == f.stat().st_size and dest.stat().st_mtime >= f.stat().st_mtime:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")  # the rename means a partial copy never looks complete
    shutil.copyfile(f, partial)
    os.replace(partial, dest)

def save(*items):
    with _drive_lock:
        for item in items:
            path = OUTPUTS / item
            for f in sorted(path.rglob("*")) if path.is_dir() else [path] if path.exists() else []:
                if f.is_file() and not skip_file(f):
                    _copy(f, RESULTS / f.relative_to(OUTPUTS))

def _copy_checkpoint(f, dest):
    """Copy a file that training may rewrite at any moment. Returns True if a new version reached Drive."""
    st = f.stat()
    state = (st.st_mtime_ns, st.st_size)
    if _synced.get(f) == state or time.time() - st.st_mtime < 10:  # unchanged, or possibly still being written
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    shutil.copyfile(f, partial)
    st = f.stat()
    if (st.st_mtime_ns, st.st_size) != state:  # rewritten during the copy: try again next time
        partial.unlink(missing_ok=True)
        return False
    os.replace(partial, dest)
    _synced[f] = state
    return True

def sync_runs():
    """Copy unfinished runs to Drive, resume files last so they never claim a best model Drive does not have yet.
    Finished runs drop their resume state from Drive: they only need what save() keeps."""
    copied = 0
    with _drive_lock:
        for run in sorted((OUTPUTS / "runs").glob("*")) if (OUTPUTS / "runs").is_dir() else []:
            dest = RESULTS / run.relative_to(OUTPUTS)
            if (run / "done.json").exists():
                for f in list(dest.rglob("*")) if dest.is_dir() else []:
                    if f.is_file() and (f.name in RESUME_FILES or f.name.startswith("checkpoint")):
                        f.unlink()
                continue
            found = [f for f in sorted(run.rglob("*")) if f.is_file() and f.suffix not in (".tmp", ".partial")
                     and not re.match(r"checkpoint_\\d", f.name)]  # RF-DETR's interval archives stay on the runtime disk
            for f in sorted(found, key=lambda p: p.name in RESUME_FILES):
                try:
                    copied += _copy_checkpoint(f, RESULTS / f.relative_to(OUTPUTS))
                except OSError as e:  # the file was replaced, or Drive hiccuped: the next pass retries
                    print(f"checkpoint copy of {f.name} failed: {e!r}", flush=True)
        save("reports", "weak")
    return copied

def _sync_loop():
    while not _stop_sync.wait(SYNC_MINUTES * 60):
        try:
            if Path("/content/drive/MyDrive").exists():  # never write into an unmounted path
                n = sync_runs()
                if n:
                    print(f"[{time.strftime('%H:%M:%S')}] {n} checkpoint files copied to Drive", flush=True)
        except Exception as e:
            print(f"checkpoint sync failed, will retry: {e!r}", flush=True)

def start_sync():
    global _sync_thread
    _stop_sync.clear()
    _sync_thread = threading.Thread(target=_sync_loop, daemon=True)
    _sync_thread.start()

def stop_sync():
    _stop_sync.set()
    if _sync_thread is not None:
        _sync_thread.join()

def restore():
    for f in RESULTS.rglob("*"):  # the zips package() left in Drive are not outputs: restoring them nests them in the next zip
        if f.is_file() and f.suffix != ".partial" and not (f.parent == RESULTS and f.suffix == ".zip"):
            _copy(f, OUTPUTS / f.relative_to(RESULTS))

failures = []

def done(name):
    return (OUTPUTS / "done" / name).exists()

def step(name, cmd, *items, needs=()):
    """Run one step unless it is done; skip it if a prerequisite is missing. Whatever exists is saved even on failure.
    `cmd` is a list, or a function returning one (called only when the step runs)."""
    if done(name):
        print(f"=== {name}: already done, skipping", flush=True)
        return True
    if any(not done(n) for n in needs):
        print(f"=== {name}: skipped, needs {[n for n in needs if not done(n)]}", flush=True)
        return False
    print(f"=== {name}", flush=True)
    start, ok = time.time(), False
    try:
        sh(cmd() if callable(cmd) else cmd, log=REPORTS / "logs" / f"{name}.txt")
        (OUTPUTS / "done").mkdir(parents=True, exist_ok=True)
        (OUTPUTS / "done" / name).touch()
        ok = True
    except Exception as e:
        failures.append(name)
        print(f"{name} FAILED: {e}", flush=True)
    finally:
        mins = (time.time() - start) / 60
        print(f"{name}: {mins:.1f} min", flush=True)
        with open(REPORTS / "step_times.csv", "a", newline="") as f:
            csv.writer(f).writerow([name, f"{mins:.2f}", "ok" if ok else "failed"])
        save(*items, "reports", "done")
        sync_runs()  # a failed run keeps its resume point; a finished one drops it
    return ok

def run_dir(config):
    """Run directory of a config (its name includes the config hash)."""
    code = "import sys; from src.core.config import load_config; from src.core.run import resolve_run_dir; print(resolve_run_dir(load_config(sys.argv[1])))"
    r = subprocess.run([PY, "-c", code, config], cwd=REPO, capture_output=True, text=True, check=True)
    return Path(r.stdout.strip().splitlines()[-1])

def package():
    """Build reports and weights zips, copy both to Drive, flush, then download reports first. Safe to call again."""
    stop_sync()
    ver = CODE_VERSION
    reports_zip, weights_zip = ZIP_DIR / f"reports_{ver}.zip", ZIP_DIR / f"weights_{ver}.zip"
    weight_names = ("best.pt", "best.pth")
    # strict_timestamps=False: a file dated before 1980, like CODE_VERSION.json from the bundle, must not stop the packaging
    with zipfile.ZipFile(reports_zip, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as z:
        for f in sorted(OUTPUTS.rglob("*")):
            rel = f.relative_to(OUTPUTS)
            if (f.is_file() and not skip_file(f) and rel.parts[0] != "crops" and f.name not in weight_names
                    and f.suffix not in (".pt", ".pth")):
                z.write(f, rel)
        if (REPO / "CODE_VERSION.json").exists():
            z.write(REPO / "CODE_VERSION.json", "CODE_VERSION.json")
    with zipfile.ZipFile(weights_zip, "w", zipfile.ZIP_STORED, strict_timestamps=False) as z:  # weights do not compress
        for f in sorted(OUTPUTS.glob("runs/*/*")):
            if f.name in weight_names and not f.parent.name.startswith("fold_"):
                z.write(f, f.relative_to(OUTPUTS))
    print(f"built {reports_zip.name} ({reports_zip.stat().st_size / 1e6:.1f} MB), {weights_zip.name} ({weights_zip.stat().st_size / 1e6:.1f} MB)")
    try:
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        RESULTS.mkdir(parents=True, exist_ok=True)
        for z in (reports_zip, weights_zip):
            _copy(z, RESULTS / z.name)
        drive.flush_and_unmount()  # Drive uploads in the background; make sure everything has reached it
        print("Results are in Drive:", RESULTS)
    except Exception as e:
        print("Copy to Drive failed:", repr(e), "- the zips are in", ZIP_DIR)
    for z in (reports_zip, weights_zip):
        try:
            files.download(str(z))
        except Exception as e:
            print(f"Download of {z.name} failed ({e!r}). Get it from Drive: {RESULTS / z.name}")'''

RUN_CLASSIFIER = '''# Every step is skipped if done, and saved to Drive right after it. The packaging in `finally` also runs after a failure.
CLS, REC = "src/configs/classifier", "src/configs/recognizer"
EVAL_GAME = "IMG_0108"
WEAK = f"{CLS}/weak_folds3.yaml"
VIT = ["vitstr_all", "vitstr_eligible", "vitstr_xcolor_no_yellow", "vitstr_xcolor_no_green"]
DINO = ["number_all", "number_eligible", "xcolor_no_yellow", "xcolor_no_green", "number_eligible_resnet34"]
CONFIG = {**{n: f"{REC}/{n}.yaml" for n in VIT}, **{n: f"{CLS}/{n}.yaml" for n in DINO}}
CROPS = tuple(f"crops_{g}" for g in GAMES)
NEEDS = {"vitstr_eligible": ("finalize",), "number_eligible": ("finalize",), "number_eligible_resnet34": ("finalize",)}

STOCK_EVAL = @STOCK_EVAL@

def train(name):
    return step(name, [PY, "-m", "src.train", "--config", CONFIG[name]], "runs", needs=NEEDS.get(name, ()))

def free_read(name):
    return step(f"free_{name}", lambda: [PY, "-m", "src.recognizer.predict", "--config", CONFIG[name], "--best", run_dir(CONFIG[name]) / "best.pt",
                                          "--game", EVAL_GAME, "--out", REPORTS / "free" / name, "--mode", "free"],
                needs=(name,))

def evaluate():
    names = [n for n in VIT + (DINO if RUN_BASELINES else []) if done(n)]
    if not names:
        print("=== evaluate: no finished runs yet")
        return
    reads = [f"{n}={next((REPORTS / 'free' / n).glob('*.csv'))}" for n in VIT if n in names and done(f"free_{n}")]
    def cmd():
        return [PY, "-m", "src.evaluate.classifier", "--game", EVAL_GAME, "--out", REPORTS / "eval" / EVAL_GAME,
                *[a for n in names for a in ("--number", f"{n}={CONFIG[n]}")],
                *(["--eligibility", f"{CLS}/eligibility.yaml"] if done("eligibility") else []),
                *[a for r in reads for a in ("--free-reads", r)]]
    step(f"evaluate_{len(names)}runs_{len(reads)}reads", cmd)

(REPORTS / "logs").mkdir(parents=True, exist_ok=True)
drive.mount("/content/drive")
restore()
start_sync()
try:
    # one step per game, so a disconnect loses at most one game's crops. A video that cannot be decoded is dropped too.
    usable = [g for g in GAMES if step(f"crops_{g}", [PY, "-m", "src.crops.extract", "--config", f"{CLS}/base.yaml", "--games", g], "crops")]
    lost = [g for g in GAMES if g not in usable]
    if lost:
        print("WARNING: continuing without", lost, "because their crops could not be extracted", flush=True)
        sh([PY, "-m", "src.core.drop_games", "--games", *lost])
        MISSING, GAMES = MISSING + lost, usable
    CROPS = tuple(f"crops_{g}" for g in GAMES)
    has_test = EVAL_GAME in GAMES
    step("weak_folds", [PY, "-m", "src.weak.folds", "--config", WEAK], "runs", "weak", needs=CROPS)
    step("eligibility", [PY, "-m", "src.train", "--config", f"{CLS}/eligibility.yaml"], "runs", needs=("weak_folds",))
    step("finalize", [PY, "-m", "src.weak.finalize", "--config", WEAK, "--eligibility", f"{CLS}/eligibility.yaml"], "weak", needs=("eligibility",))
    step("weak_qa", [PY, "-m", "src.weak.qa", "--config", WEAK], "weak", needs=("weak_folds",))
    if has_test:
        step("stock_vitstr", [PY, "-m", "src.recognizer.predict", "--config", f"{REC}/vitstr_all.yaml", "--stock", "--game", EVAL_GAME,
                              "--out", REPORTS / "stock_vitstr", "--mode", "both"], needs=CROPS)
        step("stock_vitstr_eval", [PY, "-c", STOCK_EVAL, f"{REC}/vitstr_all.yaml", REPORTS / "stock_vitstr"], needs=("stock_vitstr",))
    for name in VIT:
        if train(name) and has_test:
            free_read(name)
    if RUN_BASELINES:
        for name in DINO:
            train(name)
    if has_test:
        evaluate()
finally:
    package()

if MISSING:
    print("WARNING: this run did NOT use", MISSING)
if failures:
    raise RuntimeError(f"Failed steps: {failures}. Finished work is in Drive. Choose Run all to retry only what is missing.")'''

RUN_DETECTOR = '''# Every step is skipped if done, and saved to Drive right after it. The packaging in `finally` also runs after a failure.
EVAL_GAME = "IMG_0108"
CONFIG = "src/configs/detector/rfdetr_small.yaml"
DET = OUTPUTS / "detections"

def finetuned_weights():
    return run_dir(CONFIG) / "best.pth"

def finetuned_dets():
    hits = sorted(DET.glob(f"{EVAL_GAME}__rfdetr-small-ft-*"))
    assert hits, f"no fine-tuned detections in {DET}"
    return hits[-1]

(REPORTS / "logs").mkdir(parents=True, exist_ok=True)
drive.mount("/content/drive")
restore()
start_sync()
try:
    step("train", [PY, "-m", "src.train", "--config", CONFIG], "runs")
    if EVAL_GAME in GAMES:
        step("predict_stock", [PY, "-m", "src.detector.predict", "--weights", "stock", "--game", EVAL_GAME, "--config", CONFIG], "detections")
        step("predict_finetuned", lambda: [PY, "-m", "src.detector.predict", "--weights", finetuned_weights(), "--game", EVAL_GAME, "--config", CONFIG],
             "detections", needs=("train",))
        step("evaluate", lambda: [PY, "-m", "src.detector.evaluate", "--game", EVAL_GAME,
                                  "--stock", next(iter(DET.glob(f"{EVAL_GAME}__rfdetr-small-stock*"))), "--finetuned", finetuned_dets(),
                                  "--out", REPORTS / "detector" / f"{EVAL_GAME}.md"], needs=("predict_stock", "predict_finetuned"))
finally:
    package()

if MISSING:
    print("WARNING: this run did NOT use", MISSING)
if failures:
    raise RuntimeError(f"Failed steps: {failures}. Finished work is in Drive. Choose Run all to retry only what is missing.")'''

AGAIN = '''# Only needed if the run cell above was interrupted: rebuilds the zips from what is on disk and downloads them again.
package()'''

# Scores the untrained ViTSTR on the test game. evaluate.classifier only reads fine-tuned checkpoints, so this is separate.
# argv: recognizer config, directory holding probs_stock.npy, probs_stock_ids.csv and free_stock.csv.
STOCK_EVAL = '''import json, sys
import numpy as np, pandas as pd
from src.core.config import load_config
from src.dataio.store import crops_root, load_manifest
from src.evaluate import classifier as C, metrics as M

cfg, d = load_config(sys.argv[1]), sys.argv[2]
ids = pd.read_csv(f"{d}/probs_stock_ids.csv", dtype=str).crop_id.to_numpy()
raw = pd.DataFrame(np.load(f"{d}/probs_stock.npy"), index=ids)
reads = pd.read_csv(f"{d}/free_stock.csv", dtype=str, keep_default_na=False)
man = load_manifest(crops_root(cfg.crops), list(cfg.games.test))
man = man[(man.status == "ok") & (man.pose_min_conf >= cfg.filters.min_pose_conf) & (man.offset >= 0)].reset_index(drop=True)
probs, allowed = raw.loc[man.crop_id].to_numpy(), C.roster_mask(man.bib)
res = {"n": len(man), "scores": C.evaluate_subset(man, probs, allowed, None), "free_reads": M.free_read_stats(man, reads, allowed)}
json.dump(C._clean(res), open(f"{d}/stock_metrics.json", "w"), indent=1)
s = res["scores"]
print(f"stock ViTSTR on {len(man)} crops: 9-way {s['acc_9way']:.3f}, roster-limited 3-way {s['acc_team']:.3f}, free read {res['free_reads']}")'''


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


def fill(text: str, **kw: str) -> str:
    for k, v in kw.items():
        text = text.replace(f"@{k}@", v)
    return text


def build_notebook(name: str, bundle: str, version: str) -> dict:
    detector = name == "train_detector"
    run = RUN_DETECTOR if detector else fill(RUN_CLASSIFIER, STOCK_EVAL=repr(STOCK_EVAL))
    flags = "" if detector else ("# False skips the DINOv2 and ResNet-34 baselines, which are optional and come last\n"
                                 "RUN_BASELINES = True\n")
    setup = fill(SETUP, NOTEBOOK=name, CODE_VERSION=version, DETECTOR=str(detector), FLAGS=flags, GAMES=repr(GAMES))
    cells = [setup, DRIVE, fill(UNPACK, BUNDLE=bundle), fill(INSTALL, EXCLUDE=repr(EXCLUDE_PACKAGES)),
             fill(DATA, BOXES=BOXES_FOLDER, COURT=COURT_FOLDER, REG=REGISTRATION_ID, ROSTER=ROSTER_ID,
                  VIDEO_IDS=json.dumps(VIDEO_IDS, indent=4)), HELPERS, run, AGAIN]
    intro = fill(INTRO_DETECTOR if detector else INTRO_CLASSIFIER, NOTEBOOK=name)
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": lines(intro)}, *map(code, cells)]}


def main() -> None:
    bundle, version, _ = make_bundle()
    for name in ("train_classifier", "train_detector"):
        out = OUT_DIR / f"{name}.ipynb"
        out.write_text(json.dumps(build_notebook(name, bundle, version), indent=1))
        print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB), code version {version}")


if __name__ == "__main__":
    main()
