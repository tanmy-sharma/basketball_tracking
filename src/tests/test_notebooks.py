"""Generated Colab notebooks: structure, bundle contents, install flags and the step/save/package logic, run offline."""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pytest

from src.core.config import load_config
from src.core.constants import FORBIDDEN_MODULES
from src.core.paths import ROOT

COLAB = ROOT / "colab"
NAMES = ("train_classifier", "train_detector")
FIXED_IDS = ["1e0r3kqSBbXVCwjDj4hh3_K7Oh_KKsTa7", "1yaAoPXn-aTLwmb-LasGMdKdyXff2TndK", "1vyVrNHIOJ9Fs3x7V5__XDndjJCDOR_17",
             "1rzzWnJyLt-ZAQvuYwQY7tWw7LalYLIiY", "1Ad2OB08gQsDHQEHleC7Wz_2VtmDtcyPe", "1rBMShAVMsnU38_1YQKcykIg1f2k52bdp",
             "1yNscs0JoERtgtt43oSGlO2SJebhtXMOJ", "1g6kidSDP4mo16dLsDWMeZXKNyIrqjCsD",
             "1ONKvnUcfPG33mB9aYJBmj6aZMP_3QXvb", "1JHXE8su46m9nRweCq0WonmPnNQhBbVwg"]


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_src_notebooks", COLAB / "build_src_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(name: str) -> dict:
    return json.loads((COLAB / f"{name}.ipynb").read_text())


def code_cells(name: str) -> list[str]:
    return ["".join(c["source"]) for c in load(name)["cells"] if c["cell_type"] == "code"]


def cell(name: str, needle: str) -> str:
    hits = [c for c in code_cells(name) if needle in c]
    assert hits, f"{name}: no cell contains {needle!r}"
    return hits[0]


@pytest.mark.parametrize("name", NAMES)
def test_valid_json_and_parseable_cells(name):
    nb = load(name)
    assert nb["nbformat"] == 4 and nb["cells"][0]["cell_type"] == "markdown"
    for c in code_cells(name):
        tree = ast.parse(c)
        for node in ast.walk(tree):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            assert not [m for m in mods if m.split(".")[0] in FORBIDDEN_MODULES]
    assert (COLAB / f"{name}.ipynb").stat().st_size < 1_000_000


def test_notebooks_are_not_stale(builder):
    """A source change needs `python colab/build_src_notebooks.py`; the embedded version tells."""
    _, version, _ = builder.make_bundle()
    for name in NAMES:
        assert f'CODE_VERSION = "{version}"' in cell(name, "CODE_VERSION ="), f"{name}.ipynb is stale: rebuild it"


def test_bundle_listing_is_only_code_configs_and_lock(builder):
    names = builder.bundle_files()
    assert {"pyproject.toml", "uv.lock", "tools/eval_detections.py", "src/train.py", "src/__init__.py",
            "src/configs/classifier/weak_folds3.yaml", "src/configs/smoke/classifier_mac.yaml"} <= set(names)
    assert any(n.startswith("tools/trackid/") for n in names)
    for n in names:
        p = Path(n)
        assert p.suffix in (".py", ".yaml", ".toml", ".lock"), n
        assert p.parts[0] in ("src", "tools", "pyproject.toml", "uv.lock"), n
        assert p.parts[:2] not in (("src", "tests"), ("src", "tracker")), n
        assert not re.search(r"IMG_\d{4}|\.mov|\.mp4|\.txt|\.csv|\.npy|\.bin|data/|outputs/|docs/", n, re.I), n
    assert not [n for n in names if n.startswith("tools/") and n not in
                ("tools/eval_detections.py", *[n for n in names if n.startswith("tools/trackid/")])]


def test_bundle_tar_contents_and_import(builder, tmp_path):
    b64, version, meta = builder.make_bundle()
    import base64
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64))) as tar:
        members = [m.name for m in tar.getmembers()]
        tar.extractall(tmp_path, filter="data")
    assert sorted(members) == sorted([*builder.bundle_files(), "CODE_VERSION.json"])
    cv = json.loads((tmp_path / "CODE_VERSION.json").read_text())
    assert cv["bundle_sha256"][:8] == version and set(cv) == {"git_hash", "dirty", "bundle_sha256"} == set(meta)
    r = subprocess.run([sys.executable, "-c", "import src.core.config, src.train, src.recognizer.predict; "
                        "from src.core.run import code_version; print(code_version()['bundle_sha256'])"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.strip() == cv["bundle_sha256"]
    assert load_config(tmp_path / "src/configs/classifier/weak_folds3.yaml").weak.name == "folds3"


@pytest.mark.parametrize("name", NAMES)
def test_ids_and_data_policy(name):
    data = cell(name, "gdown.download_folder")
    text = data + cell(name, "BUNDLE = ").split("BUNDLE = ")[0]
    for i in FIXED_IDS:
        assert i in data, i
    old = (COLAB / "build_notebook.py").read_text()
    for i in re.findall(r'"(1CDx1jVsYHs9CdvF3OC63gRk31WRvXEul|1OL8kBVDOE3n6FK3swkc9Ic2DIzq9dzhH)"', old):
        assert i in data
    assert re.search(r"IMG_0104.*IMG_0111", cell(name, "GAMES = "), re.S)
    assert text.index("in_drive(game)") < text.index("gdown.download(id=VIDEO_IDS[game]")  # runtime disk, Drive, then gdown
    assert "My Drive/tracking_videos" in data and "n < last" in data and "-count_packets" in data
    assert "IMG_0114" not in data and "IMG_0115" not in data


@pytest.mark.parametrize("name,detector", [("train_classifier", False), ("train_detector", True)])
def test_install_command(name, detector):
    assert f"DETECTOR = {detector}" in cell(name, "DETECTOR =")
    cmds: list[list[str]] = []
    ns = {"sh": lambda c, **kw: cmds.append(list(map(str, c))), "sys": sys, "PY": "py", "DETECTOR": detector}
    exec(cell(name, '"sync"'), ns)
    sync = next(c for c in cmds if "sync" in c)
    for flag in ("--frozen", "--no-dev"):
        assert flag in sync
    assert sync[sync.index("--python") + 1] == "3.12"
    excluded = [sync[i + 1] for i, a in enumerate(sync) if a == "--no-install-package"]
    assert excluded == ["python-doctr", "rf-groundingdino"]
    assert ("--group" in sync and sync[sync.index("--group") + 1] == "rfdetr") == detector
    assert any("import rfdetr" in " ".join(c) for c in cmds) == detector


@pytest.mark.parametrize("name", NAMES)
def test_intro_and_runtime_check(name):
    intro = "".join(load(name)["cells"][0]["source"])
    for needle in ("Apache 2.0", "DINOv2", "ViTPose", "ViTSTR", "ResNet-34", "RF-DETR Small", "Allow multiple downloads",
                   "no separate licence", "A100", "reports_<ver>.zip"):
        assert needle in intro or needle.lower() in intro.lower(), needle
    assert "mmpose" not in intro.lower()
    setup = cell(name, "nvidia-smi")
    assert '"A100" not in gpu.stdout' in setup and "WARNING" in setup


@pytest.mark.parametrize("name", NAMES)
def test_final_cell_downloads_in_finally(name):
    run = cell(name, "if failures:")
    tree = ast.parse(run)
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
    assert any("package" in ast.unparse(n.finalbody) for n in tries)
    helpers = cell(name, "def package")
    assert "files.download" in helpers and "flush_and_unmount" in helpers
    assert helpers.index("reports_zip, weights_zip") < helpers.index("for z in (reports_zip, weights_zip):\n        try:\n            files.download")
    assert helpers.index("flush_and_unmount") < helpers.index("files.download")
    assert "except Exception" in helpers[helpers.index("files.download") - 80:] and "Get it from Drive" in helpers
    assert "tracking_results/src_{NOTEBOOK}/{CODE_VERSION}" in cell(name, "tracking_results")
    assert "package()" in code_cells(name)[-1]


def modules_in(name: str) -> set[str]:
    run = cell(name, "if failures:")
    return set(re.findall(r'"(src\.[a-z_.]+)"', run))


@pytest.mark.parametrize("name", NAMES)
def test_every_command_targets_an_existing_module_and_config(name):
    mods = modules_in(name)
    assert mods
    for m in mods:
        assert importlib.util.find_spec(m) is not None, m
    run = cell(name, "if failures:")
    for cfg in re.findall(r"src/configs/[\w/]+\.yaml", run):
        assert (ROOT / cfg).exists(), cfg
    for var, sub in (("VIT", "recognizer"), ("DINO", "classifier")):
        m = re.search(rf"^{var} = (\[.*\])$", run, re.M)
        if m:
            for n in ast.literal_eval(m.group(1)):
                assert (ROOT / f"src/configs/{sub}/{n}.yaml").exists(), n
    assert re.search(r'CONFIG = "src/configs/detector/rfdetr_small.yaml"', run) or name == "train_classifier"


def test_classifier_step_order():
    run = cell("train_classifier", "if failures:")
    order = ["src.crops.extract", "src.weak.folds", "eligibility.yaml", "src.weak.finalize", "src.weak.qa", "--stock",
            "for name in VIT", "if RUN_BASELINES", "evaluate()"]
    pos = [run.index(s, run.index("try:")) for s in order]
    assert pos == sorted(pos)
    assert re.search(r'^VIT = \["vitstr_all", "vitstr_eligible", "vitstr_xcolor_no_yellow", "vitstr_xcolor_no_green"\]$', run, re.M)
    assert re.search(r'^DINO = \["number_all", "number_eligible", "xcolor_no_yellow", "xcolor_no_green", "number_eligible_resnet34"\]$', run, re.M)
    setup = cell("train_classifier", "nvidia-smi")
    assert "RUN_BASELINES = True" in setup and setup.index("RUN_BASELINES") < setup.index("def sh")
    assert "RUN_BASELINES" not in cell("train_detector", "nvidia-smi")


class Colab:
    """Stand-in for google.colab that records what the notebook does."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.drive = types.SimpleNamespace(mount=lambda p: self.events.append(("mount", p)),
                                           flush_and_unmount=lambda: self.events.append(("flush", "")))
        self.files = types.SimpleNamespace(download=lambda p: self.events.append(("download", Path(p).name)))

    def install(self, monkeypatch) -> None:
        pkg, colab = types.ModuleType("google"), types.ModuleType("google.colab")
        colab.drive, colab.files = self.drive, self.files
        pkg.colab = colab
        monkeypatch.setitem(sys.modules, "google", pkg)
        monkeypatch.setitem(sys.modules, "google.colab", colab)


def notebook_ns(name: str, tmp_path: Path, monkeypatch, fail: set[str] = frozenset(), baselines: bool = True):
    """Run the helper and run cells with `sh` replaced by a recorder that fakes each command's outputs."""
    import os
    import subprocess as sp
    import time
    colab = Colab()
    colab.install(monkeypatch)
    repo = tmp_path / "repo"
    cmds: list[list[str]] = []

    def sh(cmd, cwd=None, log=None):
        cmd = list(map(str, cmd))
        cmds.append(cmd)
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            Path(log).write_text("log\n")
        if cmd[2:3] and cmd[2] in fail:
            raise RuntimeError("boom")
        if "--best" in cmd:  # free read -> a csv in --out
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / f"free_{Path(cmd[cmd.index('--best') + 1]).parent.name}.csv").write_text("crop_id,text\n")
        if cmd[2:3] == ["src.detector.predict"]:
            tag = "stock" if cmd[cmd.index("--weights") + 1] == "stock" else "ft-abcd1234"
            d = repo / "outputs/detections"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"IMG_0108__rfdetr-small-{tag}__720p__every5.txt").write_text("1,-1,0,0,1,1,.9,-1,-1,-1\n")
        if cmd[2:3] == ["src.train"]:  # a trained run leaves weights next to its config
            d = repo / "outputs/runs" / (Path(cmd[-1]).stem + "__abc")
            d.mkdir(parents=True, exist_ok=True)
            (d / "best.pt").write_bytes(b"w" * 10)
            (d / "last.pt").write_bytes(b"w" * 30)
            (d / "manifest.json").write_text("{}")

    ns = {"os": os, "time": time, "subprocess": sp, "sys": sys, "Path": Path, "NOTEBOOK": name, "CODE_VERSION": "v1",
          "DETECTOR": name == "train_detector", "RUN_BASELINES": baselines, "REPO": repo, "PY": "py",
          "OUTPUTS": repo / "outputs", "REPORTS": repo / "outputs/reports", "RESULTS": tmp_path / "drive/results",
          "ZIP_DIR": tmp_path / "zips", "GAMES": ["IMG_0104", "IMG_0105", "IMG_0106", "IMG_0107", "IMG_0108", "IMG_0109",
                                                  "IMG_0110", "IMG_0111"], "sh": sh, "MISSING": []}
    (tmp_path / "zips").mkdir()
    exec(cell(name, "def package"), ns)
    ns["run_dir"] = lambda cfg: repo / "outputs/runs" / (Path(cfg).stem + "__abc")
    return ns, cmds, colab


def test_helpers_save_restore_step_and_package(tmp_path, monkeypatch):
    ns, cmds, colab = notebook_ns("train_classifier", tmp_path, monkeypatch)
    out = ns["OUTPUTS"]
    for rel, n in {"runs/a__1/best.pt": 5, "runs/a__1/last.pt": 9, "runs/a__1/manifest.json": 2, "runs/fold_x__2/best.pt": 5,
                   "crops/h/IMG_0104/crops.bin": 7, "detector_dataset/h/train/i.jpg": 3, "reports/r.md": 1,
                   "runs/d__3/rfdetr/checkpoint_best_total.pth": 4, "runs/d__3/best.pth": 4}.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(b"x" * n)
    ns["REPORTS"].mkdir(parents=True, exist_ok=True)
    ns["save"]("runs", "crops", "reports", "detector_dataset")
    saved = {p.relative_to(ns["RESULTS"]).as_posix() for p in ns["RESULTS"].rglob("*") if p.is_file()}
    assert saved == {"runs/a__1/best.pt", "runs/a__1/manifest.json", "runs/fold_x__2/best.pt", "crops/h/IMG_0104/crops.bin",
                     "reports/r.md", "runs/d__3/best.pth"}  # no last.pt, no RF-DETR intermediates, no dataset jpgs
    (out / "reports/r.md").unlink()
    ns["restore"]()
    assert (out / "reports/r.md").read_text() == "x"

    assert ns["step"]("ok", ["py", "-m", "src.fine"], "reports") is True and ns["done"]("ok")
    n = len(cmds)
    assert ns["step"]("ok", ["py"], "reports") is True and len(cmds) == n  # done -> skipped
    assert ns["step"]("later", ["py", "boom"], needs=("missing",)) is False and len(cmds) == n  # prerequisite missing
    ns["sh"] = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    assert ns["step"]("bad", ["x"]) is False and ns["failures"] == ["bad"] and not ns["done"]("bad")
    rows = (ns["REPORTS"] / "step_times.csv").read_text().splitlines()
    assert [r.split(",")[0] for r in rows] == ["ok", "bad"] and rows[1].endswith("failed")
    assert (ns["RESULTS"] / "done/ok").exists() and not (ns["RESULTS"] / "done/bad").exists()

    ns["package"]()
    rz, wz = (tmp_path / "zips" / f"{k}_v1.zip" for k in ("reports", "weights"))
    with zipfile.ZipFile(rz) as z:
        rnames = set(z.namelist())
    with zipfile.ZipFile(wz) as z:
        wnames = set(z.namelist())
    assert "runs/a__1/manifest.json" in rnames and "reports/step_times.csv" in rnames and "done/ok" in rnames
    assert not [n for n in rnames if n.endswith((".pt", ".pth", "crops.bin", "last.pt", ".jpg")) or "checkpoint" in n]
    assert wnames == {"runs/a__1/best.pt", "runs/d__3/best.pth"}  # fold models stay out
    assert [e for e in colab.events if e[0] != "mount"] == [("flush", ""), ("download", rz.name), ("download", wz.name)]
    assert (ns["RESULTS"] / rz.name).exists() and (ns["RESULTS"] / wz.name).exists()


def _age(path, seconds=60):
    t = path.stat().st_mtime - seconds
    os.utime(path, (t, t))


def test_sync_keeps_a_resume_point_on_drive(tmp_path, monkeypatch):
    ns, _, _ = notebook_ns("train_detector", tmp_path, monkeypatch)
    run = ns["OUTPUTS"] / "runs/d__1"
    drive = ns["RESULTS"] / "runs/d__1"
    (run / "rfdetr").mkdir(parents=True)
    for rel in ("rfdetr/last.ckpt", "rfdetr/checkpoint_best_regular.pth", "rfdetr/checkpoint_9.ckpt", "rfdetr/metrics.csv", "x.tmp"):
        (run / rel).write_bytes(b"a")
        _age(run / rel)
    (run / "rfdetr/too_new.txt").write_bytes(b"a")  # may still be being written

    assert ns["sync_runs"]() == 3
    assert sorted(p.name for p in drive.rglob("*") if p.is_file()) == ["checkpoint_best_regular.pth", "last.ckpt", "metrics.csv"]
    assert ns["sync_runs"]() == 0  # nothing changed

    (run / "rfdetr/last.ckpt").write_bytes(b"b")  # same size, new epoch: a size check alone would miss it
    _age(run / "rfdetr/last.ckpt")
    assert ns["sync_runs"]() == 1 and (drive / "rfdetr/last.ckpt").read_bytes() == b"b"

    (ns["OUTPUTS"] / "runs/d__1/rfdetr/last.ckpt").unlink()  # a new runtime gets the checkpoint back
    ns["restore"]()
    assert (run / "rfdetr/last.ckpt").read_bytes() == b"b"

    (run / "best.pth").write_bytes(b"w")
    (run / "done.json").write_text("{}")
    ns["save"]("runs")
    ns["sync_runs"]()  # finished: resume state leaves Drive, the weights stay
    assert sorted(p.name for p in drive.rglob("*") if p.is_file()) == ["best.pth", "done.json", "metrics.csv", "too_new.txt"]


def test_sync_copies_resume_files_last(tmp_path, monkeypatch):
    ns, _, _ = notebook_ns("train_classifier", tmp_path, monkeypatch)
    run = ns["OUTPUTS"] / "runs/a__1"
    run.mkdir(parents=True)
    for n in ("best.pt", "last.pt", "metrics.jsonl"):
        (run / n).write_bytes(b"a")
        _age(run / n)
    order, real = [], ns["_copy_checkpoint"]
    ns["_copy_checkpoint"] = lambda f, d: (order.append(f.name), real(f, d))[1]
    ns["sync_runs"]()
    assert order[-1] == "last.pt" and set(order) == {"best.pt", "last.pt", "metrics.jsonl"}


@pytest.mark.parametrize("name", NAMES)
def test_timer_runs_through_every_training_step(name, tmp_path, monkeypatch):
    run = cell(name, "\nstart_sync()\ntry:")
    assert run.index("restore()") < run.index("start_sync()") < run.index("try:")
    ns, _, _ = notebook_ns(name, tmp_path, monkeypatch)
    ns["OUTPUTS"].mkdir(parents=True)
    ns["start_sync"]()
    ns["package"]()  # must stop the timer before Drive is unmounted
    assert ns["_stop_sync"].is_set() and not ns["_sync_thread"].is_alive()


def test_step_end_save_replaces_a_same_size_older_copy(tmp_path, monkeypatch):
    """The timer copied best.pt mid-training; the final best.pt has the same size and must still reach Drive."""
    ns, _, _ = notebook_ns("train_classifier", tmp_path, monkeypatch)
    best, drive = ns["OUTPUTS"] / "runs/a__1/best.pt", ns["RESULTS"] / "runs/a__1/best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"epoch6")
    ns["save"]("runs")
    old = drive.stat().st_mtime - 60
    os.utime(drive, (old, old))
    best.write_bytes(b"epoch9")
    ns["save"]("runs")
    assert drive.read_bytes() == b"epoch9"


def test_restore_does_not_bring_back_old_zips(tmp_path, monkeypatch):
    ns, _, _ = notebook_ns("train_detector", tmp_path, monkeypatch)
    ns["RESULTS"].mkdir(parents=True)
    (ns["RESULTS"] / "reports_v1.zip").write_bytes(b"z")
    (ns["RESULTS"] / "done").mkdir()
    (ns["RESULTS"] / "done/train").write_bytes(b"")
    ns["restore"]()
    assert (ns["OUTPUTS"] / "done/train").exists() and not (ns["OUTPUTS"] / "reports_v1.zip").exists()


def test_package_accepts_a_file_dated_before_1980(tmp_path, monkeypatch):
    ns, _, _ = notebook_ns("train_detector", tmp_path, monkeypatch)
    ns["OUTPUTS"].mkdir(parents=True)
    meta = ns["REPO"] / "CODE_VERSION.json"  # the bundle writes this with mtime 0, which crashed the first real run
    meta.write_text("{}")
    os.utime(meta, (0, 0))
    ns["package"]()
    with zipfile.ZipFile(tmp_path / "zips/reports_v1.zip") as z:
        assert "CODE_VERSION.json" in z.namelist()


@pytest.mark.parametrize("name", NAMES)
def test_subprocesses_do_not_inherit_colabs_matplotlib_backend(name):
    assert '"MPLBACKEND": "agg"' in cell(name, "def sh(")


def test_download_failure_prints_drive_path(tmp_path, monkeypatch, capsys):
    ns, _, colab = notebook_ns("train_detector", tmp_path, monkeypatch)
    colab.files.download = lambda p: (_ for _ in ()).throw(RuntimeError("blocked"))
    ns["files"].download = colab.files.download
    ns["OUTPUTS"].mkdir(parents=True)
    ns["package"]()
    assert str(ns["RESULTS"] / "reports_v1.zip") in capsys.readouterr().out


@pytest.mark.parametrize("baselines", [True, False])
def test_classifier_run_cell_flow(tmp_path, monkeypatch, baselines):
    ns, cmds, colab = notebook_ns("train_classifier", tmp_path, monkeypatch, baselines=baselines)
    run = cell("train_classifier", "if failures:")
    exec(run, ns)
    mods = [c[2] for c in cmds if c[:2] == ["py", "-m"]]
    first = lambda m: mods.index(m)  # noqa: E731
    assert mods[:8] == ["src.crops.extract"] * 8
    assert first("src.weak.folds") < first("src.train") < first("src.weak.finalize") < first("src.weak.qa")
    trains = [c[-1].split("/")[-1][:-5] for c in cmds if c[2:3] == ["src.train"]]
    vit = ["vitstr_all", "vitstr_eligible", "vitstr_xcolor_no_yellow", "vitstr_xcolor_no_green"]
    dino = ["number_all", "number_eligible", "xcolor_no_yellow", "xcolor_no_green", "number_eligible_resnet34"]
    assert trains == ["eligibility", *vit, *(dino if baselines else [])]
    assert sum(c[2] == "src.recognizer.predict" and "--best" in c for c in cmds) == 4
    ev = next(c for c in cmds if c[2] == "src.evaluate.classifier")
    numbers = [ev[i + 1].split("=")[0] for i, a in enumerate(ev) if a == "--number"]
    assert numbers == vit + (dino if baselines else [])
    assert sum(a == "--free-reads" for a in ev) == 4 and "--eligibility" in ev and ev[ev.index("--game") + 1] == "IMG_0108"
    assert cmds[-1] == ev and ns["failures"] == []
    assert [e[0] for e in colab.events if e[0] != "mount"] == ["flush", "download", "download"]


def test_classifier_failure_still_packages_and_continues(tmp_path, monkeypatch):
    ns, cmds, colab = notebook_ns("train_classifier", tmp_path, monkeypatch, fail={"src.weak.folds"})
    with pytest.raises(RuntimeError, match="weak_folds"):  # raised only after the packaging in `finally`
        exec(cell("train_classifier", "if failures:"), ns)
    assert ns["failures"] == ["weak_folds"]
    mods = [c[2] for c in cmds if c[:2] == ["py", "-m"]]
    assert "src.weak.finalize" not in mods and "src.weak.qa" not in mods  # dependants skipped
    trains = [c[-1].split("/")[-1][:-5] for c in cmds if c[2:3] == ["src.train"]]
    assert "eligibility" not in trains and "vitstr_eligible" not in trains and "vitstr_all" in trains
    assert "src.evaluate.classifier" in mods  # independent runs and the evaluation still happen
    assert [e[0] for e in colab.events if e[0] != "mount"] == ["flush", "download", "download"]
    assert (tmp_path / "zips/reports_v1.zip").exists()


def test_detector_run_cell_flow(tmp_path, monkeypatch):
    ns, cmds, colab = notebook_ns("train_detector", tmp_path, monkeypatch)
    exec(cell("train_detector", "if failures:"), ns)
    mods = [c[2] for c in cmds if c[:2] == ["py", "-m"]]
    assert mods == ["src.train", "src.detector.predict", "src.detector.predict", "src.detector.evaluate"]
    assert cmds[1][cmds[1].index("--weights") + 1] == "stock"
    assert cmds[2][cmds[2].index("--weights") + 1].endswith("rfdetr_small__abc/best.pth")
    ev = cmds[3]
    assert "stock" in ev[ev.index("--stock") + 1] and "-ft-" in ev[ev.index("--finetuned") + 1]
    assert ns["failures"] == []


ALL_GAMES = ["IMG_0104", "IMG_0105", "IMG_0106", "IMG_0107", "IMG_0108", "IMG_0109", "IMG_0110", "IMG_0111"]


def _configs_without(games: list[str], tmp_path: Path) -> Path:
    from src.core.drop_games import drop_games
    root = tmp_path / "configs"
    shutil.copytree(ROOT / "src/configs", root)
    drop_games(games, root)
    return root


@pytest.mark.parametrize("game", ALL_GAMES)
def test_every_config_still_loads_without_any_one_game(game, tmp_path):
    root = _configs_without([game], tmp_path)
    for p in root.rglob("*.yaml"):
        assert game not in p.read_text(), p
        if "smoke" not in p.parts and p != root / "recognizer/base.yaml":  # that base has no run_name of its own
            cfg = load_config(p)
            assert game not in cfg.games.train + cfg.games.test
    weak = load_config(root / "classifier/weak_folds3.yaml").weak
    assert len(weak.folds) == 3 and game not in [g for f in weak.folds for g in f]


def test_emptied_fold_is_removed_and_hash_changes(tmp_path):
    from src.core.config import config_hash
    root = _configs_without(["IMG_0104", "IMG_0109"], tmp_path)
    cfg = load_config(root / "classifier/weak_folds3.yaml")
    assert cfg.weak.folds == [["IMG_0105", "IMG_0110"], ["IMG_0106", "IMG_0107", "IMG_0111"]]
    assert config_hash(cfg) != config_hash(load_config(ROOT / "src/configs/classifier/weak_folds3.yaml"))
    full = load_config(root / "detector/rfdetr_small.yaml")
    assert full.games.train == ["IMG_0105", "IMG_0106", "IMG_0107", "IMG_0110", "IMG_0111"] and full.games.test == ["IMG_0108"]


def _data_cell(name, tmp_path, monkeypatch, drive_has, short=()):
    """Run the data cell with fake gdown and ffprobe. Annotations exist for all games; videos only where given."""
    calls = {"gdown": []}
    gd = types.ModuleType("gdown")
    gd.download_folder = lambda **k: None
    gd.download = lambda **k: calls["gdown"].append(k["id"])  # always fails, like Google's quota
    monkeypatch.setitem(sys.modules, "gdown", gd)
    repo = tmp_path / "repo"
    train = repo / "data/Training"
    for d in ("full player annotations/Image-space bounding boxes", "full player annotations/court position"):
        (train / d).mkdir(parents=True)
        for g in ALL_GAMES:
            (train / d / f"{g}.txt").write_text("100,1,0,0,1,1\n")
    (train / "court_registration.csv").write_text("x")
    (train / "games_roster.csv").write_text("x")
    videos = tmp_path / "drive/tracking_videos"
    videos.mkdir(parents=True)
    for g in drive_has:
        (videos / f"{g}.MOV").write_bytes(b"v")
    sub = types.SimpleNamespace(run=lambda cmd, **k: types.SimpleNamespace(
        stdout="5," if any(g in cmd[-1] for g in short) else "100,"))
    cmds = []
    ns = {"os": os, "subprocess": sub, "Path": Path, "REPO": repo, "GAMES": list(ALL_GAMES), "PY": "py", "CODE_VERSION": "v1",
          "DRIVE_VIDEOS": videos, "RESULTS": tmp_path / "drive/results/v1", "REPORTS": repo / "outputs/reports",
          "sh": lambda cmd, **k: cmds.append(list(map(str, cmd)))}
    exec(cell(name, "gdown.download_folder"), ns)
    return ns, cmds, calls


@pytest.mark.parametrize("name", NAMES)
def test_all_videos_present_changes_nothing(name, tmp_path, monkeypatch):
    ns, cmds, calls = _data_cell(name, tmp_path, monkeypatch, ALL_GAMES)
    assert ns["GAMES"] == ALL_GAMES and ns["MISSING"] == [] and cmds == [] and calls["gdown"] == []
    assert ns["RESULTS"].name == "v1"  # same folder, so a normal run resumes as before


@pytest.mark.parametrize("name", NAMES)
def test_missing_and_short_videos_do_not_stop_the_run(name, tmp_path, monkeypatch, capsys):
    have = [g for g in ALL_GAMES if g != "IMG_0105"]
    ns, cmds, calls = _data_cell(name, tmp_path, monkeypatch, have, short=["IMG_0107"])
    assert ns["MISSING"] == ["IMG_0105", "IMG_0107"]
    assert "IMG_0105" not in ns["GAMES"] and "IMG_0107" not in ns["GAMES"] and len(ns["GAMES"]) == 6
    assert cmds == [["py", "-m", "src.core.drop_games", "--games", "IMG_0105", "IMG_0107"]]
    assert ns["RESULTS"].name == "v1_without_0105_0107" and ns["RESULTS"].is_dir()
    assert not (ns["REPO"] / "data/Training/vidoes/IMG_0107.MOV").exists()  # the bad file is not kept
    assert "IMG_0105" in (ns["REPORTS"] / "missing_videos.txt").read_text()
    assert "CONTINUING WITHOUT 2 OF 8" in capsys.readouterr().out


@pytest.mark.parametrize("name", NAMES)
def test_only_no_training_video_at_all_stops_the_run(name, tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="No training video"):
        _data_cell(name, tmp_path, monkeypatch, ["IMG_0108"])


def test_detector_without_test_game_still_trains(tmp_path, monkeypatch, capsys):
    ns, cmds, _ = notebook_ns("train_detector", tmp_path, monkeypatch)
    ns["GAMES"] = [g for g in ALL_GAMES if g != "IMG_0108"]
    ns["MISSING"] = ["IMG_0108"]
    exec(cell("train_detector", "if failures:"), ns)
    assert [c[2] for c in cmds if c[:2] == ["py", "-m"]] == ["src.train"] and ns["failures"] == []
    assert "did NOT use ['IMG_0108']" in capsys.readouterr().out


def test_classifier_without_test_game_still_trains(tmp_path, monkeypatch):
    ns, cmds, _ = notebook_ns("train_classifier", tmp_path, monkeypatch)
    ns["GAMES"] = [g for g in ALL_GAMES if g != "IMG_0108"]
    exec(cell("train_classifier", "if failures:"), ns)
    mods = [c[2] for c in cmds if c[:2] == ["py", "-m"]]
    assert "src.train" in mods and "src.weak.folds" in mods and ns["failures"] == []
    assert "src.evaluate.classifier" not in mods and "src.recognizer.predict" not in mods


def test_undecodable_video_is_dropped_and_the_rest_continue(tmp_path, monkeypatch):
    ns, cmds, _ = notebook_ns("train_classifier", tmp_path, monkeypatch)
    fake = ns["sh"]

    def sh(cmd, **k):
        if "IMG_0105" in map(str, cmd) and "src.crops.extract" in map(str, cmd):
            raise RuntimeError("decoded too few frames")
        return fake(cmd, **k)
    ns["sh"] = sh
    with pytest.raises(RuntimeError, match="crops_IMG_0105"):  # still reported at the very end
        exec(cell("train_classifier", "if failures:"), ns)
    mods = [c[2] for c in cmds if c[:2] == ["py", "-m"]]
    drop = next(c for c in cmds if c[2] == "src.core.drop_games")
    assert drop[-1] == "IMG_0105" and cmds.index(drop) < mods.index("src.weak.folds")
    assert "src.weak.folds" in mods and "src.evaluate.classifier" in mods and ns["failures"] == ["crops_IMG_0105"]
    assert ns["MISSING"] == ["IMG_0105"]


def test_smoke_configs_load():
    for n, plugin in (("classifier_mac", "classifier"), ("recognizer_mac", "recognizer"), ("detector_mac", "detector")):
        cfg = load_config(ROOT / "src/configs/smoke" / f"{n}.yaml")
        assert cfg.plugin == plugin and cfg.games.train == ["IMG_0106"] and cfg.games.test == ["IMG_0108"]
        assert cfg.run_name == f"smoke_{n}"
    c = load_config(ROOT / "src/configs/smoke/classifier_mac.yaml")
    assert (c.crops.every, c.crops.limit_frames, c.split.guard_frames, c.train.batch_size, c.train.num_workers) == (30, 300, 0, 8, 0)
    assert [p.epochs for p in c.train.phases] == [1, 1]
    d = load_config(ROOT / "src/configs/smoke/detector_mac.yaml")
    assert d.device == "cpu" and d.detector.epochs == 1 and d.detector.batch_size == 2


def test_readme_discloses_and_reports_stock_baseline():
    text = (ROOT / "src/README.md").read_text()
    for needle in ("239 crops", "0.4%", "8.4%", "31.4%", "33.3%", "Apache 2.0", "mmpose", "build_src_notebooks.py",
                   "python -m src.train", "python -m src.weak.folds", "Camera shift", "Leakage"):
        assert needle in text, needle
