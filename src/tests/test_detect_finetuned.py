"""Fine-tuned detector inference on any video (src/infer/detect_video.py) and its Colab notebook."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.paths import ROOT
from src.infer import detect_video as dv
from src.tests.fixtures import make_video
from src.tests.test_detector_predict import install_fake, read
from src.tests.test_notebooks import Colab

COLAB = ROOT / "colab"
NB = COLAB / "detect_finetuned_videos.ipynb"


@pytest.fixture
def video(tmp_path, monkeypatch):
    monkeypatch.setenv("SRC_OUTPUTS", str(tmp_path / "outputs"))
    p = make_video(tmp_path / "IMG_0114.mkv", 20, 160, 90)
    monkeypatch.setattr(dv, "probe", lambda p: (1280, 720))  # a challenge video is already 720p
    return p


def weights(tmp_path):
    w = tmp_path / "best.pth"
    w.write_bytes(b"abc")
    return str(w)


def test_every_frame_in_the_tracker_format_with_previews(tmp_path, video, monkeypatch):
    made = install_fake(monkeypatch, (0, 1))
    out = dv.detect_video(video, weights(tmp_path), previews=3)
    assert made[0].ckpt == weights(tmp_path)
    assert out.name == "IMG_0114__rfdetr-small-ft-ba7816bf__720p.txt"  # same naming as tools/run_detector.py
    d = read(out)
    assert sorted(d.frame.unique()) == list(range(1, 21)) and len(d) == 20  # every frame; class-1 decoy dropped
    assert out.read_text().splitlines()[0] == "1,-1,100.00,200.00,150.00,400.00,0.9000,-1,-1,-1"
    shots = sorted(p.name for p in (tmp_path / "outputs/previews").glob("*.jpg"))
    assert shots == [f"IMG_0114__rfdetr-small-ft-ba7816bf__f{f:06d}.jpg" for f in (1, 10, 20)]
    assert not list(out.parent.glob("*.tmp"))


def test_more_frames_than_packets_leaves_no_output(tmp_path, video, monkeypatch):
    install_fake(monkeypatch, (0, 1))
    monkeypatch.setattr(dv, "frame_count", lambda p: 10)
    with pytest.raises(SystemExit):
        dv.detect_video(video, weights(tmp_path))
    assert not list((tmp_path / "outputs/detections").glob("*"))


def test_challenge_videos_are_allowed_here_but_not_for_training(tmp_path, video, monkeypatch):
    from src.dataio.video import video_path
    install_fake(monkeypatch, (0, 1))
    assert dv.detect_video(video, weights(tmp_path), limit=2).exists()
    with pytest.raises(ValueError, match="challenge video"):
        video_path("IMG_0114")


@pytest.fixture(scope="module")
def builder():
    sys.path.insert(0, str(COLAB))
    try:
        spec = importlib.util.spec_from_file_location("build_detect_finetuned_notebook", COLAB / "build_detect_finetuned_notebook.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(COLAB))


def cells() -> list[str]:
    return ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]


def test_notebook_is_current_and_valid(builder):
    _, version = builder.make_bundle()
    text = "\n".join(cells())
    assert f'CODE_VERSION = "{version}"' in text, "detect_finetuned_videos.ipynb is stale: rebuild it"
    for c in cells():
        ast.parse(c)
    assert builder.WEIGHTS_SHA256 in text and "src.infer.detect_video" in text and '"MPLBACKEND": "agg"' in text
    assert "--group" in text and "python-doctr" in text and "rf-groundingdino" in text
    assert list(builder.VIDEOS)[:3] == ["IMG_0114.MP4", "IMG_0115.MP4", "IMG_0108.MOV"] and len(builder.VIDEOS) == 10
    assert "src/infer/detect_video.py" in builder.infer_bundle_files()


def test_training_notebooks_do_not_bundle_the_inference_script():
    """Keeps their code version, which names the Drive folder a training run resumes from."""
    spec = importlib.util.spec_from_file_location("bsn", COLAB / "build_src_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not [n for n in mod.bundle_files() if n.startswith("src/infer/")]


def test_run_cell_saves_each_video_skips_finished_ones_and_downloads_one_zip(tmp_path, monkeypatch):
    colab = Colab()
    colab.install(monkeypatch)
    run = cells()[-1]
    root, outputs, cmds = tmp_path / "drive/tracking_results", tmp_path / "repo/outputs", []
    sources = {"IMG_0114.MP4": tmp_path / "v/IMG_0114.MP4", "IMG_0108.MOV": tmp_path / "v/IMG_0108.MOV"}
    for p in sources.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"v" * 3)
    model = "rfdetr-small-ft-471b3fff"

    def sh(cmd, cwd=None, log=None):
        cmd = list(map(str, cmd))
        cmds.append(cmd)
        Path(log).write_text("log\n")
        stem = Path(cmd[cmd.index("--video") + 1]).stem
        if stem == "IMG_0108":
            raise RuntimeError("decode failed")
        (outputs / "detections").mkdir(parents=True, exist_ok=True)
        (outputs / "detections" / f"{stem}__{model}__720p.txt").write_text("1,-1,0,0,1,1,.9,-1,-1,-1\n")
        (outputs / "previews").mkdir(parents=True, exist_ok=True)
        (outputs / "previews" / f"{stem}__{model}__f000001.jpg").write_bytes(b"j")

    import shutil
    import time
    ns = {"os": os, "shutil": shutil, "time": time, "zipfile": zipfile, "Path": Path, "sh": sh, "PY": "py",
          "VIDEOS": {n: 3 for n in sources}, "SOURCES": dict(sources), "problems": {"IMG_0115.MP4": "not in the folder."},
          "RESULTS_ROOT": root, "OUTPUTS": outputs, "CODE_VERSION": "v1", "MODEL": model, "PREVIEWS": 8,
          "WEIGHTS": tmp_path / "best.pth", "drive": colab.drive, "files": colab.files}
    monkeypatch.setattr(Path, "unlink", Path.unlink)  # the run removes its local video copies under /content
    run = run.replace('Path("/content") / name', 'tmp_path / name').replace('Path(f"/content/', 'Path(f"{tmp_path}/')
    ns["tmp_path"] = tmp_path
    with pytest.raises(RuntimeError, match="IMG_0115.MP4: not in the folder") as e:
        exec(run, ns)
    assert "IMG_0108.MOV: decode failed" in str(e.value)
    done = root / "IMG_0114/v1"
    assert (done / f"detections/IMG_0114__{model}__720p.txt").exists() and (done / "done/detection_ft").exists()
    assert (done / f"previews/IMG_0114__{model}__f000001.jpg").exists()
    assert not (root / "IMG_0108/v1/done/detection_ft").exists() and (root / "IMG_0108/v1/reports/detector_log.txt").exists()
    with zipfile.ZipFile(tmp_path / "finetuned_detections_v1.zip") as z:
        assert f"IMG_0114/detections/IMG_0114__{model}__720p.txt" in z.namelist()
    assert [e[0] for e in colab.events] == ["flush", "download"]

    n = len(cmds)
    with pytest.raises(RuntimeError):
        exec(run, ns)
    assert [Path(c[c.index("--video") + 1]).stem for c in cmds[n:]] == ["IMG_0108"]  # finished video skipped


# --- jersey numbers per detection (src/infer/jnr_video.py) and colab/jnr_all_videos.ipynb ---

from src.core.constants import NUMBERS  # noqa: E402
from src.infer import jnr_video as jv  # noqa: E402


class FakePose:
    def __init__(self):
        self.calls = 0

    def __call__(self, frame_rgb, boxes_xywh):
        self.calls += 1
        assert frame_rgb.shape == (720, 1280, 3)
        out = np.zeros((len(boxes_xywh), 17, 3), np.float32)
        for i, (x, y, w, h) in enumerate(boxes_xywh):
            for k, (fx, fy) in {5: (.75, .25), 6: (.25, .25), 11: (.7, .55), 12: (.3, .55)}.items():
                out[i, k] = (x + fx * w, y + fy * h, .9)
        return out


class FakeReaders:
    def __init__(self):
        self.batches = []

    def __call__(self, crops):
        self.batches.append(len(crops))
        p = np.zeros((len(crops), len(NUMBERS)), np.float32)
        p[:, NUMBERS.index("9")] = 1
        return np.full(len(crops), .8, np.float32), p, ["9"] * len(crops)


def test_jnr_rows_join_back_to_detections(tmp_path, video, monkeypatch):
    monkeypatch.setattr(jv, "probe", lambda p: (1280, 720))
    dets = tmp_path / "IMG_0114__rfdetr-small-ft-ba7816bf__720p.txt"
    lines = []
    for f in range(1, 21):
        lines.append(f"{f},-1,100,100,80,240,0.9,-1,-1,-1")   # a player
        lines.append(f"{f},-1,600,200,60,180,0.2,-1,-1,-1")   # below --min-conf
        lines.append(f"{f},-1,900,100,0.5,0.5,0.8,-1,-1,-1")  # degenerate box
    dets.write_text("\n".join(lines) + "\n")
    readers, pose = FakeReaders(), FakePose()
    out = jv.jnr_video(video, dets, readers, pose, every=2, min_conf=0.3, batch=4)
    assert out.name == "IMG_0114__rfdetr-small-ft-ba7816bf__720p__jnr.csv.gz"
    d = pd.read_csv(out, dtype={"read": str})
    assert list(d.columns) == jv.COLUMNS
    assert sorted(d.frame.unique()) == list(range(1, 21, 2)) and len(d) == 20  # every 2nd frame, low-conf dropped
    ok, bad = d[d.status == "ok"], d[d.status == "degenerate"]
    assert len(ok) == 10 and len(bad) == 10
    assert (ok.det_row % 3 == 0).all() and (bad.det_row % 3 == 2).all()  # det_row is the line in the detection file
    assert (ok.p_9 == 1).all() and np.allclose(ok.elig, .8) and (ok.read == "9").all()
    assert bad.elig.isna().all() and bad.p_9.isna().all()
    assert sum(readers.batches) == 10 and max(readers.batches) <= 5 and pose.calls == 10


def test_jnr_rejects_detections_beyond_the_video(tmp_path, video, monkeypatch):
    monkeypatch.setattr(jv, "probe", lambda p: (1280, 720))
    dets = tmp_path / "d.txt"
    dets.write_text("1,-1,100,100,80,240,0.9,-1,-1,-1\n500,-1,100,100,80,240,0.9,-1,-1,-1\n")
    with pytest.raises(SystemExit, match="detections reach frame 500"):
        jv.jnr_video(video, dets, FakeReaders(), FakePose())


@pytest.fixture(scope="module")
def jnr_builder():
    sys.path.insert(0, str(COLAB))
    try:
        spec = importlib.util.spec_from_file_location("build_jnr_notebook", COLAB / "build_jnr_notebook.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(COLAB))


def jnr_cells() -> list[str]:
    return ["".join(c["source"]) for c in json.loads((COLAB / "jnr_all_videos.ipynb").read_text())["cells"]
            if c["cell_type"] == "code"]


def test_jnr_notebook_is_current_and_valid(jnr_builder):
    text = "\n".join(jnr_cells())
    assert f'CODE_VERSION = "{jnr_builder.make_bundle()[1]}"' in text, "jnr_all_videos.ipynb is stale: rebuild it"
    for c in jnr_cells():
        ast.parse(c)
    for digest in (jnr_builder.ELIG_SHA256, jnr_builder.VITSTR_SHA256, jnr_builder.WEIGHTS_SHA256):
        assert digest in text
    assert "src.infer.jnr_video" in text and "src.infer.detect_video" in text and '"MPLBACKEND": "agg"' in text


def test_jnr_run_cell_reuses_detections_and_detects_when_missing(tmp_path, monkeypatch):
    import shutil
    import time
    colab = Colab()
    colab.install(monkeypatch)
    model = "rfdetr-small-ft-471b3fff"
    root, outputs, cmds = tmp_path / "drive/tracking_results", tmp_path / "repo/outputs", []
    old = root / "IMG_0108/2fb1eb84/detections"
    old.mkdir(parents=True)
    (old / f"IMG_0108__{model}__720p.txt").write_text("1,-1,0,0,1,1,.9,-1,-1,-1\n")
    sources = {"IMG_0114.MP4": tmp_path / "v/IMG_0114.MP4", "IMG_0108.MOV": tmp_path / "v/IMG_0108.MOV"}
    for p in sources.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"v" * 3)

    def sh(cmd, cwd=None, log=None):
        cmd = list(map(str, cmd))
        cmds.append(cmd)
        Path(log).write_text("log\n")
        stem = Path(cmd[cmd.index("--video") + 1]).stem
        if cmd[2] == "src.infer.detect_video":
            (outputs / "detections").mkdir(parents=True, exist_ok=True)
            (outputs / "detections" / f"{stem}__{model}__720p.txt").write_text("1,-1,0,0,1,1,.9,-1,-1,-1\n")
        else:
            assert Path(cmd[cmd.index("--dets") + 1]).exists()
            (outputs / "jnr").mkdir(parents=True, exist_ok=True)
            (outputs / "jnr" / f"{stem}__{model}__720p__jnr.csv.gz").write_bytes(b"z")

    ns = {"os": os, "shutil": shutil, "time": time, "zipfile": zipfile, "Path": Path, "sh": sh, "PY": "py",
          "VIDEOS": {n: 3 for n in sources}, "SOURCES": dict(sources), "problems": {}, "RESULTS_ROOT": root,
          "OUTPUTS": outputs, "CODE_VERSION": "v2", "MODEL": model, "WEIGHTS": "w", "ELIG": "e", "VITSTR": "t",
          "WEIGHTS_USED": {"eligibility.pt": {"sha256": "aaaa", "expected": True}},
          "sha256": lambda p: "d" * 64, "drive": colab.drive, "files": colab.files}
    run = jnr_cells()[-1].replace('Path("/content") / name', 'Path(TMP) / name').replace('Path(f"/content/', 'Path(f"{TMP}/')
    ns["TMP"] = str(tmp_path)
    exec(run, ns)
    mods = [(c[2], Path(c[c.index("--video") + 1]).stem) for c in cmds]
    assert mods == [("src.infer.detect_video", "IMG_0114"), ("src.infer.jnr_video", "IMG_0114"),
                    ("src.infer.jnr_video", "IMG_0108")]  # IMG_0108 reused its detections from the earlier run
    assert (root / f"IMG_0114/v2/detections/IMG_0114__{model}__720p.txt").exists()
    for stem in ("IMG_0114", "IMG_0108"):
        assert (root / f"{stem}/v2/jnr/{stem}__{model}__720p__jnr.csv.gz").exists() and (root / f"{stem}/v2/done/jnr").exists()
    with zipfile.ZipFile(tmp_path / "jnr_v2.zip") as z:
        assert f"IMG_0114/jnr/IMG_0114__{model}__720p__jnr.csv.gz" in z.namelist()
    assert (root / "IMG_0108/v2/reports/weights_used.json").exists()
    # reused detections are kept with this run's results, with where they came from
    assert (root / f"IMG_0108/v2/detections/IMG_0108__{model}__720p.txt").exists()
    assert str(old) in (root / "IMG_0108/v2/reports/detections_source.json").read_text()
    with zipfile.ZipFile(tmp_path / "jnr_v2.zip") as z:
        assert f"IMG_0108/detections/IMG_0108__{model}__720p.txt" in z.namelist()
    n = len(cmds)
    exec(run, ns)
    assert len(cmds) == n  # both finished: nothing reruns
    ns["WEIGHTS_USED"] = {"eligibility.pt": {"sha256": "bbbb", "expected": True}}
    exec(run, ns)  # other weights now: finished results are not trusted, both videos redo their jersey reads
    assert [c[2] for c in cmds[n:]] == ["src.infer.jnr_video", "src.infer.jnr_video"]
