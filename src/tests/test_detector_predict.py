from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.core.paths import add_tools_to_path
from src.detector import evaluate as det_eval
from src.detector import predict
from src.tests.test_coco import detector_cfg, make_world

BATCHES: list[int] = []
KWARGS: list[dict] = []


class FakeDet:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.confidence, self.class_id = np.array(xyxy, float), np.array(conf, float), np.array(cls)

    def __len__(self) -> int:
        return len(self.class_id)


def install_fake(monkeypatch, class_ids: tuple[int, int]) -> list:
    """Fake RFDETRSmall: every image gets one box of the wanted class and one decoy of another class."""
    made = []
    want, decoy = class_ids

    class RFDETRSmall:
        def __init__(self, ckpt: str | None = None):
            self.ckpt = ckpt
            made.append(self)

        @classmethod
        def from_checkpoint(cls, path):
            return cls(str(path))

        def predict(self, images, threshold, include_source_image):
            BATCHES.append(len(images))
            KWARGS.append({"threshold": threshold, "include_source_image": include_source_image})
            assert images[0].dtype == np.uint8 and images[0].shape == (720, 1280, 3)
            return [FakeDet([[100, 200, 250, 600], [10, 10, 60, 90]], [0.9, 0.8], [want, decoy]) for _ in images]

    BATCHES.clear()
    KWARGS.clear()
    monkeypatch.setitem(sys.modules, "rfdetr", types.SimpleNamespace(RFDETRSmall=RFDETRSmall))
    return made


def read(path: Path):
    add_tools_to_path()
    import eval_detections
    return eval_detections.load_dets(path)


def test_stock_keeps_coco_person_id_and_scales_to_full_resolution(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch, games=["IMG_0106"])
    made = install_fake(monkeypatch, (1, 2))
    out = predict.predict_game("IMG_0106", "stock", 5, None, detector_cfg().detector)
    assert out.name == "IMG_0106__rfdetr-small-stock__720p__every5.txt" and out.parent.parent == tmp_path / "outputs"
    assert made[0].ckpt is None
    d = read(out)
    assert sorted(d.frame.unique()) == list(range(1, 61, 5)) and len(d) == 12
    assert d.iloc[0].to_dict() == {"frame": 1.0, "x": 300.0, "y": 600.0, "w": 450.0, "h": 1200.0, "conf": 0.9}
    assert BATCHES == [8, 4] and KWARGS[0] == {"threshold": 0.05, "include_source_image": False}
    assert out.read_text().splitlines()[0] == "1,-1,300.00,600.00,450.00,1200.00,0.9000,-1,-1,-1"
    assert not list(out.parent.glob("*.tmp"))


def test_finetuned_keeps_class_zero_and_names_file_by_weights_hash(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch, games=["IMG_0106"])
    made = install_fake(monkeypatch, (0, 1))
    w = tmp_path / "best.pth"
    w.write_bytes(b"abc")
    out = predict.predict_game("IMG_0106", str(w), 5, None, detector_cfg().detector)
    assert made[0].ckpt == str(w)
    assert out.name == "IMG_0106__rfdetr-small-ft-ba7816bf__720p__every5.txt"  # sha256("abc")[:8]
    assert len(read(out)) == 12  # one kept box per frame, the class-1 decoy is dropped


def test_limit_skips_frame_count_check(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch, games=["IMG_0106"])
    install_fake(monkeypatch, (1, 2))
    out = predict.predict_game("IMG_0106", "stock", 5, 3, detector_cfg().detector)
    assert sorted(read(out).frame.unique()) == [1, 6, 11]


def test_frame_count_mismatch_leaves_no_output(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch, games=["IMG_0106"])
    install_fake(monkeypatch, (1, 2))
    monkeypatch.setattr(predict, "frame_count", lambda p: 999)
    with pytest.raises(SystemExit):
        predict.predict_game("IMG_0106", "stock", 5, None, detector_cfg().detector)
    assert not list((tmp_path / "outputs" / "detections").glob("*"))


def test_challenge_video_refused(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch, games=["IMG_0106"])
    install_fake(monkeypatch, (1, 2))
    with pytest.raises(ValueError):
        predict.predict_game("IMG_0114", "stock", 5, None, detector_cfg().detector)


def gt_as_dets(path: Path, every: int, keep: int = 6) -> Path:
    add_tools_to_path()
    from trackid import io
    bb = io.load_bboxes("IMG_0108")
    bb = bb[(bb.frame % every == 1) & (bb.track_id <= keep)]
    with open(path, "w") as f:
        for r in bb.itertuples():
            f.write(f"{r.frame},-1,{r.x:.2f},{r.y:.2f},{r.w:.2f},{r.h:.2f},0.9,-1,-1,-1\n")
    return path


def test_evaluate_gt_as_detections(tmp_path, monkeypatch):
    make_world(tmp_path, monkeypatch, games=["IMG_0108"])
    perfect = gt_as_dets(tmp_path / "perfect.txt", 5)
    half = gt_as_dets(tmp_path / "half.txt", 5, keep=3)
    out, rows = det_eval.run("IMG_0108", {"stock": perfect, "ft": half}, out=tmp_path / "report.md")
    by = {r["name"]: r for r in rows}
    assert by["stock"]["recall"] == pytest.approx(1.0) and by["stock"]["precision"] == pytest.approx(1.0)
    assert by["stock"]["err_median"] < 0.1 and by["stock"]["within_13"] == 1.0
    assert by["ft"]["recall"] == pytest.approx(0.5) and by["ft"]["precision"] == pytest.approx(1.0)
    text = out.read_text()
    assert "Camera-shift risk" in text and "oblique challenge camera" in text
    assert "| stock |" in text and "| ft |" in text and "eval_detections output: stock" in text


def test_evaluate_cli_default_report_path_and_extra_dets(tmp_path, monkeypatch, capsys):
    make_world(tmp_path, monkeypatch, games=["IMG_0108"])
    p = gt_as_dets(tmp_path / "a.txt", 5)
    det_eval.main(["--game", "IMG_0108", "--dets", f"mine={p}"])
    report = tmp_path / "outputs" / "reports" / "detector" / "IMG_0108.md"
    assert report.exists() and "| mine |" in report.read_text()
    with pytest.raises(SystemExit):
        det_eval.main(["--game", "IMG_0108"])
