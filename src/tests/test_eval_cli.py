from __future__ import annotations

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest
import yaml

from src.core.config import load_config
from src.core.run import resolve_run_dir
from src.dataio.store import crops_root
from src.evaluate import classifier as C
from src.tests.fixtures import make_crop_store, toy_config_dict

PHASE = {"name": "p", "epochs": 1, "unfreeze_blocks": 0, "head_lr": 1e-3}
GREEN, YELLOW = ["6", "9", "11"], ["6", "9", "10"]


def write_cfg(path, run_name, plugin, test=("IMG_0108",), head="number"):
    d = toy_config_dict(run_name)
    d.update(plugin=plugin, games={"train": ["IMG_0106"], "test": list(test)})
    if plugin == "classifier":
        d["model"] = {"backbone": {"pretrained": False}, "head": head}
        if head == "eligibility":
            d["data"] = {"label_csv": "x.csv"}
    else:
        d["recognizer"] = {"pretrained": False}
    path.write_text(yaml.safe_dump(d))
    cfg = load_config(path)
    run = resolve_run_dir(cfg)
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"x")
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    cfgs = {n: write_cfg(tmp_path / f"{n}.yaml", n, p, head=h) for n, p, h in
            [("dino", "classifier", "number"), ("vitstr", "recognizer", "number"),
             ("elig", "classifier", "eligibility")]}
    root = crops_root(load_config(cfgs["dino"]).crops)
    a = make_crop_store(root, "IMG_0108", 60, GREEN, rng)
    # second team (yellow) in the same game: rewrite manifest by building a second store and merging
    b = make_crop_store(tmp_path / "other", "IMG_0108", 60, YELLOW, rng)
    from src.dataio.store import save_manifest
    b = b.assign(track_id=b.track_id + 3, offset=b.offset + 10**6)
    b["crop_id"] = [f"IMG_0108:{f:06d}:{t:02d}" for f, t in zip(b.frame, b.track_id)]
    # keep it simple: evaluate on the green store only plus yellow rows pointing at valid offsets
    b["offset"] = a.offset.to_numpy()[:len(b)]
    b["crop_w"], b["crop_h"] = a.crop_w.to_numpy()[:len(b)], a.crop_h.to_numpy()[:len(b)]
    b.loc[:, "facing"] = "back"
    b.loc[b.index[:10], "max_iou"] = .6
    save_manifest(root / "IMG_0108", pd.concat([a, b], ignore_index=True))
    seen = []

    def make(plugin):
        def predict(best, df, store, device, batch_size=256):
            n = len(df)
            seen.append((plugin, str(best), n))
            if "/elig__" in str(best):
                p = np.zeros((n, 2))
                p[np.arange(n), df.frame.to_numpy() % 2] = 1
                return p
            p = np.full((n, 9), .02)
            if plugin == "classifier":
                p[np.arange(n), df.number_idx.to_numpy()] = .84   # perfect reader
            else:
                p[:, 0] = .9                                      # always says "1"
            return p / p.sum(1, keepdims=True)
        return predict

    for plugin in ("classifier", "recognizer"):
        mod = types.ModuleType(f"src.{plugin}.predict")
        mod.predict_crops = make(plugin)
        monkeypatch.setitem(sys.modules, f"src.{plugin}.predict", mod)
    return cfgs, tmp_path, seen


def test_refuses_non_test_game_without_flag(setup):
    cfgs, tmp, _ = setup
    with pytest.raises(ValueError, match="allow-nontest"):
        C.run_eval("IMG_0106", {"dino": str(cfgs["dino"])}, tmp / "out")


def test_writes_all_outputs_and_report_content(setup, capsys):
    cfgs, tmp, _ = setup
    reads = tmp / "reads.csv"
    from src.dataio.store import load_manifest
    man = load_manifest(crops_root(load_config(cfgs["dino"]).crops), ["IMG_0108"])
    pd.DataFrame({"crop_id": man.crop_id[:20], "text": ["6"] * 10 + ["12"] * 5 + [""] * 5}).to_csv(reads, index=False)
    out = tmp / "out"
    res = C.run_eval("IMG_0108", {"dino": str(cfgs["dino"]), "vitstr": str(cfgs["vitstr"])}, out,
                     eligibility=str(cfgs["elig"]), free_reads={"vitstr": str(reads)}, device="cpu")
    for f in ("metrics.json", "report.md", "confusion_dino.csv", "confusion_vitstr.csv",
              "predictions_dino.csv.gz", "predictions_vitstr.csv.gz"):
        assert (out / f).exists(), f
    m = json.loads((out / "metrics.json").read_text())
    d = m["models"]["dino"]
    assert d["all"]["acc_9way"] == 1.0 and d["all"]["acc_team"] == 1.0
    assert d["clean"]["n"] == d["all"]["n"] - 10 and m["n_clean"] == m["n_all"] - 10
    assert len(d["all"]["per_track_team"]) == 6
    assert {"all_crops/unmasked/whole", "eligible_only/roster_masked/2s"} <= set(d["all"]["votes"])
    assert set(d["all"]["coverage"]) == {"max_softmax", "eligibility"}
    assert m["models"]["vitstr"]["all"]["acc_9way"] < .5
    fr = m["models"]["vitstr"]["free_reads"]
    assert fr["n"] == 20 and fr["off_roster_rate"] == .25 and fr["empty_rate"] == .25
    assert {b["bib"] for b in d["all"]["by_bib"]} == {"green", "yellow"}
    rep = (out / "report.md").read_text()
    for needle in ("Leakage", "Camera shift", "per track", "6 vs 9 vs 10 by bib", "By facing", "Tracklet votes",
                   "Coverage vs accuracy", "Free decoding"):
        assert needle in rep, needle
    pr = pd.read_csv(out / "predictions_dino.csv.gz")
    assert len(pr) == res["n_all"] and {"pred_9way", "pred_team", "elig"} <= set(pr.columns)
    assert pd.read_csv(out / "confusion_dino.csv", index_col=0).shape == (9, 9)


def test_allow_nontest_prints_banner(setup, capsys):
    cfgs, tmp, _ = setup
    write_cfg(tmp / "tr.yaml", "tr", "classifier", test=("IMG_0107",))
    r = C.run_eval("IMG_0108", {"dino": str(tmp / "tr.yaml")}, tmp / "o2", allow_nontest=True, device="cpu")
    assert "NOT held-out" in capsys.readouterr().out
    assert "NOT held-out" in (tmp / "o2" / "report.md").read_text() and r["banner"]


def test_cli_args_and_missing_checkpoint(setup, tmp_path):
    cfgs, tmp, _ = setup
    assert C._kv(["a=b.yaml"]) == {"a": "b.yaml"}
    with pytest.raises(SystemExit):
        C._kv(["nope"])
    import shutil
    shutil.rmtree(resolve_run_dir(load_config(cfgs["dino"])))
    with pytest.raises(FileNotFoundError):
        C.run_eval("IMG_0108", {"dino": str(cfgs["dino"])}, tmp / "o3", device="cpu")
