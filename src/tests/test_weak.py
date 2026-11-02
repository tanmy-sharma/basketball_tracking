from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.core.config import Config, WeakCfg, load_config
from src.core.constants import NUMBERS
from src.core.paths import outputs_dir
from src.core.plugin import FitResult
from src.dataio.select import training_rows
from src.dataio.store import CropStore, crops_root, load_manifest
from src.tests.fixtures import make_crop_store, toy_config_dict
from src.weak import folds as F
from src.weak import qa
from src.weak.assemble import assemble_labels, coverage_markdown, covered_numbers, eligibility_labels
from src.weak.finalize import finalize_eligible

FOLDS = [["IMG_0104", "IMG_0109"], ["IMG_0105", "IMG_0110"], ["IMG_0106", "IMG_0107", "IMG_0111"]]
# team A and team B jerseys per game, from data/Training/games_roster.csv
ROSTERS = {"IMG_0104": ("4 10 12", "6 9 11"), "IMG_0105": ("6 9 10", "1 3 5"), "IMG_0106": ("6 9 11", "1 3 5"),
           "IMG_0107": ("6 9 10", "4 10 12"), "IMG_0109": ("4 10 12", "1 3 5"), "IMG_0110": ("4 10 12", "6 9 11"),
           "IMG_0111": ("1 3 5", "6 9 10")}
PHASE = {"name": "p", "epochs": 1, "unfreeze_blocks": 0, "head_lr": 1e-3}


def weak_cfg(plugin: str = "recognizer", **over) -> Config:
    d = toy_config_dict()
    d.update(plugin="classifier", games={"train": [g for f in FOLDS for g in f], "test": ["IMG_0108"]},
             model={"backbone": {"pretrained": False}}, split={"val_start_frac": 0.8, "guard_frames": 0},
             weak={"name": "t", "folds": FOLDS, "fold_plugin": plugin, "fold_phases": [PHASE], "fold_subsample_every": 2, **over})
    d["train"]["phases"] = [dict(PHASE, name="orig", epochs=9)]
    return Config.model_validate(d)


def numbers_of(game: str) -> set[str]:
    return {n for team in ROSTERS[game] for n in team.split()}


def test_fold_config_moves_group_to_test_and_training_rows_raises(tmp_path):
    cfg = weak_cfg()
    f = F.fold_config(cfg, FOLDS[0])
    assert f.games.test == ["IMG_0108", "IMG_0104", "IMG_0109"] and f.games.label_pool == []
    assert f.games.train == ["IMG_0105", "IMG_0110", "IMG_0106", "IMG_0107", "IMG_0111"]
    assert f.plugin == "recognizer" and f.recognizer is not None and f.weak is None
    assert [p.epochs for p in f.train.phases] == [1] and f.filters.subsample_every == 2
    assert f.run_name == "fold_t_0104_0109"
    root = tmp_path / "c"
    df = make_crop_store(root, "IMG_0104", 60, ["6", "9", "11"], np.random.default_rng(0))
    with pytest.raises(ValueError, match="test"):
        training_rows(df, f)
    with pytest.raises(ValueError, match="not one of"):
        F.fold_config(cfg, ["IMG_0104"])


def test_fold_plugin_classifier_and_predict_dispatch(monkeypatch):
    f = F.fold_config(weak_cfg("classifier"), FOLDS[2])
    assert f.plugin == "classifier" and f.model.head == "number"
    for name in ("classifier", "recognizer"):
        mod = types.ModuleType(f"src.{name}.predict")
        mod.predict_crops = lambda *a, _n=name, **k: _n
        monkeypatch.setitem(sys.modules, f"src.{name}.predict", mod)
        assert F.predict_fn(name)() == name
    with pytest.raises(ValueError):
        F.predict_fn("detector")


@pytest.mark.parametrize("held", FOLDS)
def test_every_fold_covers_all_nine_numbers_from_rosters(held):
    train = [g for f in FOLDS if f is not held for g in f]
    covered = set().union(*(numbers_of(g) for g in train))
    assert covered == set(NUMBERS), (held, set(NUMBERS) - covered)
    assert numbers_of(held[0]) <= set(NUMBERS)


def test_weak_cfg_rejects_bad_folds():
    base = dict(name="x", fold_phases=[PHASE])
    with pytest.raises(ValueError):
        WeakCfg(folds=[["IMG_0104"]], **base)
    with pytest.raises(ValueError):
        WeakCfg(folds=[["IMG_0104"], ["IMG_0104"]], **base)


def _df(rows):
    return pd.DataFrame(rows, columns=["crop_id", "video", "number_idx", "bib", "facing", "max_iou"])


def test_assemble_labels_all_five_values():
    weak = WeakCfg(name="x", folds=FOLDS, fold_phases=[PHASE], tau_pos=.7, tau_neg=.3)
    df = _df([("a", "V", 0, "pink", "front", 0.0), ("b", "V", 0, "pink", "front", 0.0), ("c", "V", 0, "pink", "side", 0.0),
              ("d", "V", 7, "green", "back", 0.0), ("e", "V", 0, "pink", "front", .5),
              ("f", "V", 0, "pink", "front", 0.0)])
    probs = np.full((6, 9), 0.0)
    probs[0, 0] = .9; probs[0, 1] = .1            # eligible
    probs[1, 1] = .8; probs[1, 0] = .2            # wrong and low -> ineligible
    probs[2, 0] = .5; probs[2, 1] = .5            # between -> ambiguous
    probs[3, 7] = .9; probs[3, 0] = .1            # number not covered
    probs[4, 0] = .9; probs[4, 1] = .1            # overlapping
    probs[5, 1] = .6; probs[5, 0] = .4            # argmax wrong, p_true .4 -> ambiguous
    out = assemble_labels(df, probs, covered={0, 1}, train_bibs={"pink"}, weak=weak, max_iou=.1, fold="g")
    assert out.label.tolist() == [1, 0, -1, -2, -3, -1]
    assert out.fold_saw_bib.tolist() == [True, True, True, False, True, True]
    assert out.p_true.iloc[0] == .9 and out.pred_idx.iloc[1] == 1 and (out.fold == "g").all()
    assert eligibility_labels(out).label.tolist() == [1, 0]
    assert assemble_labels(df, probs, {0, 1, 7}, set(), weak, None).label.iloc[4] == 1


def test_covered_numbers():
    assert covered_numbers(pd.DataFrame({"number_idx": [0, 0, 4, 8]})) == {0, 4, 8}


def test_coverage_markdown_warns_only_when_uncovered():
    weak = WeakCfg(name="x", folds=FOLDS, fold_phases=[PHASE])
    df = _df([("a", "V", 0, "pink", "front", 0.0), ("b", "V", 7, "green", "front", 0.0)])
    p = np.zeros((2, 9)); p[:, 0] = 1
    md = coverage_markdown(assemble_labels(df, p, {0}, {"pink"}, weak, .1))
    assert "WARNING" in md and "number 11" in md
    assert "WARNING" not in coverage_markdown(assemble_labels(df, p, {0, 7}, {"pink"}, weak, .1))


def test_finalize_sources_and_fill():
    weak = WeakCfg(name="x", folds=FOLDS, fold_phases=[PHASE], head_threshold=.7, fill_min_per_class=2)
    rows = [("w1", 0, 1), ("w2", 0, 1), ("neg", 0, 0), ("amb", 0, -1), ("ov", 0, -3),
            ("h1", 7, -2), ("h2", 7, -2), ("low1", 7, -2), ("low2", 7, -2), ("low3", 7, -2), ("one", 1, 1)]
    wd = pd.DataFrame(rows, columns=["crop_id", "number_idx", "label"])
    sc = pd.Series({"w1": .1, "neg": .99, "amb": .99, "ov": .99, "h1": .8, "h2": .9, "low1": .5, "low2": .6, "low3": .1,
                    "one": .2, "w2": .2})
    res = finalize_eligible(wd, sc, weak).set_index("crop_id")
    assert res.source.to_dict() == {"w1": "weak", "w2": "weak", "one": "weak", "h1": "head", "h2": "head"}
    assert not {"neg", "amb", "ov"} & set(res.index)
    fill = lambda n: finalize_eligible(wd, sc, weak.model_copy(update={"fill_min_per_class": n})).query("source == 'fill'")  # noqa: E731
    assert fill(3).crop_id.tolist() == ["low2"]
    assert set(fill(4).crop_id) == {"low1", "low2"}
    assert set(fill(99).crop_id) == {"low1", "low2", "low3"}  # only uncovered crops can fill
    assert not {"neg", "amb", "ov"} & set(fill(99).crop_id)
    arr = finalize_eligible(wd, sc.reindex(wd.crop_id).to_numpy(), weak)
    assert len(arr) == 5


def test_qa_rates_and_sheets(tmp_path):
    wd = pd.DataFrame({"crop_id": list("abcd"), "label": [1, 0, 1, -1], "facing": ["front", "front", "side", "side"],
                       "fold_saw_bib": [True, True, False, False]})
    r = qa.eligibility_rates(wd)
    assert r["facing"].set_index("facing").eligible_rate.to_dict() == {"front": .5, "side": 1.0}
    assert r["fold_saw_bib"].set_index("fold_saw_bib").n.to_dict() == {False: 1, True: 2}
    root = tmp_path / "c"
    man = make_crop_store(root, "IMG_0106", 12, ["6", "9", "11"], np.random.default_rng(0))
    w = pd.DataFrame({"crop_id": man.crop_id, "video": man.video, "number_idx": man.number_idx,
                      "label": [1, 0] * 6})
    paths = qa.write_sheets(w, man, CropStore(root), tmp_path / "s", per=3)
    assert len(paths) == 6 and all(p.stat().st_size > 0 for p in paths)


def test_build_weak_labels_end_to_end_with_stubs(monkeypatch):
    cfg = weak_cfg("recognizer")
    rng = np.random.default_rng(0)
    root = crops_root(cfg.crops)
    for g in numbers_of_all():
        a, b = ROSTERS[g]
        make_crop_store(root, g, 60, a.split() + b.split(), rng)
    trained = []

    def fake_train(fcfg, argv=None):
        trained.append(fcfg)
        return FitResult(outputs_dir() / f"{fcfg.run_name}.pt", "m", 0.0)

    calls = []

    def fake_predict(best, df, store, device, batch_size=256):
        calls.append(set(df.video))
        p = np.full((len(df), 9), .01)
        p[np.arange(len(df)), df.number_idx.to_numpy()] = .9
        return p / p.sum(1, keepdims=True)

    mod = types.ModuleType("src.recognizer.predict")
    mod.predict_crops = fake_predict
    monkeypatch.setitem(sys.modules, "src.recognizer.predict", mod)
    monkeypatch.setattr("src.train.run_training", fake_train)
    wd = F.build_weak_labels(cfg, "cpu")
    assert [set(c) for c in calls] == [set(f) for f in FOLDS]
    assert not any("IMG_0108" in c.games.train for c in trained)
    assert set(wd.video) == set(numbers_of_all()) and (wd.label == 1).all()
    assert (wd.label != -2).all() and wd.fold.nunique() == 3
    F.write_outputs(wd, outputs_dir() / "weak" / "t")
    out = outputs_dir() / "weak" / "t"
    assert set(pd.read_csv(out / "eligibility_labels.csv").label) == {1}
    assert "WARNING" not in (out / "coverage.md").read_text()
    n = len(calls)
    F.build_weak_labels(cfg, "cpu")  # cached probabilities: predict not called again
    assert len(calls) == n


def numbers_of_all():
    return [g for f in FOLDS for g in f]


def test_weak_folds3_yaml_loads(tmp_path):
    import shutil
    from src.core.paths import ROOT
    d = ROOT / "src" / "configs" / "classifier"
    if not (d / "base.yaml").exists():
        pytest.skip("classifier base.yaml not written yet")
    cfg = load_config(d / "weak_folds3.yaml")
    assert cfg.weak.name == "folds3" and cfg.weak.folds == FOLDS and cfg.weak.fold_plugin == "recognizer"
    F.fold_config(cfg, FOLDS[1])
