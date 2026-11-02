from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.config import AugCfg, Config, RecognizerCfg, load_config
from src.core.constants import NUMBERS
from src.core.paths import ROOT, outputs_dir
from src.core.plugin import get_plugin
from src.core.run import open_run
from src.dataio.store import CropStore, crops_root
from src.recognizer import tokens as T
from src.recognizer.dataset import DigitCropDataset, roster_mask
from src.recognizer.model import ViTSTR, build_vitstr, fetch_weights, remap_state_dict
from src.recognizer.plugin import RecognizerPlugin, param_groups
from src.recognizer.predict import predict_crops, predict_free
from src.recognizer.scoring import score_numbers
from src.recognizer.transforms import eval_transform, to_input, train_transform
from src.tests.fixtures import make_crop_store

TINY = dict(img_size=32, patch_size=16, embed_dim=48, depth=4, num_heads=2)


def tiny_cfg(**kw) -> RecognizerCfg:
    return RecognizerCfg(input_size=32, pretrained=False, config_overrides=TINY, **kw)


def test_digit_targets_all_numbers():
    expect = {"1": [0, 3, 1, 0], "3": [0, 5, 1, 0], "4": [0, 6, 1, 0], "5": [0, 7, 1, 0], "6": [0, 8, 1, 0],
              "9": [0, 11, 1, 0], "10": [0, 3, 2, 1], "11": [0, 3, 3, 1], "12": [0, 3, 4, 1]}
    for n in NUMBERS:
        assert T.encode_number(n) == expect[n]
    with pytest.raises(ValueError):
        T.encode_number("123")
    tgt, mask = T.candidate_targets()
    assert tgt.shape == (9, 4) and mask.sum(1).tolist() == [2, 2, 2, 2, 2, 2, 3, 3, 3]
    assert T.decode_free(tgt) == list(NUMBERS)


def test_head_remap_copies_digit_rows():
    g = torch.Generator().manual_seed(0)
    sd = {"module.vitstr.head.weight": torch.randn(96, 8, generator=g),
          "module.vitstr.head.bias": torch.randn(96, generator=g),
          "module.vitstr.cls_token": torch.zeros(1, 1, 8)}
    out = remap_state_dict(sd)
    w, b = sd["module.vitstr.head.weight"], sd["module.vitstr.head.bias"]
    assert out["head.weight"].shape == (12, 8) and "cls_token" in out
    assert torch.equal(out["head.weight"][0], w[0]) and torch.equal(out["head.weight"][1], w[1])
    orig = T.ORIG_TOKENS
    for d in "0123456789":
        assert torch.equal(out["head.weight"][T.TOKENS.index(d)], w[orig.index(d)])
        assert out["head.bias"][T.TOKENS.index(d)] == b[orig.index(d)]


def test_forward_shape_and_greyscale():
    m = build_vitstr(tiny_cfg())
    assert isinstance(m, ViTSTR) and m.patch_embed.proj.in_channels == 1
    assert m(torch.randn(3, 1, 32, 32)).shape == (3, 4, 12)
    with pytest.raises(RuntimeError):
        m(torch.randn(1, 3, 32, 32))


def test_scoring_shape_order_and_roster_argmax():
    logits = torch.full((9, 4, 12), -10.0)
    tgt, _ = T.candidate_targets()
    for i in range(9):  # row i strongly encodes number i
        for p in range(4):
            logits[i, p, tgt[i, p]] = 10.0
    s = score_numbers(logits)
    assert s.shape == (9, 9) and s.argmax(1).tolist() == list(range(9))
    probs = s.softmax(-1)
    assert torch.allclose(probs.sum(1), torch.ones(9))
    # "10" and "1": [s] after the first digit separates them
    assert probs[0, 0] > .99 and probs[6, 6] > .99
    roster = torch.from_numpy(roster_mask("pink"))  # 1, 3, 5
    assert roster.sum() == 3
    assert s[1].masked_fill(~roster, float("-inf")).argmax().item() == 1


def test_freeze_groups():
    m = build_vitstr(tiny_cfg())
    from src.core.config import PhaseCfg
    ph = PhaseCfg(name="p", epochs=1, unfreeze_blocks=2, head_lr=1e-3, backbone_lr=1e-4, llrd=.5)
    groups = param_groups(m, ph, 0.05)
    trainable = {id(p) for g in groups for p in g["params"]}
    assert trainable == {id(p) for p in m.parameters() if p.requires_grad}
    blocks = m.block_list()
    assert all(not p.requires_grad for p in blocks[0].parameters())
    assert all(p.requires_grad for p in blocks[-1].parameters())
    assert not m.patch_embed.proj.weight.requires_grad and not m.cls_token.requires_grad
    lrs = {round(g["lr"], 9) for g in groups}
    assert lrs == {1e-3, 1e-4, 5e-5}
    assert any(p is blocks[-1].norm1.weight for g in groups if g["weight_decay"] == 0 for p in g["params"])
    head_only = param_groups(m, PhaseCfg(name="h", epochs=1, unfreeze_blocks=0, head_lr=1e-3), .05)
    assert {id(p) for g in head_only for p in g["params"]} == {id(p) for p in m.head.parameters()}


def _rgb(rng, w=24, h=36):
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_input_is_greyscale_normalised_and_no_flip():
    cfg = tiny_cfg()
    rgb = np.zeros((36, 24, 3), np.uint8)
    rgb[:, :6] = 255  # bright on the left only
    x = to_input(rgb, cfg)
    assert x.shape == (1, 32, 32) and x.dtype == np.float32 and x.min() >= -1 and x.max() <= 1
    assert x[0, :, :16].mean() > x[0, :, 16:].mean()
    aug = AugCfg(max_rotate_deg=8, subcrop_frac=.0, grey_p=0, hue_deg=0, sat=0, brightness=0, contrast=0,
                 blur_p=0, jpeg_p=0)
    tf = train_transform(cfg, aug)
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = tf(rgb, rng)
        assert y[0, :, :16].mean() > y[0, :, 16:].mean()
    src = (ROOT / "src" / "recognizer").glob("*.py")
    assert not any("flip" in p.read_text().lower() for p in src)
    assert not hasattr(AugCfg(), "hflip")
    stretch = to_input(rgb, tiny_cfg(resize="stretch"))
    assert stretch.shape == (1, 32, 32)


def test_sha256_mismatch_raises(tmp_path):
    f = tmp_path / "w.pth"
    f.write_bytes(b"not weights")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch_weights(f.as_uri(), "0" * 64, cache_dir=tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*"))
    ok = hashlib.sha256(b"not weights").hexdigest()
    assert fetch_weights(f.as_uri(), ok, cache_dir=tmp_path / "cache").read_bytes() == b"not weights"


def test_configs_load():
    for name in ("base", "vitstr_all", "vitstr_eligible", "vitstr_xcolor_no_yellow", "vitstr_xcolor_no_green"):
        p = ROOT / "src" / "configs" / "recognizer" / f"{name}.yaml"
        if name == "base":
            continue
        c = load_config(p)
        assert c.plugin == "recognizer" and c.games.test == ["IMG_0108"] and len(c.games.train) == 7
        assert [ph.unfreeze_blocks for ph in c.train.phases] == [2, 6]
    assert load_config(ROOT / "src/configs/recognizer/vitstr_eligible.yaml").filters.include_csv.endswith(
        "folds3/eligible_final.csv")
    assert load_config(ROOT / "src/configs/recognizer/vitstr_xcolor_no_green.yaml").filters.exclude[0].bib == "green"


def _synthetic_run(n_train=480, numbers=("1", "3", "5")):
    rng = np.random.default_rng(0)
    cfg_d = {"run_name": "rec_test", "plugin": "recognizer", "device": "cpu", "seed": 0,
             "games": {"train": ["IMG_0106"], "test": ["IMG_0108"]},
             "split": {"guard_frames": 0},
             "recognizer": tiny_cfg().model_dump(),
             "aug": {"max_rotate_deg": 3, "grey_p": .3, "jpeg_p": 0, "blur_p": 0},
             "train": {"batch_size": 32, "num_workers": 0, "patience": 5, "monitor": "val_balanced_acc",
                       "label_smoothing": 0.0,
                       "phases": [{"name": "a", "epochs": 15, "unfreeze_blocks": 4, "head_lr": 3e-3,
                                   "backbone_lr": 1e-3, "warmup_frac": .1}]}}
    cfg = Config.model_validate(cfg_d)
    make_crop_store(crops_root(cfg.crops), "IMG_0106", n_train, list(numbers), rng)
    return cfg


def test_plugin_trains_above_chance_and_predicts():
    cfg = _synthetic_run()
    assert get_plugin("recognizer") is RecognizerPlugin
    run = open_run(cfg)
    plugin = RecognizerPlugin(cfg)
    res = plugin.fit(run)
    ck = torch.load(res.best_path, weights_only=True)
    assert ck["head"] == "digits" and ck["classes"] == T.TOKENS and ck["backbone"] == "vitstr_small"
    assert res.metric > 0.5, res.metric  # chance over the 3 present classes is 0.33

    store = CropStore(crops_root(cfg.crops))
    df = pd.read_csv(crops_root(cfg.crops) / "IMG_0106" / "manifest.csv", dtype=str).head(10)
    df["offset"] = df.offset.astype(int)
    df["crop_w"], df["crop_h"] = df.crop_w.astype(int), df.crop_h.astype(int)
    probs = predict_crops(res.best_path, df, store, "cpu", batch_size=4)
    assert probs.shape == (10, 9) and np.allclose(probs.sum(1), 1, atol=1e-5)
    text = predict_free(res.best_path, df, store, "cpu", batch_size=4)
    assert len(text) == 10 and all(t == "" or t.isdigit() for t in text)
    # the scoring path matches the manual log-prob sum for one crop
    from src.recognizer.predict import load_model
    model, rec = load_model(res.best_path, tiny_cfg())
    r = df.iloc[0]
    x = torch.from_numpy(eval_transform(rec)(store.read(r.video, int(r.offset), int(r.crop_w), int(r.crop_h)), None))[None]
    lp = model(x).log_softmax(-1)[0]
    manual = [sum(lp[p, t].item() for p, t in enumerate(T.encode_number(n)) if 1 <= p <= len(n) + 1) for n in NUMBERS]
    assert np.allclose(torch.tensor(manual).softmax(0).numpy(), probs[0], atol=1e-4)


def test_dataset_item_shapes():
    cfg = _synthetic_run(30, ("1", "3", "5"))
    store = CropStore(crops_root(cfg.crops))
    df = pd.read_csv(crops_root(cfg.crops) / "IMG_0106" / "manifest.csv", dtype=str)
    for c in ("offset", "crop_w", "crop_h"):
        df[c] = df[c].astype(int)
    ds = DigitCropDataset(df, store, train_transform(cfg.recognizer, cfg.aug), 4, augment=True)
    x, tok, y, ros = ds[0]
    assert x.shape == (1, 32, 32) and tok.shape == (4,) and ros.shape == (9,) and ros[y]
    assert outputs_dir().exists()
