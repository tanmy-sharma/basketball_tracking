from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from src.classifier import transforms
from src.classifier.transforms import CropTransform, augment_native, letterbox
from src.core.config import AugCfg, Config

CLASSIFIER_DIR = Path(transforms.__file__).parent


def asym_image(h: int = 33, w: int = 21) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = np.linspace(40, 215, w, dtype=np.uint8)[None, :, None]  # left dark, right bright
    img[3:15, 2:7] = 250
    return img


def test_output_shape_and_aspect():
    t = CropTransform(224)
    out = t(np.full((33, 21, 3), 128, np.uint8))
    assert out.shape == (3, 224, 224) and out.dtype == torch.float32
    lb = letterbox(np.full((33, 21, 3), 7, np.uint8), 224, 0)
    cols = np.where((lb != 0).any(axis=(0, 2)))[0]
    rows = np.where((lb != 0).any(axis=(1, 2)))[0]
    assert abs((np.ptp(cols) + 1) / (np.ptp(rows) + 1) - 21 / 33) < 0.03
    assert rows.min() == 0 and rows.max() == 223  # tall crop fills the height


def test_letterbox_grey_and_pad_values():
    g = letterbox(np.full((10, 40), 9, np.uint8), 64, 200)
    assert g.shape == (64, 64) and g[0, 0] == 200 and g[32, 32] == 9
    c = letterbox(np.full((40, 10, 3), 9, np.uint8), 64, (1, 2, 3))
    assert c.shape == (64, 64, 3) and tuple(c[10, 0]) == (1, 2, 3)


def test_augment_native_keeps_uint8_rgb():
    rng = np.random.default_rng(0)
    img = asym_image()
    for _ in range(50):
        out = augment_native(img, AugCfg(), rng)
        assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3
        assert out.shape[0] >= 33 * (1 - 2 * .08) - 2 and out.shape[1] >= 21 * (1 - 2 * .08) - 2


def test_never_mirrors():
    rng = np.random.default_rng(1)
    t = CropTransform(64, AugCfg(grey_p=.3))
    img = asym_image()
    ref = CropTransform(64)(img).numpy().ravel()
    mirror_img = img[:, np.arange(img.shape[1] - 1, -1, -1)]
    mirrored = CropTransform(64)(mirror_img).numpy().ravel()
    for _ in range(200):
        out = t(img, rng).numpy().ravel()
        assert np.corrcoef(out, ref)[0, 1] > np.corrcoef(out, mirrored)[0, 1]


def test_rotation_cap_enforced_by_config():
    with pytest.raises(ValueError):
        AugCfg(max_rotate_deg=30)
    rng = np.random.default_rng(2)
    # a 90 degree turn can never happen: the stroke stays vertical-ish
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, 18:22] = 255
    for _ in range(50):
        out = augment_native(img, AugCfg(subcrop_frac=0, jpeg_p=0, blur_p=0, sat=0, hue_deg=0, brightness=0,
                                         contrast=0), rng)
        assert out[:, :, 0].std(axis=0).argmax() in range(10, 30)


def test_aug_config_has_no_mirror_option():
    assert not any("flip" in f.lower() or "mirror" in f.lower() for f in AugCfg.model_fields)
    with pytest.raises(ValueError):
        AugCfg.model_validate({"hflip": True})
    with pytest.raises(ValueError):
        Config.model_validate({"run_name": "t", "plugin": "x", "games": {"train": ["IMG_0106"]},
                               "aug": {"hflip": 0.5}})


def test_no_mirror_calls_in_classifier_source():
    bad = []
    for p in CLASSIFIER_DIR.glob("*.py"):
        src = p.read_text()
        if re.search(r"\[[^\]]*::\s*-1\s*\]", src):
            bad.append(f"{p.name}: reversed slice")
        for node in ast.walk(ast.parse(src)):
            name = getattr(node, "attr", getattr(node, "id", ""))
            if isinstance(name, str) and "flip" in name.lower():
                bad.append(f"{p.name}:{node.lineno} {name}")
    assert not bad, bad
