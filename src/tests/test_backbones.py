from __future__ import annotations

import pytest
import torch

from src.classifier.backbone_dino import DinoBackbone
from src.classifier.backbone_resnet import ResNet34Backbone
from src.classifier.model import MultiHeadClassifier, build_model, llrd_param_groups, mask_logits
from src.classifier.plugin import ClassifierPlugin
from src.core.config import BackboneCfg, Config, PhaseCfg
from src.tests.fixtures import tiny_dino_overrides


def cfg_for(name: str = "dino", head: str = "number") -> Config:
    bb = {"name": name, "pretrained": False}
    if name == "dino":
        bb["config_overrides"] = tiny_dino_overrides()
    return Config.model_validate({
        "run_name": "t", "plugin": "classifier", "games": {"train": ["IMG_0106"]},
        "model": {"backbone": bb, "head": head, "head_hidden": 16, "input_size": 56},
        "data": {"label_csv": "x.csv"} if head == "eligibility" else {},
        "train": {"phases": [{"name": "p", "epochs": 1, "unfreeze_blocks": 0, "head_lr": 1e-3}]}})


def test_tiny_dino_dim_and_blocks():
    bb = DinoBackbone(BackboneCfg(pretrained=False, config_overrides=tiny_dino_overrides()))
    assert bb.feature_dim == 96 and len(bb.blocks()) == 4
    assert bb(torch.randn(2, 3, 56, 56)).shape == (2, 96)
    assert bb(torch.randn(1, 3, 28, 42)).shape == (1, 96)  # position embeddings interpolate


def test_resnet34_dim_and_blocks():
    bb = ResNet34Backbone(BackboneCfg(name="resnet34", pretrained=False))
    assert bb.feature_dim == 512 and len(bb.blocks()) == 4
    assert bb(torch.randn(2, 3, 64, 64)).shape == (2, 512)


def test_backbone_swap_changes_only_backbone():
    a, b = build_model(cfg_for("dino")), build_model(cfg_for("resnet34"))
    assert type(a) is type(b) is MultiHeadClassifier
    assert a(torch.randn(2, 3, 56, 56), "number").shape == b(torch.randn(2, 3, 56, 56), "number").shape == (2, 9)
    assert a.heads["number"][-1].out_features == b.heads["number"][-1].out_features == 9


def test_attachable_heads():
    m = build_model(cfg_for("dino"))
    m.add_head("eligibility", 2)
    x = torch.randn(2, 3, 56, 56)
    assert m(x, "number").shape == (2, 9) and m(x, "eligibility").shape == (2, 2)


@pytest.mark.parametrize("n", [0, 1, 2, 4, 9])
def test_requires_grad_matches_unfreeze_blocks(n):
    m = build_model(cfg_for("dino"))
    llrd_param_groups(m, "number", PhaseCfg(name="p", epochs=1, unfreeze_blocks=n, head_lr=1e-3, backbone_lr=1e-4), .05)
    blocks = m.backbone.blocks()
    k = min(n, 4)
    for i, blk in enumerate(blocks):
        want = i >= len(blocks) - k
        assert all(p.requires_grad == want for p in blk.parameters())
    assert all(p.requires_grad == (k > 0) for p in m.backbone.net.layernorm.parameters())
    assert not any(p.requires_grad for p in m.backbone.net.embeddings.parameters())
    assert all(p.requires_grad for p in m.heads["number"].parameters())


def test_llrd_ratios_and_no_decay_group():
    m = build_model(cfg_for("dino"))
    ph = PhaseCfg(name="p", epochs=1, unfreeze_blocks=3, head_lr=3e-4, backbone_lr=1e-4, llrd=.5)
    groups = llrd_param_groups(m, "number", ph, .05)
    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
    wd_of = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
    blocks = m.backbone.blocks()
    for k in range(3):
        for p in blocks[3 - k].parameters():
            assert lr_of[id(p)] == pytest.approx(1e-4 * .5 ** k)
            assert wd_of[id(p)] == (.05 if p.ndim > 1 else 0.0)
    assert id(next(blocks[0].parameters())) not in lr_of
    for p in m.heads["number"].parameters():
        assert lr_of[id(p)] == 3e-4 and wd_of[id(p)] == (.05 if p.ndim > 1 else 0.0)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))


def test_mask_logits_restricts_argmax():
    logits = torch.tensor([[5., 1., 2.], [0., 3., 1.]])
    allowed = torch.tensor([[False, True, True], [True, False, True]])
    assert mask_logits(logits, allowed).argmax(1).tolist() == [2, 2]


def test_frozen_batchnorm_stays_fixed_in_training():
    cfg = cfg_for("resnet34")
    plugin = ClassifierPlugin(cfg)
    m = build_model(cfg)
    ph = PhaseCfg(name="p", epochs=1, unfreeze_blocks=1, head_lr=1e-2, backbone_lr=1e-2)
    groups = plugin.configure_phase(m, ph)
    m.train()
    plugin.set_train_mode(m, ph)
    net = m.backbone.net
    before = {k: v.clone() for k, v in m.state_dict().items() if "running" in k}
    opt = torch.optim.AdamW(groups)
    for _ in range(2):
        loss = m(torch.randn(4, 3, 64, 64), "number").sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = m.state_dict()
    for k, v in before.items():
        moved = not torch.equal(v, after[k])
        assert moved == k.startswith("backbone.net.layer4"), k
    assert not net.bn1.training and not net.layer1[0].bn1.training and net.layer4[0].bn1.training
