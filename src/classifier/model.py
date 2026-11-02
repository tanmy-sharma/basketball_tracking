"""Backbone plus named heads, LLRD parameter groups and roster masking."""
from __future__ import annotations

import torch
from torch import nn

from src.classifier.backbone_base import Backbone
from src.classifier.heads import HEAD_CLASSES, make_head
from src.core.config import Config, PhaseCfg


class MultiHeadClassifier(nn.Module):
    def __init__(self, backbone: Backbone, heads: dict[str, int], hidden: int = 256, dropout: float = .2) -> None:
        super().__init__()
        self.backbone, self.hidden, self.dropout = backbone, hidden, dropout
        self.heads = nn.ModuleDict()
        for name, n in heads.items():
            self.add_head(name, n)

    def add_head(self, name: str, num_classes: int) -> None:
        self.heads[name] = make_head(self.backbone.feature_dim, self.hidden, num_classes, self.dropout)

    def forward(self, pixels: torch.Tensor, head: str) -> torch.Tensor:
        return self.heads[head](self.backbone(pixels))


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~allowed, float("-inf"))


def set_backbone_trainable(backbone: Backbone, unfreeze_blocks: int) -> list[list[nn.Module]]:
    """Freeze everything, then unfreeze the last `unfreeze_blocks` blocks (plus top extras with the last one).

    Returns the unfrozen module groups from the top block downwards (group 0 = top block + extras).
    """
    for p in backbone.parameters():
        p.requires_grad = False
    blocks = backbone.blocks()
    n = min(max(unfreeze_blocks, 0), len(blocks))
    groups: list[list[nn.Module]] = []
    for k in range(n):
        mods = [blocks[len(blocks) - 1 - k]] + (backbone.extra_top_modules() if k == 0 else [])
        for m in mods:
            for p in m.parameters():
                p.requires_grad = True
        groups.append(mods)
    return groups


def _split_decay(params: list[nn.Parameter], lr: float, wd: float) -> list[dict]:
    decay = [p for p in params if p.ndim > 1]
    plain = [p for p in params if p.ndim <= 1]
    return ([{"params": decay, "lr": lr, "weight_decay": wd}] if decay else []) + \
           ([{"params": plain, "lr": lr, "weight_decay": 0.0}] if plain else [])


def llrd_param_groups(model: MultiHeadClassifier, head: str, phase: PhaseCfg, weight_decay: float) -> list[dict]:
    """Freeze per phase and return groups: head at head_lr, top block at backbone_lr, each lower block x llrd.

    Parameters with ndim <= 1 get no weight decay. Other heads stay frozen.
    """
    for p in model.heads.parameters():
        p.requires_grad = False
    for p in model.heads[head].parameters():
        p.requires_grad = True
    groups = _split_decay(list(model.heads[head].parameters()), phase.head_lr, weight_decay)
    for k, mods in enumerate(set_backbone_trainable(model.backbone, phase.unfreeze_blocks)):
        params = [p for m in mods for p in m.parameters()]
        groups += _split_decay(params, phase.backbone_lr * phase.llrd ** k, weight_decay)
    return groups


def build_backbone(cfg: Config, pretrained: bool | None = None) -> Backbone:
    bcfg = cfg.model.backbone
    if pretrained is not None:
        bcfg = bcfg.model_copy(update={"pretrained": pretrained})
    if bcfg.name == "dino":
        from src.classifier.backbone_dino import DinoBackbone
        return DinoBackbone(bcfg)
    from src.classifier.backbone_resnet import ResNet34Backbone
    return ResNet34Backbone(bcfg)


def build_model(cfg: Config, pretrained: bool | None = None) -> MultiHeadClassifier:
    """Model with only the head named in cfg.model.head. `pretrained=False` skips weight downloads."""
    m = cfg.model
    return MultiHeadClassifier(build_backbone(cfg, pretrained), {m.head: HEAD_CLASSES[m.head]},
                               m.head_hidden, m.dropout)
