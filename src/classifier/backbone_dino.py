"""DINOv2 backbone: CLS token concatenated with the mean patch token, as in Dinov2ForImageClassification."""
from __future__ import annotations

import torch
from torch import nn
from transformers import Dinov2Config, Dinov2Model

from src.classifier.backbone_base import Backbone
from src.core.config import BackboneCfg

# facebook/dinov2-small architecture, so pretrained=False needs no download.
SMALL_CONFIG: dict = dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=6, mlp_ratio=4,
                          patch_size=14, image_size=518, layerscale_value=1.0, qkv_bias=True,
                          use_swiglu_ffn=False, drop_path_rate=0.0)


class DinoBackbone(Backbone):
    def __init__(self, cfg: BackboneCfg) -> None:
        super().__init__()
        if cfg.pretrained:
            self.net = Dinov2Model.from_pretrained(cfg.hf_id, revision=cfg.hf_revision)
        else:
            self.net = Dinov2Model(Dinov2Config(**{**SMALL_CONFIG, **cfg.config_overrides}))
        self.feature_dim = 2 * self.net.config.hidden_size

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        h = self.net(pixel_values=pixels).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)

    def blocks(self) -> list[nn.Module]:
        return list(self.net.encoder.layer)

    def extra_top_modules(self) -> list[nn.Module]:
        return [self.net.layernorm]
