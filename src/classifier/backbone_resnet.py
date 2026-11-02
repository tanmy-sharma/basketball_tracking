"""ResNet-34 baseline (torchvision, ImageNet-1k weights)."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34

from src.classifier.backbone_base import Backbone
from src.core.config import BackboneCfg

RESNET34_URL = ResNet34_Weights.IMAGENET1K_V1.url


class ResNet34Backbone(Backbone):
    feature_dim = 512

    def __init__(self, cfg: BackboneCfg) -> None:
        super().__init__()
        self.net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if cfg.pretrained else None)
        self.net.fc = nn.Identity()

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.net(pixels)

    def blocks(self) -> list[nn.Module]:
        n = self.net
        return [n.layer1, n.layer2, n.layer3, n.layer4]
