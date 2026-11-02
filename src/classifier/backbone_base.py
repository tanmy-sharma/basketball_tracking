"""Backbone interface: pooled features plus an ordered list of blocks for progressive unfreezing."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Backbone(nn.Module, ABC):
    feature_dim: int

    @abstractmethod
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) normalised pixels -> (B, feature_dim)."""

    @abstractmethod
    def blocks(self) -> list[nn.Module]:
        """Blocks from input to output; unfreezing counts from the end."""

    def extra_top_modules(self) -> list[nn.Module]:
        """Modules trained together with the last block (e.g. a final norm)."""
        return []
