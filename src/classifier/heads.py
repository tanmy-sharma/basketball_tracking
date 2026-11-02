from __future__ import annotations

from torch import nn

HEAD_CLASSES = {"number": 9, "eligibility": 2}


def make_head(in_dim: int, hidden: int, n_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, n_classes))
