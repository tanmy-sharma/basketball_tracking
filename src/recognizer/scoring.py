"""Roster scoring: sum of token log-probs through [s] for each of the 9 numbers, softmax across the 9."""
from __future__ import annotations

import numpy as np
import torch

from src.recognizer.tokens import candidate_targets


def score_numbers(logits: torch.Tensor, seqlen: int | None = None) -> torch.Tensor:
    """(B, S, 12) logits -> (B, 9) summed log-probs in NUMBERS order."""
    s = logits.shape[1]
    tgt, mask = candidate_targets(s)
    tgt, mask = torch.from_numpy(tgt).to(logits.device), torch.from_numpy(mask).to(logits.device)
    lp = logits.float().log_softmax(-1)
    pos = torch.arange(s, device=logits.device)[None, :]
    picked = lp[:, pos, tgt]  # (B, 9, S)
    return (picked * mask).sum(-1)


def number_probs(logits: torch.Tensor) -> np.ndarray:
    return score_numbers(logits).softmax(-1).cpu().numpy()
