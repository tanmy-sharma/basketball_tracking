"""Token set of the fine-tuned reader: [GO], [s], digits 0-9 (ids 0, 1, 2..11)."""
from __future__ import annotations

import string

import numpy as np

from src.core.constants import NUMBERS

GO, EOS = 0, 1
TOKENS: list[str] = ["[GO]", "[s]", *"0123456789"]
# token order of the public 96-way head
ORIG_TOKENS: list[str] = ["[GO]", "[s]", *string.printable[:-6]]
NUM_TOKENS = len(TOKENS)


def encode_number(text: str, seqlen: int = 4) -> list[int]:
    """`10` -> [GO] 1 0 [s]; `5` -> [GO] 5 [s] [GO] (padding is [GO], ignored by the loss)."""
    if not text.isdigit() or len(text) > seqlen - 2:
        raise ValueError(f"cannot encode {text!r} with seqlen={seqlen}")
    ids = [GO, *(TOKENS.index(c) for c in text), EOS]
    return ids + [GO] * (seqlen - len(ids))


def candidate_targets(seqlen: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Token targets (9, seqlen) in NUMBERS order, and the mask (9, seqlen) of scored positions (through [s])."""
    tgt = np.array([encode_number(n, seqlen) for n in NUMBERS], dtype=np.int64)
    mask = np.zeros_like(tgt, dtype=bool)
    for i, n in enumerate(NUMBERS):
        mask[i, 1:len(n) + 2] = True
    return tgt, mask


def decode_free(ids: np.ndarray) -> list[str]:
    """Greedy ids (B, seqlen) -> text: skip position 0, stop at [s] or [GO]."""
    out = []
    for row in np.asarray(ids):
        s = []
        for t in row[1:]:
            if t in (GO, EOS):
                break
            s.append(TOKENS[t])
        out.append("".join(s))
    return out
