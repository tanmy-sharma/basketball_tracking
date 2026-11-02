"""ViTSTR-small: a timm ViT whose first `seqlen` output tokens are each classified into a character.

Architecture and weights follow roatienza/deep-text-recognition-benchmark (Apache License 2.0),
Atienza, "Vision Transformer for Fast and Efficient Scene Text Recognition", ICDAR 2021.
This is our own small subclass of timm's VisionTransformer; no upstream module is vendored.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from src.core.config import RecognizerCfg
from src.recognizer.tokens import NUM_TOKENS, ORIG_TOKENS, TOKENS

PREFIX = "module.vitstr."
SMALL = dict(img_size=224, patch_size=16, in_chans=1, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
             qkv_bias=True)


class ViTSTR(VisionTransformer):
    def __init__(self, seqlen: int = 4, num_classes: int = NUM_TOKENS, **kw) -> None:
        super().__init__(num_classes=num_classes, **kw)
        self.seqlen = seqlen

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W) -> (B, seqlen, num_classes)
        feats = self.forward_features(x)  # patch embed, cls, pos_embed, pos_drop, blocks, norm
        return self.head(feats[:, :self.seqlen])

    def block_list(self) -> list[nn.Module]:
        return list(self.blocks)


def remap_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the DataParallel prefix and cut the 96-way head down to [GO], [s], 0-9, copying matching rows."""
    out = {k.removeprefix(PREFIX): v for k, v in sd.items()}
    rows = [ORIG_TOKENS.index(t) for t in TOKENS]
    for name in ("head.weight", "head.bias"):
        out[name] = out[name][rows].clone()
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(url: str, sha256: str, cache_dir: Path | None = None) -> Path:
    """Download into a local cache, verify sha256 (raises ValueError on mismatch), return the path."""
    cache = Path(cache_dir) if cache_dir else Path(torch.hub.get_dir()) / "checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / url.rsplit("/", 1)[-1]
    if dest.exists() and file_sha256(dest) == sha256:
        return dest
    tmp = dest.with_name(dest.name + ".part")
    torch.hub.download_url_to_file(url, str(tmp), progress=False)
    got = file_sha256(tmp)
    if got != sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch for {url}: expected {sha256}, got {got}")
    os.replace(tmp, dest)
    return dest


def build_vitstr(cfg: RecognizerCfg, cache_dir: Path | None = None) -> ViTSTR:
    kw = {**SMALL, **cfg.config_overrides}
    if "img_size" not in cfg.config_overrides:
        kw["img_size"] = cfg.input_size
    model = ViTSTR(seqlen=cfg.seqlen, **kw)
    if cfg.pretrained:
        path = fetch_weights(cfg.url, cfg.sha256, cache_dir)
        sd = remap_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.load_state_dict(sd)
    return model
