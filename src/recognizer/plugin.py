"""RecognizerPlugin: fine-tunes ViTSTR-small on digit tokens with the generic phase loop."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.core.config import PhaseCfg
from src.core.constants import NUMBERS
from src.core.plugin import LoopPlugin, WeightSource
from src.core.run import Run
from src.dataio.select import training_rows
from src.dataio.store import CropStore, crops_root, load_manifest
from src.recognizer.dataset import DigitCropDataset, worker_init
from src.recognizer.model import ViTSTR, build_vitstr
from src.recognizer.scoring import score_numbers
from src.recognizer.tokens import TOKENS, decode_free
from src.recognizer.transforms import eval_transform, train_transform


def param_groups(model: ViTSTR, phase: PhaseCfg, weight_decay: float) -> list[dict]:
    """Head at head_lr; top unfrozen block (and final norm) at backbone_lr, each lower block times llrd."""
    for p in model.parameters():
        p.requires_grad = False
    blocks = model.block_list()
    n = min(phase.unfreeze_blocks, len(blocks))
    groups: list[tuple[list[nn.Module], float]] = [([model.head], phase.head_lr)]
    if n > 0:
        groups.append(([model.norm], phase.backbone_lr))
        for k, blk in enumerate(reversed(blocks[len(blocks) - n:])):
            groups.append(([blk], phase.backbone_lr * phase.llrd ** k))
    out = []
    for mods, lr in groups:
        for decay, sel in ((weight_decay, lambda p: p.ndim > 1), (0.0, lambda p: p.ndim <= 1)):
            ps = [p for m in mods for p in m.parameters() if sel(p)]
            for p in ps:
                p.requires_grad = True
            if ps:
                out.append({"params": ps, "lr": lr, "weight_decay": decay})
    return out


class RecognizerPlugin(LoopPlugin):
    def build_model(self) -> nn.Module:
        return build_vitstr(self.cfg.recognizer)

    def weight_sources(self) -> list[WeightSource]:
        r = self.cfg.recognizer
        if not r.pretrained:
            return []
        return [WeightSource("vitstr_small_patch16_224_aug", r.url, sha256=r.sha256,
                             trained_on="synthetic text datasets (upstream paper); head remapped to digits")]

    def checkpoint_meta(self) -> dict:
        return {"head": "digits", "classes": list(TOKENS), "backbone": "vitstr_small"}

    def build_loaders(self, run: Run) -> tuple[DataLoader, DataLoader]:
        cfg, t, r = self.cfg, self.cfg.train, self.cfg.recognizer
        df = training_rows(load_manifest(crops_root(cfg.crops), cfg.games.train), cfg)
        store = CropStore(crops_root(cfg.crops))
        tr, va = df[df.split == "train"], df[df.split == "val"]
        counts = tr.jersey.value_counts()
        run.update_manifest(train_rows_per_number={n: int(counts.get(n, 0)) for n in NUMBERS})
        for n in NUMBERS:
            if counts.get(n, 0) < 100:
                run.log(f"warning: only {counts.get(n, 0)} train rows for number {n}")
        train_ds = DigitCropDataset(tr, store, train_transform(r, cfg.aug), r.seqlen, augment=True, seed=cfg.seed)
        val_ds = DigitCropDataset(va, store, eval_transform(r), r.seqlen)
        kw = dict(batch_size=t.batch_size, num_workers=t.num_workers, worker_init_fn=worker_init)
        sampler = None
        if t.class_balance:
            w = 1.0 / np.bincount(train_ds.idx, minlength=len(NUMBERS))[train_ds.idx]
            sampler = WeightedRandomSampler(torch.from_numpy(w), len(train_ds),
                                            generator=torch.Generator().manual_seed(cfg.seed))
        return (DataLoader(train_ds, sampler=sampler, shuffle=sampler is None, drop_last=len(train_ds) > t.batch_size, **kw),
                DataLoader(val_ds, shuffle=False, **kw))

    def configure_phase(self, model: nn.Module, phase: PhaseCfg) -> list[dict]:
        return param_groups(model, phase, self.cfg.train.weight_decay)

    def loss(self, model: nn.Module, batch, device: torch.device) -> torch.Tensor:
        x, tok = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tok.reshape(-1), ignore_index=0,
                               label_smoothing=self.cfg.train.label_smoothing)

    def evaluate(self, model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
        loss_sum, n, scores, ids, ys, rost, free = 0.0, 0, [], [], [], [], []
        for x, tok, y, ros in loader:
            logits = model(x.to(device))
            tok = tok.to(device)
            loss_sum += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tok.reshape(-1),
                                        ignore_index=0, reduction="sum").item()
            n += int((tok != 0).sum())
            scores.append(score_numbers(logits).cpu())
            free += [a == b for a, b in zip(decode_free(logits.argmax(-1).cpu().numpy()),
                                            [NUMBERS[i] for i in y.tolist()])]
            ys.append(y)
            rost.append(ros)
        s, y, ros = torch.cat(scores), torch.cat(ys), torch.cat(rost)
        pred = s.argmax(-1)
        within = s.masked_fill(~ros, float("-inf")).argmax(-1)
        recalls = [(pred[y == c] == c).float().mean().item() for c in y.unique().tolist()]
        return {"val_loss": loss_sum / max(n, 1), "val_acc": (pred == y).float().mean().item(),
                "val_balanced_acc": float(np.mean(recalls)),
                "val_within_team_acc": (within == y).float().mean().item(),
                "val_free_acc": float(np.mean(free))}
