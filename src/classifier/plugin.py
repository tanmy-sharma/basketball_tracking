"""ClassifierPlugin: DINOv2 / ResNet-34 with an attachable number or eligibility head."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.classifier.dataset import CropDataset, balanced_sampler, worker_init
from src.classifier.heads import HEAD_CLASSES
from src.classifier.model import MultiHeadClassifier, build_model, llrd_param_groups
from src.classifier.transforms import build_transform
from src.core.config import PhaseCfg
from src.core.constants import NUMBERS
from src.core.paths import resolve_path
from src.core.plugin import LoopPlugin, WeightSource
from src.core.run import Run
from src.dataio.select import training_rows
from src.dataio.store import CropStore, crops_root, load_manifest

CLASS_NAMES = {"number": list(NUMBERS), "eligibility": ["ineligible", "eligible"]}
MIN_PER_CLASS = 100


def label_column(head: str) -> str:
    return "number_idx" if head == "number" else "label"


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    present = np.unique(y)
    return float(np.mean([(pred[y == c] == c).mean() for c in present]))


class ClassifierPlugin(LoopPlugin):
    def build_model(self) -> MultiHeadClassifier:
        return build_model(self.cfg)

    def _rows(self, run: Run) -> pd.DataFrame:
        cfg = self.cfg
        rows = training_rows(load_manifest(crops_root(cfg.crops), cfg.games.train), cfg)
        if cfg.model.head == "eligibility":
            lab = pd.read_csv(resolve_path(cfg.data.label_csv), dtype={"crop_id": str})
            rows = rows.merge(lab[["crop_id", "label"]], on="crop_id", how="inner")
            rows = rows[rows.label.isin([0, 1])].reset_index(drop=True)
            for s in ("train", "val"):
                if rows[rows.split == s].label.nunique() < 2:
                    raise ValueError(f"eligibility labels in {s} need both classes")
        return rows

    def build_loaders(self, run: Run) -> tuple[DataLoader, DataLoader]:
        cfg, t = self.cfg, self.cfg.train
        rows = self._rows(run)
        col = label_column(cfg.model.head)
        n_cls = HEAD_CLASSES[cfg.model.head]
        counts = {s: np.bincount(rows[rows.split == s][col], minlength=n_cls).tolist() for s in ("train", "val")}
        run.update_manifest(class_counts=counts, n_rows={s: int((rows.split == s).sum()) for s in counts})
        for c, n in enumerate(counts["train"]):
            if n < MIN_PER_CLASS:
                run.log(f"WARNING: class {CLASS_NAMES[cfg.model.head][c]} has only {n} train rows")
        store = CropStore(crops_root(cfg.crops))
        size = cfg.model.input_size
        train = CropDataset(rows[rows.split == "train"], store, build_transform(size, cfg.aug, True), col, cfg.seed)
        val = CropDataset(rows[rows.split == "val"], store, build_transform(size, None, False), col, cfg.seed)
        kw = dict(batch_size=t.batch_size, num_workers=t.num_workers, worker_init_fn=worker_init,
                  pin_memory=torch.cuda.is_available())
        sampler = balanced_sampler(train.labels, cfg.seed) if t.class_balance else None
        return (DataLoader(train, sampler=sampler, shuffle=sampler is None, drop_last=len(train) > t.batch_size, **kw),
                DataLoader(val, shuffle=False, **kw))

    def configure_phase(self, model: MultiHeadClassifier, phase: PhaseCfg) -> list[dict]:
        return llrd_param_groups(model, self.cfg.model.head, phase, self.cfg.train.weight_decay)

    def set_train_mode(self, model: MultiHeadClassifier, phase: PhaseCfg) -> None:
        """Frozen BatchNorm layers stay in eval so running statistics do not drift."""
        for m in model.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
                m.eval()

    def checkpoint_meta(self) -> dict:
        m = self.cfg.model
        return {"head": m.head, "classes": CLASS_NAMES[m.head], "backbone": m.backbone.name}

    def loss(self, model: MultiHeadClassifier, batch, device: torch.device) -> torch.Tensor:
        x, y = batch
        logits = model(x.to(device), self.cfg.model.head)
        return F.cross_entropy(logits.float(), y.to(device), label_smoothing=self.cfg.train.label_smoothing)

    def evaluate(self, model: MultiHeadClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
        head = self.cfg.model.head
        logits, ys = [], []
        for x, y in loader:
            logits.append(model(x.to(device), head).float().cpu())
            ys.append(y)
        lg, y = torch.cat(logits), torch.cat(ys)
        pred = lg.argmax(1).numpy()
        yn = y.numpy()
        out = {"val_loss": F.cross_entropy(lg, y).item(), "val_acc": float((pred == yn).mean()),
               "val_balanced_acc": balanced_accuracy(pred, yn)}
        if head == "number":
            allowed = torch.as_tensor(loader.dataset.allowed)
            masked = lg.masked_fill(~allowed, float("-inf")).argmax(1).numpy()
            out["val_within_team_acc"] = float((masked == yn).mean())
        return out

    def weight_sources(self) -> list[WeightSource]:
        b = self.cfg.model.backbone
        if not b.pretrained:
            return []
        if b.name == "dino":
            return [WeightSource("dinov2-small", f"https://huggingface.co/{b.hf_id}", b.hf_revision,
                                 trained_on="LVD-142M self-supervised (DINOv2)")]
        from src.classifier.backbone_resnet import RESNET34_URL
        return [WeightSource("resnet34", RESNET34_URL, trained_on="ImageNet-1k")]
