from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from src.core.config import Config, PhaseCfg
    from src.core.run import Run


@dataclass
class WeightSource:
    name: str
    url: str
    revision: str | None = None
    sha256: str | None = None
    trained_on: str = ""


@dataclass
class FitResult:
    best_path: Path
    metric_name: str
    metric: float
    extra: dict = field(default_factory=dict)


class ModelPlugin(ABC):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def fit(self, run: Run) -> FitResult: ...

    def weight_sources(self) -> list[WeightSource]:
        return []


class LoopPlugin(ModelPlugin):
    @abstractmethod
    def build_model(self) -> nn.Module: ...

    @abstractmethod
    def build_loaders(self, run: Run) -> tuple[DataLoader, DataLoader]: ...

    @abstractmethod
    def configure_phase(self, model: nn.Module, phase: PhaseCfg) -> list[dict]:
        """Freeze/unfreeze for the phase and return optimizer param groups (each with `lr`)."""

    @abstractmethod
    def loss(self, model: nn.Module, batch: Any, device: torch.device) -> torch.Tensor: ...

    @abstractmethod
    def evaluate(self, model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]: ...

    def set_train_mode(self, model: nn.Module, phase: PhaseCfg) -> None:
        """Called after model.train() every epoch (e.g. keep frozen BatchNorm in eval)."""

    def checkpoint_meta(self) -> dict:
        """Optional overrides for best.pt keys `head`, `classes`, `backbone`."""
        return {}

    def fit(self, run: Run) -> FitResult:
        from src.core.loop import fit_loop
        return fit_loop(self, run)


PLUGINS: dict[str, str | type[ModelPlugin]] = {
    "classifier": "src.classifier.plugin:ClassifierPlugin",
    "detector": "src.detector.plugin:DetectorPlugin",
    "recognizer": "src.recognizer.plugin:RecognizerPlugin",
}


def register_plugin(name: str, target: str | type[ModelPlugin]) -> None:
    PLUGINS[name] = target


def get_plugin(name: str) -> type[ModelPlugin]:
    """Lazy import so optional dependencies (rfdetr) are only needed by the plugin that uses them."""
    target = PLUGINS.get(name, name if ":" in name else None)
    if target is None:
        raise KeyError(f"unknown plugin {name!r}; registered: {sorted(PLUGINS)}")
    if isinstance(target, str):
        mod, _, attr = target.partition(":")
        target = getattr(importlib.import_module(mod), attr)
    return target
