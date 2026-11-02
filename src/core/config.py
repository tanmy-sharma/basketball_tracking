"""Pydantic config schema, YAML `_base` inheritance, `--set` overrides and hashing."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.constants import CHALLENGE_VIDEOS
from src.core.paths import resolve_path

GAME_RE = re.compile(r"IMG_\d{4}")


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_games(games: list[str]) -> list[str]:
    for g in games:
        if not GAME_RE.fullmatch(g):
            raise ValueError(f"bad game id {g!r}, expected IMG_dddd")
        if g in CHALLENGE_VIDEOS:
            raise ValueError(f"{g} is a challenge video and must never be used")
    return games


class GamesCfg(_M):
    train: list[str]
    test: list[str] = []
    label_pool: list[str] = []

    @field_validator("train", "test", "label_pool")
    @classmethod
    def _games(cls, v: list[str]) -> list[str]:
        return _check_games(v)

    @model_validator(mode="after")
    def _no_overlap(self) -> GamesCfg:
        both = set(self.test) & (set(self.train) | set(self.label_pool))
        if both:
            raise ValueError(f"test games overlap train/label_pool: {sorted(both)}")
        return self


class SplitCfg(_M):
    val_start_frac: float = .8
    val_end_frac: float = 1.
    guard_frames: int = 150


class CropCfg(_M):
    every: int = 5
    height: Literal[720] = 720
    pad_frac: float = .15
    min_aspect: float = .4
    min_side_px: int = 8
    limit_frames: int | None = None
    pose_model: str = "usyd-community/vitpose-base-simple"
    pose_revision: str = "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45"


class ExcludeRule(_M):
    bib: str
    numbers: list[str]


class FilterCfg(_M):
    max_iou: float | None = .1
    min_pose_conf: float = .5
    exclude: list[ExcludeRule] = []
    subsample_every: int = 1
    include_csv: str | None = None


class AugCfg(_M):
    max_rotate_deg: float = 8.
    subcrop_frac: float = .08
    grey_p: float = .3
    hue_deg: float = 18.
    sat: float = .4
    brightness: float = .2
    contrast: float = .2
    blur_p: float = .2
    jpeg_p: float = .5
    jpeg_q: tuple[int, int] = (50, 95)

    @field_validator("max_rotate_deg")
    @classmethod
    def _rot(cls, v: float) -> float:
        if v > 15:
            raise ValueError("max_rotate_deg must be <= 15 (digits change meaning when rotated)")
        return v


class BackboneCfg(_M):
    name: Literal["dino", "resnet34"] = "dino"
    pretrained: bool = True
    hf_id: str = "facebook/dinov2-small"
    hf_revision: str = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    config_overrides: dict[str, Any] = {}  # tiny random model in tests


class ModelCfg(_M):
    backbone: BackboneCfg = Field(default_factory=BackboneCfg)
    head: Literal["number", "eligibility"] = "number"
    head_hidden: int = 256
    dropout: float = .2
    input_size: int = 224


class PhaseCfg(_M):
    name: str
    epochs: int
    unfreeze_blocks: int
    head_lr: float
    backbone_lr: float = 0.
    llrd: float = .75
    warmup_frac: float = .05


class TrainCfg(_M):
    batch_size: int = 128
    num_workers: int = 8
    label_smoothing: float = .1
    weight_decay: float = .05
    grad_clip: float = 1.
    monitor: str = "val_balanced_acc"
    patience: int = 4
    class_balance: bool = True
    phases: list[PhaseCfg]


class DataCfg(_M):
    label_csv: str | None = None  # crop_id,label; required when head == "eligibility"


class WeakCfg(_M):
    name: str
    folds: list[list[str]]  # each inner list is a held-out group; the model for it trains on the other groups
    fold_plugin: Literal["recognizer", "classifier"] = "recognizer"
    tau_pos: float = .7
    tau_neg: float = .3
    head_threshold: float = .7
    fill_min_per_class: int = 300
    fold_subsample_every: int = 2
    fold_phases: list[PhaseCfg]

    @field_validator("folds")
    @classmethod
    def _folds(cls, v: list[list[str]]) -> list[list[str]]:
        flat = [g for grp in v for g in _check_games(grp)]
        if len(flat) != len(set(flat)):
            raise ValueError("a game appears in more than one fold")
        if len(v) < 2 or not all(v):
            raise ValueError("need at least 2 non-empty folds")
        return v


class RecognizerCfg(_M):
    url: str = ("https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/"
                "vitstr_small_patch16_224_aug.pth")
    sha256: str = "3ff5c0490dabcb99b8540e66feb61d60e57821b55dabaacecbf31920824e42aa"
    input_size: int = 224
    seqlen: int = 4
    resize: Literal["letterbox", "stretch"] = "letterbox"
    pretrained: bool = True
    config_overrides: dict[str, Any] = {}  # tiny random model in tests


class DetectorCfg(_M):
    resolution: int = 512
    sample_every: int = 15
    jpeg_quality: int = 95
    epochs: int = 30
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    num_workers: int = 8
    patience: int = 8
    checkpoint_interval: int = 10
    conf_floor: float = .05
    predict_every: int = 5
    predict_batch: int = 8


class Config(_M):
    run_name: str
    plugin: str  # registry name, or "module:Class"
    seed: int = 0
    device: str = "auto"
    games: GamesCfg
    split: SplitCfg = Field(default_factory=SplitCfg)
    crops: CropCfg = Field(default_factory=CropCfg)
    filters: FilterCfg = Field(default_factory=FilterCfg)
    aug: AugCfg = Field(default_factory=AugCfg)
    model: ModelCfg | None = None
    train: TrainCfg | None = None
    data: DataCfg = Field(default_factory=DataCfg)
    weak: WeakCfg | None = None
    detector: DetectorCfg | None = None
    recognizer: RecognizerCfg | None = None

    @model_validator(mode="after")
    def _plugin_needs(self) -> Config:
        if self.plugin == "classifier" and (self.model is None or self.train is None):
            raise ValueError("plugin 'classifier' needs `model` and `train`")
        if self.plugin == "detector" and self.detector is None:
            raise ValueError("plugin 'detector' needs `detector`")
        if self.plugin == "recognizer" and (self.recognizer is None or self.train is None):
            raise ValueError("plugin 'recognizer' needs `recognizer` and `train`")
        if self.model is not None and self.model.head == "eligibility" and not self.data.label_csv:
            raise ValueError("head 'eligibility' needs data.label_csv")
        return self


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular _base: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("_base", None)
    if base is None:
        return raw
    return _deep_merge(_load_yaml(path.parent / base, seen + (path,)), raw)


def _apply_override(d: dict, item: str) -> None:
    key, sep, val = item.partition("=")
    if not sep or not key:
        raise ValueError(f"override must look like a.b=c, got {item!r}")
    *parents, leaf = key.strip().split(".")
    for p in parents:
        if not isinstance(d.get(p), dict):
            d[p] = {}
        d = d[p]
    d[leaf] = yaml.safe_load(val)


def load_config(path: Path | str, overrides: list[str] = ()) -> Config:
    d = _load_yaml(Path(path))
    for o in overrides:
        _apply_override(d, o)
    return Config.model_validate(d)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]


def config_hash(cfg: Config) -> str:
    """sha256[:8] of the config minus runtime-only fields, plus the content of include_csv/label_csv."""
    d = cfg.model_dump(mode="json")
    d.pop("device")
    for sec in ("train", "detector"):
        if d.get(sec):
            d[sec].pop("num_workers", None)
    files = {}
    for name, p in (("include_csv", cfg.filters.include_csv), ("label_csv", cfg.data.label_csv)):
        if p:
            fp = resolve_path(p)
            files[name] = _sha256_file(fp) if fp.exists() else "missing"
    d["_files"] = files
    return _digest(d)


def crop_hash(crops: CropCfg) -> str:
    return _digest(crops.model_dump(mode="json"))
