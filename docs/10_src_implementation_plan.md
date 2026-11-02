# Plan: `src/` package for RF-DETR fine-tune, jersey classifier, weak eligibility labels and Colab runs

> **Read §8 first.** It records later decisions and overrides anything below that conflicts with it.

Nothing was written or changed. Only the local caches may have been touched. API facts below come from reading or running the installed code.

## 0. Findings that change the design

1. **`src/data/` would be silently gitignored.** `.gitignore` line 2 (`data/`) matches at any depth, and `git check-ignore` confirmed it. The package uses `src/dataio/` instead, and no directory under `src/` may be named `data` or `outputs`.

2. **Leave-one-game-out with only 0106 and 0107 is nearly useless.**
   - Holding out 0106 means the model trained on 0107 covers only numbers 6 and 9.
   - Holding out 0107 means the model trained on 0106 also covers only 6 and 9.
   - A model cannot read a number it never saw, so 8 of the 12 players in those two games would get no label. Those crops must be marked "unknown", not "ineligible".
   - Adding IMG_0109, which you gave a Drive ID for, fixes almost all of this. Only #11 in 0106 stays uncovered.
   - The default weak-label pool is therefore `[0106, 0107, 0109]`, never 0108.
   - Final number training stays on 0106 + 0107, as you confirmed. The IMG_0109 decision is yours to veto (see §7).

3. **docTR is already in the environment, against your hard rule.**
   - `inference-models` hard-requires `python-doctr[torch]`, and `trackers[detection]` pulls that in. It is in `uv.lock` and `.venv`.
   - It ships a text-recognition model the user has ruled out, so `doctr` must never be imported.
   - `rf-groundingdino` is also pulled in, and 5 of its files mention mmcv, mmdet or mmengine.
   - None of this is new. `src/` never imports them.
   - The Colab notebooks exclude them from install with `--no-install-package`, and a test enforces this.
   - The Mac `.venv` and `uv.lock` still contain them, so you need to decide how to handle that (see §7).

4. **ViTPose has mmpose ancestry.** The `transformers` port has no mmpose dependency. However, its processor cites functions sourced from `open-mmlab/mmpose`, and the upstream ViTPose repo that produced the checkpoint is built on mmpose. You named this model explicitly, so the plan uses it, but check this against your NDA (see §7).

5. **RF-DETR must be wrapped, not fitted into a custom loop.** Its `train()` runs PyTorch Lightning. A second version of the loop would be pretending. Both kinds of model share the entry point, config and run directory, but not the loop.

Measured on real labels:

| Measurement | Result |
|---|---|
| `track_identities` | Correct team and jersey on all 8 games |
| Stray boxes | 1 in 0110, 0 elsewhere |
| Boxes with IoU above 0.1 against another player | About 31-33% on 0106-0109 |
| Boxes with IoU above 0.3 | About 12-14% on 0106-0109 |
| ViTPose preprocessing on a synthetic frame | About 8 ms per box on the Mac, so the built-in processor is fast enough |

## 1. Folder tree

```
src/
  __init__.py                     empty; run everything as `python -m src.x` from repo root
  README.md                       commands, disclosures table, reproduction steps
  train.py                        general entry point: config -> run dir -> plugin.fit
  core/
    __init__.py
    paths.py                      ROOT, OUTPUTS, add_tools_to_path()
    constants.py                  NUMBERS, BIB_OF_ROSTER, CHALLENGE_VIDEOS, FORBIDDEN_MODULES
    config.py                     pydantic schema, YAML `_base` inheritance, --set overrides, hashing
    run.py                        run dir, manifest, logging, done sentinel
    checkpoint.py                 save/load best.pt and last.pt
    plugin.py                     ModelPlugin, LoopPlugin, registry
    loop.py                       generic loop: phases, AMP, early stop, resume
  dataio/
    __init__.py
    schema.py                     crop manifest columns and validation
    store.py                      CropWriter, CropStore, load/save manifest, crops_root
    split.py                      contiguous-block val split, guard band
    select.py                     training_rows: filters plus test-game guard
    labels.py                     bridge to tools/trackid: labelled boxes, stray dropping
    video.py                      ffmpeg frame iterator, frame count, video path
  crops/
    __init__.py
    geometry.py                   torso box, facing, max IoU
    pose.py                       PoseEstimator protocol, ViTPose wrapper
    extract.py                    CLI: one game -> crops.bin, manifest, keypoints
  classifier/
    __init__.py
    backbone_base.py              Backbone ABC
    backbone_dino.py              DinoBackbone (facebook/dinov2-small)
    backbone_resnet.py            ResNet34Backbone (torchvision)
    heads.py                      ClassifierHead, HEAD_CLASSES
    model.py                      MultiHeadClassifier, mask_logits, llrd_param_groups, build_model
    transforms.py                 train/eval transforms, no flip
    dataset.py                    CropDataset, class-balanced sampler
    plugin.py                     ClassifierPlugin (LoopPlugin)
    predict.py                    predict_crops
  weak/
    __init__.py
    folds.py                      CLI: fold configs, train folds, predict held-out, assemble
    assemble.py                   weak-label rules, coverage table (pure)
    finalize.py                   CLI: eligible_final.csv
    qa.py                         contact sheets, eligibility by facing
  evaluate/
    __init__.py
    metrics.py                    pure metric functions
    classifier.py                 CLI: report on the test game
  detector/
    __init__.py
    coco.py                       Roboflow-layout COCO export at 720p
    plugin.py                     DetectorPlugin wrapping RFDETRSmall.train
    predict.py                    CLI: MOT-format detections, stock or fine-tuned
    evaluate.py                   CLI: stock vs fine-tuned using tools/eval_detections.py
  configs/
    classifier/base.yaml, number_all.yaml, eligibility.yaml, number_eligible.yaml,
               number_eligible_resnet34.yaml, xcolor_no_yellow.yaml, xcolor_no_green.yaml, number_submit.yaml
    classifier/weak_pool3.yaml, weak_pool2.yaml, weak_pool4.yaml
    detector/rfdetr_small.yaml
    smoke/classifier_mac.yaml, detector_mac.yaml
  tests/
    __init__.py, conftest.py, fixtures.py
    test_core.py, test_loop.py, test_dataio.py, test_compliance.py        WP0
    test_labels.py, test_video.py, test_geometry.py, test_extract.py      WP1
    test_backbones.py, test_transforms.py, test_classifier_plugin.py      WP2
    test_weak.py, test_metrics.py, test_eval_cli.py                       WP3
    test_coco.py, test_detector_plugin.py, test_detector_predict.py       WP4
    test_notebooks.py, test_e2e_synthetic.py                              WP5
colab/
  build_src_notebooks.py          new; the existing build_notebook.py is untouched
  train_classifier.ipynb          generated
  train_detector.ipynb            generated
```

## 2. Interfaces

### Constants (`core/constants.py`)

```python
NUMBERS = ("1", "3", "4", "5", "6", "9", "10", "11", "12")   # index = class id
NUMBER_TO_IDX = {n: i for i, n in enumerate(NUMBERS)}
BIB_OF_ROSTER = {frozenset({"6", "9", "11"}): "green", frozenset({"6", "9", "10"}): "yellow",
                 frozenset({"1", "3", "5"}): "pink", frozenset({"4", "10", "12"}): "purple"}
CHALLENGE_VIDEOS = frozenset({"IMG_0114", "IMG_0115"})
FORBIDDEN_MODULES = ("mmdet", "mmcv", "mmengine", "mmpose", "mmocr", "doctr", "strhub",
                     "ultralytics", "groundingdino")
```

### Config schema (`core/config.py`, pydantic v2, `extra="forbid"`, frozen)

```python
class GamesCfg:  train: list[str]; test: list[str] = []; label_pool: list[str] = []
class SplitCfg:  val_start_frac: float = .8; val_end_frac: float = 1.; guard_frames: int = 150
class CropCfg:   every: int = 5; height: Literal[720] = 720; pad_frac: float = .15; min_aspect: float = .4
                 min_side_px: int = 8; limit_frames: int | None = None
                 pose_model: str = "usyd-community/vitpose-base-simple"
                 pose_revision: str = "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45"
class ExcludeRule: bib: str; numbers: list[str]
class FilterCfg: max_iou: float | None = .1; min_pose_conf: float = .5; exclude: list[ExcludeRule] = []
                 subsample_every: int = 1; include_csv: str | None = None
class AugCfg:    max_rotate_deg: float = 8.; subcrop_frac: float = .08; grey_p: float = .3; hue_deg: float = 18.
                 sat: float = .4; brightness: float = .2; contrast: float = .2; blur_p: float = .2
                 jpeg_p: float = .5; jpeg_q: tuple[int, int] = (50, 95)
class BackboneCfg: name: Literal["dino", "resnet34"] = "dino"; pretrained: bool = True
                 hf_id: str = "facebook/dinov2-small"
                 hf_revision: str = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
                 config_overrides: dict = {}        # tiny random model in tests
class ModelCfg:  backbone: BackboneCfg; head: Literal["number", "eligibility"] = "number"
                 head_hidden: int = 256; dropout: float = .2; input_size: int = 224
class PhaseCfg:  name: str; epochs: int; unfreeze_blocks: int; head_lr: float; backbone_lr: float = 0.
                 llrd: float = .75; warmup_frac: float = .05
class TrainCfg:  batch_size: int = 128; num_workers: int = 8; label_smoothing: float = .1; weight_decay: float = .05
                 grad_clip: float = 1.; monitor: str = "val_balanced_acc"; patience: int = 4
                 class_balance: bool = True; phases: list[PhaseCfg]
class DataCfg:   label_csv: str | None = None   # crop_id,label; required when head == "eligibility"
class WeakCfg:   name: str; pool: list[str]; tau_pos: float = .7; tau_neg: float = .3; head_threshold: float = .7
                 fill_min_per_class: int = 300; fold_subsample_every: int = 2; fold_phases: list[PhaseCfg]
class DetectorCfg: resolution: int = 512; sample_every: int = 15; jpeg_quality: int = 95; epochs: int = 30
                 batch_size: int = 16; grad_accum_steps: int = 1; lr: float = 1e-4; lr_encoder: float = 1.5e-4
                 num_workers: int = 8; patience: int = 8; checkpoint_interval: int = 10
                 conf_floor: float = .05; predict_every: int = 5; predict_batch: int = 8
class Config:    run_name: str; plugin: str; seed: int = 0; device: str = "auto"
                 games: GamesCfg; split: SplitCfg = SplitCfg(); crops: CropCfg = CropCfg()
                 filters: FilterCfg = FilterCfg(); aug: AugCfg = AugCfg(); model: ModelCfg | None = None
                 train: TrainCfg | None = None; data: DataCfg = DataCfg()
                 weak: WeakCfg | None = None; detector: DetectorCfg | None = None

def load_config(path: Path | str, overrides: list[str] = ()) -> Config  # `_base:` deep merge; `a.b=c` parsed as YAML
def config_hash(cfg: Config) -> str   # sha256[:8]: canonical JSON minus device, data.num_workers-like
                                      # runtime fields, plus the sha256 of include_csv and label_csv if present
def crop_hash(crops: CropCfg) -> str  # sha256[:8] of all fields
```

Validators must reject:
- Any game not matching `IMG_\d{4}`.
- Any game in `CHALLENGE_VIDEOS`, in any list.
- Any overlap between `test` and `train ∪ label_pool`.
- `max_rotate_deg > 15`.
- `plugin == "classifier"` without `model` and `train`, or `plugin == "detector"` without `detector`.
- `head == "eligibility"` without `label_csv`.

`AugCfg` has no flip field, and `extra="forbid"` makes `hflip` a load error.

### Plugin (`core/plugin.py`)

```python
@dataclass
class WeightSource: name: str; url: str; revision: str | None = None; sha256: str | None = None; trained_on: str = ""
@dataclass
class FitResult: best_path: Path; metric_name: str; metric: float; extra: dict = field(default_factory=dict)

class ModelPlugin(ABC):
    def __init__(self, cfg: Config) -> None: self.cfg = cfg
    @abstractmethod
    def fit(self, run: Run) -> FitResult: ...
    def weight_sources(self) -> list[WeightSource]: return []

class LoopPlugin(ModelPlugin):
    @abstractmethod
    def build_model(self) -> nn.Module: ...
    @abstractmethod
    def build_loaders(self, run: Run) -> tuple[DataLoader, DataLoader]: ...
    @abstractmethod
    def configure_phase(self, model: nn.Module, phase: PhaseCfg) -> list[dict]: ...   # freezes; returns param groups
    @abstractmethod
    def loss(self, model: nn.Module, batch: Any, device: torch.device) -> torch.Tensor: ...
    @abstractmethod
    def evaluate(self, model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]: ...
    def fit(self, run: Run) -> FitResult: return fit_loop(self, run)

PLUGINS = {"classifier": "src.classifier.plugin:ClassifierPlugin", "detector": "src.detector.plugin:DetectorPlugin"}
def register_plugin(name: str, target: str | type[ModelPlugin]) -> None: ...
def get_plugin(name: str) -> type[ModelPlugin]: ...     # imports lazily so rfdetr stays optional
```

### Run directory (`core/run.py`)

```
outputs/runs/<run_name>__<config_hash>/
  config.json      manifest.json      log.txt      metrics.csv
  last.pt          best.pt            summary.json  done.json     rfdetr/ (detector only)
```

```python
def resolve_run_dir(cfg: Config) -> Path
def open_run(cfg: Config, argv: list[str] | None = None) -> Run
class Run:   # cfg, dir, hash
    def log(self, msg: str) -> None
    def log_metrics(self, row: dict) -> None
    def update_manifest(self, **fields) -> None
    def finish(self, result: FitResult) -> None     # summary.json, then done.json by atomic rename
    @property
    def is_done(self) -> bool
```

`manifest.json` holds:
- Git hash and dirty flag, or `CODE_VERSION.json` on Colab.
- Bundle sha256, seed, argv, config and config hash.
- Package versions for torch, transformers and rfdetr, plus GPU name.
- `weight_sources` and input files with sha256.
- Frame counts per game, start time and finish time.

Dirty state matters because nothing is committed. The bundle sha is the real pin for code.

### Checkpoint (`core/checkpoint.py`)

```python
# best.pt, loadable with weights_only=True:
{"format_version": 1, "plugin": str, "head": str, "classes": list[str], "backbone": str,
 "state_dict": dict, "config": dict, "epoch": int, "phase": str, "metric_name": str, "metric": float}
# last.pt adds: "optimizer", "scheduler", "phase_index", "epoch_in_phase", "best_metric", "bad_epochs", "rng"
def save_checkpoint(path: Path, payload: dict) -> None   # write .tmp, then os.replace
def load_checkpoint(path: Path, full: bool = False) -> dict
```

The detector's `best.pth` is a copy of RF-DETR's own `checkpoint_best_total.pth` in its native format. It loads through `RFDETRSmall.from_checkpoint(path)`.

### Crop store (`dataio/`)

```
outputs/crops/<crop_hash>/<video>/
  crops.bin        raw uint8 RGB concatenated, native 720p pixels
  manifest.csv     one row per labelled box on a sampled frame, including rejected rows
  keypoints.npy    float32 (N, 17, 3), same row order
  meta.json        decoded frames, expected frames, cfg, pose revision
  done.json        sentinel
```

Manifest columns, exactly (`schema.py: CROP_COLUMNS`):

```
crop_id, video, frame, track_id, team, jersey, number_idx, bib, in_game,
box_x, box_y, box_w, box_h, max_iou, pose_min_conf,
crop_x0, crop_y0, crop_x1, crop_y1, crop_w, crop_h, offset, facing, status
```

- `crop_id` is `f"{video}:{frame:06d}:{track_id:02d}"`.
- Boxes are in 720p pixels.
- `facing` is one of `front | back | side | unknown`.
- `status` is one of `ok | low_pose | degenerate`.
- `offset` is -1 when no pixels were stored.
- Rows with `low_pose` keep their pixels so the threshold can change later without re-extracting.

```python
class CropWriter:
    def __init__(self, video_dir: Path): ...
    def add(self, rgb: np.ndarray) -> int: ...      # returns byte offset
    def close(self) -> None: ...
class CropStore:
    def __init__(self, root: Path): ...             # opens np.memmap lazily per video, per worker
    def read(self, video: str, offset: int, w: int, h: int) -> np.ndarray: ...
def crops_root(crops: CropCfg) -> Path
def load_manifest(root: Path, videos: Iterable[str]) -> pd.DataFrame    # validates, jersey stays str, fails if not done
def save_manifest(video_dir: Path, df: pd.DataFrame) -> None
def assign_split(df: pd.DataFrame, split: SplitCfg) -> pd.Series        # train|val|guard, per video
def training_rows(df: pd.DataFrame, cfg: Config) -> pd.DataFrame        # adds `split`, applies every filter
def predict_crops(best_pt: Path, df: pd.DataFrame, store: CropStore, device: str, batch_size: int = 256) -> np.ndarray
    # classifier/predict.py; softmax (N, C) in df row order
```

## 3. Module specs

### `core/loop.py`

`fit_loop` steps:
1. Seed everything.
2. Pick the device: cuda, then mps, then cpu.
3. Build the model and loaders.
4. For each phase, call `configure_phase`, build AdamW, and run a linear warmup followed by cosine decay per step.

Other behaviour:
- Use bf16 autocast only on cuda. No GradScaler is needed.
- Clip gradients.
- Each epoch, run `evaluate`, append to `metrics.csv` and write `last.pt`. Write `best.pt` when the monitor improves.
- Apply early stopping only in phases with `unfreeze_blocks > 0`.
- Resume from `last.pt` at the right phase and epoch, rebuilding the optimizer before restoring state.
- Before validating, assert that the monitor key exists in `evaluate`'s output.

### `train.py`

```python
def run_training(cfg: Config, argv=None) -> FitResult   # returns stored result if run.is_done
```

CLI: `python -m src.train --config PATH [--set a.b=c ...]`.

Flow: open the run, write the manifest, call `plugin.fit`, merge `weight_sources` into the manifest, then finish. Exceptions leave `last.pt` and no `done.json`.

### `dataio/labels.py`

This is the only module that touches `tools/`.

```python
def load_labelled_boxes(video: str) -> pd.DataFrame
    # frame, track_id, x, y, w, h (4K), team, jersey, number_idx, bib, in_game
def roster_mask(video: str, team: str) -> np.ndarray   # bool (9,)
```

Behaviour:
- Call `add_tools_to_path()`, then `from trackid import io`.
- Use `io.load_bboxes`, `io.load_roster` and `io.track_identities`.
- **Drop and log any box whose track ID is not on the roster.** This covers the IMG_0110 track 0 bug.
- Drop duplicate `(frame, track_id)` rows, keeping the first, and log them.
- Refuse `CHALLENGE_VIDEOS`.

Why reuse rather than copy: `io.py` and `eval_detections.py` use `__file__`-based absolute paths, so the "run from `tools/`" assumption only affects the bare `trackid` import. A `sys.path` insert solves that, there is one parser, and it works on Colab because `REPO/data` is where `io.DATA` points.

Never import `tools/run_detector.py`, because it imports `inference_models` at the top level.

### `dataio/video.py`

```python
def video_path(video: str) -> Path                    # data/Training/vidoes/<video>.MOV, case-insensitive
def probe(path: Path) -> tuple[int, int]
def frame_count(path: Path) -> int
def iter_frames(path: Path, every: int, height: int = 720, limit: int | None = None) -> Iterator[tuple[int, np.ndarray]]
```

- Follow `tools/run_detector.py::frames`: `select='not(mod(n,N))',scale=W:H:flags=area`, passthrough sync with the `-fps_mode` / `-vsync` detection, and `-hwaccel auto`.
- Use `pix_fmt=rgb24`, because ViTPose and RF-DETR `predict` both expect RGB.
- Frame number is `n * every + 1`.
- Return the scale factor from a helper, and assert 16:9 for now.

### `crops/geometry.py`

```python
def torso_box(kps: np.ndarray, pad_frac: float, min_aspect: float, min_side: int, img_w: int, img_h: int
              ) -> tuple[int, int, int, int] | None
def facing(kps: np.ndarray) -> str
def max_iou_per_box(boxes_xywh: np.ndarray) -> np.ndarray
```

- `torso_box` takes the bounding box of keypoints 5, 6, 11 and 12, pads 15% of width and height on each side, then widens to `width >= min_aspect * height`. The widening prevents a slit crop for side views, where the shoulders overlap.
- It clamps to the image and returns None when a side is below `min_side`.
- `facing` compares shoulder x-positions, with `side` when the gap is under 0.25 of torso height. Person's left at image right means front. Confidences below 0.3 give `unknown`.

### `crops/pose.py`

```python
class PoseEstimator(Protocol):
    def __call__(self, frame_rgb: np.ndarray, boxes_xywh: np.ndarray) -> np.ndarray: ...   # (N, 17, 3)
class ViTPose:
    def __init__(self, model_id: str, revision: str, device: str): ...
```

Verified call sequence:

```python
inp = proc(images=[frame], boxes=[boxes_list], return_tensors="pt")
out = model(pixel_values=inp["pixel_values"].to(device))
res = proc.post_process_pose_estimation(out, boxes=[boxes_list])
# res[0][i]: "keypoints" (17, 2) and "scores" (17,)
```

- Do not pass `threshold`, or keypoints get dropped and indices shift.
- `dataset_index` is not needed for this model.
- Boxes are COCO xywh in 720p pixels.
- Forward per frame. Do not batch across frames.

### `crops/extract.py`

```python
def extract_game(video: str, crops: CropCfg, pose: PoseEstimator, out_root: Path | None = None) -> Path
```

CLI: `python -m src.crops.extract --config C --games ...`, which skips any game with `done.json`.

- Sample frames, divide boxes by the scale factor, and run pose.
- Compute `max_iou` per frame.
- **Cut pixels from the downscaled frame**, never from 4K.
- Decoded count must equal GT `frame.max()`. Fail hard otherwise, unless `limit_frames` is set.
- Write everything to `.tmp` names and rename only when finished.
- Log status counts and the median crop size, which should be near your measured 21x33 at 720p.
- Refuse `CHALLENGE_VIDEOS`.

### `dataio/select.py`

`training_rows` works in this order:
1. Raise if any row's video is in `games.test`, in `CHALLENGE_VIDEOS`, or not in `games.train`.
2. Keep `status == ok` and `pose_min_conf >= min_pose_conf`.
3. Apply `max_iou`.
4. Apply `exclude` rules.
5. Apply `include_csv`.
6. Apply `subsample_every` by frame (`frame_rank % n == 0`), never by row.
7. Assign `split` and drop `guard`.
8. Raise if train or val is empty.

### `dataio/split.py`

The val block is `[start_frac, end_frac)` of `df.frame.max()` per game, using the sampled range so that `limit_frames` still works. `guard_frames` on each side is excluded from both sides.

**Leakage limit, also stated in the README and report:** val shares players, bibs, lighting and background with train. Only near-duplicate frames are removed. It is used for early stopping only and is never reported as generalisation.

### `classifier/`

```python
class Backbone(nn.Module, ABC):
    feature_dim: int
    @abstractmethod
    def forward(self, pixels: torch.Tensor) -> torch.Tensor: ...     # (B, feature_dim)
    @abstractmethod
    def blocks(self) -> list[nn.Module]: ...                         # input to output
    def extra_top_modules(self) -> list[nn.Module]: return []        # trained with the last block

class MultiHeadClassifier(nn.Module):
    def __init__(self, backbone: Backbone, heads: dict[str, int], hidden: int = 256, dropout: float = .2): ...
    def add_head(self, name: str, num_classes: int) -> None: ...
    def forward(self, pixels: torch.Tensor, head: str) -> torch.Tensor: ...

def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor   # masked_fill(~allowed, -inf)
def llrd_param_groups(model, head: str, phase: PhaseCfg, weight_decay: float) -> list[dict]
def build_model(cfg: Config) -> MultiHeadClassifier
```

- **`backbone_dino.py`:** `Dinov2Model.from_pretrained(hf_id, revision=...)`. With `pretrained=False`, use `Dinov2Model(Dinov2Config(**config_overrides))`. The feature is `cat(last_hidden_state[:, 0], last_hidden_state[:, 1:].mean(1))`, so 768 dims for small. This matches `Dinov2ForImageClassification`. `blocks()` returns `list(self.net.encoder.layer)`, and `extra_top_modules()` returns `[self.net.layernorm]`. Position embeddings interpolate automatically at 224, confirmed in `Dinov2Embeddings.forward`.
- **`backbone_resnet.py`:** `resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)` with `fc = Identity`. `blocks()` returns `[layer1, layer2, layer3, layer4]`, and the feature dim is 512. `conv1` and `bn1` stay frozen. Frozen blocks must stay in `.eval()` during training so BatchNorm statistics do not drift, and the loop calls `model.train()` then re-applies this through a plugin hook inside `configure_phase`/`set_train_mode`.
- **`heads.py`:** `HEAD_CLASSES = {"number": 9, "eligibility": 2}`. The head is LayerNorm, Linear(hidden), GELU, Dropout, Linear(n).
- **`llrd_param_groups`:** the top unfrozen block gets `backbone_lr` and each lower block is multiplied by `llrd`. Parameters with `ndim <= 1` get no weight decay. The head gets `head_lr`.
- **`transforms.py`:** no flip anywhere, enforced by test.
  1. Sub-crop jitter.
  2. Small rotation with replicated borders.
  3. JPEG and blur at native resolution, to mimic H.264 720p.
  4. Letterbox to 224, preserving aspect ratio, with bicubic resize and ImageNet-mean pad. This matters because 21x33 stretched to square would distort digit shapes.
  5. Hue, saturation, brightness and contrast jitter.
  6. Greyscale with `grey_p`.
  7. ImageNet normalise.

  Eval uses only letterbox and normalise.
- **`dataset.py`:** `CropDataset(df, store, transform, label_col)` opens the memmap lazily in workers, and a worker init calls `cv2.setNumThreads(0)`. `number` uses `number_idx`. `eligibility` joins `label_csv`. The sampler is a seeded `WeightedRandomSampler` with inverse class frequency.
- **`plugin.py`:** `build_loaders` loads manifests for `games.train` and calls `training_rows`. `evaluate` returns `val_loss`, `val_acc` and `val_balanced_acc`, and for `number` also `val_within_team_acc`. Log per-class train counts in the summary and warn when a number has fewer than 100 rows. Weight sources:
  - dinov2-small with its revision, trained on ImageNet-1k
  - ResNet-34 from the torchvision URL
  - ViTPose goes in the crop `meta.json` and is merged into the run manifest.

### `weak/`

**Fold construction, in `folds.py`:**

```python
def fold_config(cfg: Config, held_out: str) -> Config
# games.train = pool - {held_out}; games.test = cfg.games.test + [held_out]; label_pool = []
# phases = weak.fold_phases; filters.subsample_every = weak.fold_subsample_every
# run_name = f"fold_{weak.name}_{held_out}"
```

Putting the held-out game in `test` means `training_rows` raises if it leaks into training. Raise if `len(pool) < 2`.

**Cheap by design:** every 2nd sampled frame, and 2 head epochs plus 4 fine-tune epochs with the last 2 blocks unfrozen. Each fold is roughly 3 minutes, with resumable fold runs.

**`assemble.py` rules:**

```python
def covered_numbers(train_df: pd.DataFrame) -> set[int]
def assemble_labels(df: pd.DataFrame, probs: np.ndarray, covered: set[int], train_bibs: set[str],
                    weak: WeakCfg, max_iou: float | None) -> pd.DataFrame
```

| Label | Meaning | Rule |
|---|---|---|
| 1 | eligible | Number covered, 9-way argmax equals truth, `p_true >= tau_pos` |
| 0 | ineligible | Number covered, `p_true < tau_neg` |
| -1 | ambiguous, not trained | Anything between |
| -2 | uncovered, not trained | Fold never saw this number |
| -3 | overlap, not trained | `max_iou` above threshold |

`weak_labels.csv` columns: `crop_id, video, number_idx, bib, facing, label, p_true, p_max, pred_idx, fold, fold_saw_bib`.
`eligibility_labels.csv` columns: `crop_id, label`, containing only 0 and 1.
`coverage.md` shows a game-by-number table of label percentages, marks `fold_saw_bib`, and prints a large warning when any number is uncovered.

**`qa.py`:** writes contact sheets of positive and negative crops per number, plus eligibility rate by facing and by `fold_saw_bib`.

These two are the check against person-memory contamination. If `side` crops are eligible as often as `front` crops, or if rates are far higher when `fold_saw_bib` is true, the labels are learning the person rather than reading digits.

**`finalize.py`:**

```python
def finalize_eligible(weak_df, elig_scores, weak: WeakCfg) -> pd.DataFrame   # crop_id, source, elig_score
```

- Label 1 crops are in, with source `weak`.
- Uncovered crops (-2) with head score at or above `head_threshold` are in, with source `head`.
- Labels 0, -1 and -3 are out. These uncovered crops were never in eligibility training, so their head scores are out-of-sample.
- If a number is below `fill_min_per_class`, take the top-scoring uncovered crops of that number, with source `fill`. Without this, final training could have no #11 class at all, and #11 is the known gap in the 3-game pool.

**Coverage by pool:**

| Pool | Held out | Trained on | Covered | Uncovered in held-out game |
|---|---|---|---|---|
| 2 games | 0106 | 0107 | 4, 6, 9, 10, 12 | 1, 3, 5, 11 |
| 2 games | 0107 | 0106 | 1, 3, 5, 6, 9, 11 | 4, 10, 12 |
| 3 games (default) | 0106 | 0107 + 0109 | all but 11 | 11 |
| 3 games | 0107 | 0106 + 0109 | all 9 | none |
| 3 games | 0109 | 0106 + 0107 | all 9 | none |
| 4 games (submission only) | any | the other three | all 9 | none |

**Using a 2-game pool:** the code runs, warns, and labels only 6 and 9. The eligibility head then generalises to other digits from those two alone, which is a weak claim. The report must say so. A within-game time-block fold is deliberately not built. The same person, bib and lighting appear on both sides, so "read correctly" would mean little.

### `evaluate/`

Pure functions in `metrics.py`:

```python
def per_crop_accuracy(y, probs) -> float
def within_team_accuracy(y, probs, allowed) -> float
def confusion(y, pred, n=9) -> np.ndarray
def accuracy_by_bib(df, probs, allowed, numbers=("6", "9", "10")) -> pd.DataFrame
def tracklet_vote(df, probs, weights, window_s, allowed=None, fps=29.97) -> pd.DataFrame
def coverage_curve(y, probs, scores, thresholds, allowed=None) -> pd.DataFrame
def auc(y, scores) -> float     # scipy rankdata
```

CLI: `python -m src.evaluate.classifier --game IMG_0108 --number NAME=CONFIG ... --eligibility CONFIG --out DIR [--allow-nontest]`.

It refuses a game that is not in the config's `games.test` unless `--allow-nontest` is set, and then prints a banner. Output: `metrics.json`, `report.md`, `confusion_<name>.csv`, `predictions_<name>.csv.gz`.

**Reported per run, on both `all` crops (pose ok) and `clean` crops (`max_iou` within threshold):**
- 9-way accuracy.
- Within-team 3-way accuracy, masked to the team's roster.
- Confusion matrix, with the 1/11 and 10/12 cells called out.
- Accuracy per track (6 players). This is the honest variance estimate, because crops are correlated.
- 6 vs 9 by bib colour, including the 6-to-9 and 9-to-6 confusion rates.
- Accuracy by facing.

**Tracklet votes:**
- The whole-game GT track is an easy case. Split tracks into windows of whole, 60 s, 10 s and 2 s to mimic real tracklets.
- Vote with `sum(p)` over eligible crops, both unmasked and roster-masked.
- Report accuracy and abstention rate for each.

**Coverage against accuracy:** thresholds `[0, .1, .3, .5, .7, .9]`, for the eligibility head and for plain max-softmax as a baseline. If max-softmax does as well, the head adds nothing, and that is worth knowing.

**Cross-colour:** two extra trainings on the same split.
- `xcolor_no_yellow` excludes yellow 6, 9 and 10, then tests on yellow in 0108.
- `xcolor_no_green` excludes green 6 and 9, then tests on green in 0108.

This is the only result here that isolates digit reading from person and bib memory. Note in the report that person memory is not purely harmful, because the challenge teams also appear in training. It is just fragile under the camera change.

### `detector/`

**`coco.py`:** `build_dataset(cfg) -> Path`, writing to `outputs/detector_dataset/<hash>/`:

```
train/_annotations.coco.json + 720p JPEGs
valid/_annotations.coco.json + 720p JPEGs
```

- Category: `[{"id": 1, "name": "player", "supercategory": "player"}]`. Using its own name as supercategory means `filter_parent_categories` keeps it, confirmed from source.
- Sample every `sample_every` frames. Use the same `assign_split` block for `valid`.
- Scale boxes to 720p, clamp, and drop boxes under 2 px.
- Each image has at least one box in practice. Skip frames with none.
- No `test/` folder. `run_test` defaults to False and `setup("test")` is only reached for testing.
- Refuse any game in `games.test`.

**`plugin.py`:**

```python
from rfdetr import RFDETRSmall          # imported lazily inside fit
model = RFDETRSmall()
model.train(dataset_dir=str(ds), output_dir=str(run.dir / "rfdetr"), epochs=..., batch_size=...,
            grad_accum_steps=..., lr=..., lr_encoder=..., resolution=..., seed=cfg.seed,
            num_workers=..., device=..., tensorboard=False, early_stopping=True,
            early_stopping_patience=..., checkpoint_interval=..., run_test=False,
            progress_bar=None, notes=run.hash, resume=<last.ckpt if present, else None>)
```

Every kwarg was checked against `TrainConfig` (which uses `extra="forbid"`) or the `_prepare_run_config` absorbed list. After training:
- Copy `rfdetr/checkpoint_best_total.pth` to `run.dir/best.pth`.
- Read `rfdetr/metrics.csv` for the metric. Report mAP at best epoch, and if the expected column is missing, log the column list and store NaN rather than failing a finished run.
- Weight source: `rf-detr-small.pth`, URL `https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth`, MD5 `fb37061c1af7bace359c91b723a8d5c1`, COCO-trained, Apache 2.0.

The default RF-DETR augmentation is HorizontalFlip only. That is acceptable for a person detector and is left as is. `num_classes` auto-aligns from the dataset, and an explicit value must not be passed.

**`predict.py`:**

CLI: `--weights stock|PATH --game G --every N --limit N`. Output, same format as `run_detector.py`, in full-resolution pixels:

```
outputs/detections/<game>__rfdetr-small-<stock|ft-hash>__720p[__everyN].txt
frame,-1,x,y,w,h,conf,-1,-1,-1
```

- Stock: `RFDETRSmall()`, keep `class_id == 1`, because stock models return COCO category IDs (source lines 2805 and 3085).
- Fine-tuned: `RFDETRSmall.from_checkpoint(path)`, keep `class_id == 0`.
- `predict(list_of_rgb, threshold=conf_floor, include_source_image=False)`, batched.
- Write to `.tmp` and rename when finished.
- Use `rfdetr` for both models, so that the comparison shares one preprocessing path.

**`evaluate.py`:**
- Import `eval_detections` through `add_tools_to_path()`.
- Capture `evaluate()` stdout into the report.
- Build a compact table from `load_dets`, `score`, `match` and `on_court`: recall, precision, F1 and foot-point error (median, p90, within 13/30/100 cm).
- Accept extra `--dets name=path` inputs.
- The report header must state the camera-shift risk (see §7).

### Configs

`classifier/base.yaml`:

```yaml
plugin: classifier
seed: 0
games: {train: [IMG_0106, IMG_0107], test: [IMG_0108]}
model: {backbone: {name: dino}, head: number}
train:
  phases:
    - {name: head, epochs: 3, unfreeze_blocks: 0, head_lr: 1.0e-3}
    - {name: finetune, epochs: 10, unfreeze_blocks: 3, head_lr: 3.0e-4, backbone_lr: 1.0e-4, llrd: 0.75}
```

| File | Contents |
|---|---|
| `number_all.yaml` | `_base: base.yaml`, `run_name: number_all` |
| `weak_pool3.yaml` | `weak: {name: pool3, pool: [IMG_0106, IMG_0107, IMG_0109], fold_phases: [2 head, 4 fine-tune with unfreeze 2]}` |
| `weak_pool2.yaml` | Same with pool of 2 |
| `weak_pool4.yaml` | Pool of all four, test empty, **submission only** |
| `eligibility.yaml` | `head: eligibility`, train on 0106, 0107, 0109, `label_csv: outputs/weak/pool3/eligibility_labels.csv` |
| `number_eligible.yaml` | `include_csv: outputs/weak/pool3/eligible_final.csv` |
| `number_eligible_resnet34.yaml` | Only `model.backbone.name: resnet34`; `unfreeze_blocks: 2` |
| `xcolor_no_yellow.yaml` | `exclude: [{bib: yellow, numbers: ["6", "9", "10"]}]` |
| `xcolor_no_green.yaml` | `exclude: [{bib: green, numbers: ["6", "9"]}]` |
| `number_submit.yaml` | Train on all four, test empty; only after you approve |

`detector/rfdetr_small.yaml`: `plugin: detector`, `games: {train: [IMG_0106, IMG_0107], test: [IMG_0108]}`, default `detector` block.

### Colab (`colab/build_src_notebooks.py`)

Reuse the existing patterns from `build_notebook.py`: tar+base64 bundle, code-version hash, `sh()`, `step()`/`save()`/`restore()` with done markers, and `drive.flush_and_unmount()`.

**Bundle:** `pyproject.toml`, `uv.lock`, all `src/**/*.py` and `src/configs/**/*.yaml` except `src/tests`, `tools/trackid/*.py`, `tools/eval_detections.py`, plus a generated `CODE_VERSION.json` (git hash, dirty flag, bundle sha). Nothing else.

**Install, in the notebook:**

```python
EXCLUDE = ["python-doctr", "rf-groundingdino"]
sh([sys.executable, "-m", "uv", "sync", "--frozen", "--no-dev", "--python", "3.12",
    *("--group rfdetr".split() if DETECTOR else []),
    *[a for p in EXCLUDE for a in ("--no-install-package", p)]])
```

**Data cell:**
- Annotations: `gdown.download_folder(id=...)`, with `1ONKvnUcfPG33mB9aYJBmj6aZMP_3QXvb` for boxes and `1JHXE8su46m9nRweCq0WonmPnNQhBbVwg` for court positions. `court_registration.csv` and `games_roster.csv` use the file IDs already in `build_notebook.py`.
- Videos, in order: runtime disk, then `My Drive/tracking_videos` (rglob as now), then `gdown` with your IDs.
- Verify each video with ffprobe frame count against GT `frame.max()`, since only 0106's byte size is known.
- On gdown quota failure, stop with a message saying to add the file to `My Drive/tracking_videos`.

**Notebooks:**
- `train_classifier.ipynb`: extract 4 games, `weak.folds`, eligibility, finalize, number_all, number_eligible, resnet34, two cross-colour runs, evaluate.
- `train_detector.ipynb`: train (which builds the dataset), predict stock on 0108, predict fine-tuned on 0108, evaluate.

**Results and auto-download:**
- Save to `My Drive/tracking_results/src_<notebook>/<code_version>/` after every step.
- The final cell, in a `finally` that also runs on failure, builds two zips:
  - `reports_<ver>.zip`, for reports, manifests, configs, logs, metrics and weak-label tables.
  - `weights_<ver>.zip`, for each `best.pt` / `best.pth`.
- Copy both to Drive, flush and unmount, then call `google.colab.files.download` on each inside try/except, printing the Drive path if it fails. Reports go first, because a browser may struggle with a large second file.
- Runtime `outputs/crops` is saved to Drive but left out of the zips, so a restart skips decoding.

The intro must disclose the public checkpoints used and tell the user to allow multiple downloads in the browser.

## 4. Pipeline order

### Mac

| Step | Command | Time |
|---|---|---|
| Add deps, once (WP0) | `uv add --no-sync "pydantic>=2.14.2" "pyyaml>=6.0.3"` then `uv add --no-sync --group rfdetr "rfdetr[train]==1.11.2"` | seconds |
| Tests | `uv run python -m pytest src/tests -q` | Under 2 min |
| Real-video smoke | See §6 | About 10 min |
| Build notebooks | `uv run python colab/build_src_notebooks.py` | seconds |

Other notes on dependencies:
- `--no-sync` leaves the shared `.venv` alone.
- `pydantic` and `pyyaml` are already installed transitively, so they are promoted to direct deps.
- `torch`, `torchvision`, `transformers`, `cv2`, `scipy` and `pycocotools` stay transitive through `trackers[detection]`. Add no pins for them, because `inference-models` already caps `transformers<5.16`.
- A dry-run universal resolve with `rfdetr[train]` kept `torch==2.15.0` and `transformers==5.15.1`, and added no mm-family or ultralytics packages.
- Check `git diff pyproject.toml` afterwards to confirm the change is additive, since that file belongs to the parallel session.

### Colab A100

These times are estimates, not measured. Run both notebooks one after the other, or at the same time if your plan allows two runtimes.

**`train_classifier.ipynb`, about 1.5 h:**

| Step | Command | Est. time |
|---|---|---|
| Setup, annotations, 4 videos | notebook cells | 15-25 min |
| Extract crops | `python -m src.crops.extract --config src/configs/classifier/base.yaml --games IMG_0106 IMG_0107 IMG_0109 IMG_0108` | 20-30 min |
| Folds and weak labels | `python -m src.weak.folds --config src/configs/classifier/weak_pool3.yaml` | 10-15 min |
| Eligibility head | `python -m src.train --config src/configs/classifier/eligibility.yaml` | 5-8 min |
| Finalize | `python -m src.weak.finalize --config .../weak_pool3.yaml --eligibility-config .../eligibility.yaml` | 2 min |
| number_all | `python -m src.train --config .../number_all.yaml` | 7 min |
| number_eligible | `python -m src.train --config .../number_eligible.yaml` | 5 min |
| ResNet-34 baseline | `python -m src.train --config .../number_eligible_resnet34.yaml` | 4 min |
| Cross-colour × 2 | `python -m src.train --config .../xcolor_no_yellow.yaml`, then `xcolor_no_green.yaml` | 12 min |
| Evaluate | `python -m src.evaluate.classifier --game IMG_0108 --number ... --eligibility ...` | 3 min |

**`train_detector.ipynb`, about 45-60 min:**

| Step | Command | Est. time |
|---|---|---|
| Setup, annotations, 3 videos | notebook cells | 12-20 min |
| Dataset and training | `python -m src.train --config src/configs/detector/rfdetr_small.yaml` | 25-35 min |
| Predict stock and fine-tuned on 0108 | `python -m src.detector.predict --weights stock --game IMG_0108 --every 5`, then `--weights .../best.pth` | 8 min |
| Evaluate | `python -m src.detector.evaluate --game IMG_0108 --stock ... --finetuned ...` | 2 min |

**Throughput assumptions:**
- `every=5` gives about 13.5k rows per game. After the 0.1 IoU filter and the pose check, about 8.9k remain.
- Stored crops are about 2 KB each, so `crops.bin` is about 30 MB per game. Memory and disk are not constraints.
- CPU augmentation, at roughly 600 images/s, is likely the bottleneck rather than the GPU.

## 5. Parallel work breakdown

Every file has one owner. Do not commit. Do not edit `tools/`, `colab/build_notebook.py`, or `colab/detect_track_IMG_0106.ipynb`.

```
WP0 (first, alone)  ->  WP1, WP2, WP3, WP4, WP5 authoring (parallel)  ->  WP5 E2E acceptance (last)
```

### WP0: Contracts and general train script (serial, about 45 min)

- **Owns:** `src/__init__.py`, `src/train.py`, `src/core/*`, `src/dataio/{__init__,schema,store,split,select}.py`, `src/tests/{__init__,conftest,fixtures,test_core,test_loop,test_dataio,test_compliance}.py`, empty `__init__.py` in every other subpackage, and the `uv add --no-sync` commands.
- **Provides in `fixtures.py`:**
  - `make_training_tree(root, games)`: synthetic rosters, boxes and court positions, using a fake homography so `track_identities` resolves.
  - `patch_trackid_paths(monkeypatch, root)`: patches `io.TRAIN`, `io.GT_BBOX`, `io.GT_COURT`.
  - `make_crop_store(root, video, n, numbers, rng)`: crops with `cv2.putText` digits, so a tiny net can learn them.
  - `make_video(path, n_frames, w, h)`: lossless, each frame encoding its index as colour.
  - `tiny_dino_overrides()`: `dict(image_size=56, patch_size=14, hidden_size=48, num_hidden_layers=4, num_attention_heads=2, mlp_ratio=2)`, verified to run.
- **Tests:**
  - Config: `_base` merge, `--set`, unknown-key rejection, `hflip` rejected, rotation above 15 rejected, 0114 rejected, test/train overlap rejected.
  - Hash: stable, and changes with `include_csv` content.
  - Run: resume skip on `done.json`, manifest fields present.
  - Checkpoint: atomic write, `best.pt` loads under `weights_only=True`.
  - Loop, with a toy plugin on random data: two phases, early stop, resume after a simulated crash gives the same epoch count.
  - dataio: store round-trip, split block with guard band, `training_rows` raising on a test game, `exclude`, `include_csv`, frame-level subsampling.
  - Compliance: AST scan of `src/` and `colab/build_src_notebooks.py` for `FORBIDDEN_MODULES`. A subprocess imports entry modules and checks `sys.modules`, using `importorskip`-style guards for modules not yet written. `uv.lock` contains no mmcv, mmdet, mmpose, mmocr, mmengine or ultralytics names. Every path under `src/` passes `git check-ignore` as not ignored.

### WP1: Labels, video, crops (parallel)

- **Owns:** `src/dataio/{labels,video}.py`, `src/crops/*.py`, `src/tests/{test_labels,test_video,test_geometry,test_extract}.py`.
- **Honours:** `CROP_COLUMNS`, `CropWriter`, `crops_root`, `CropCfg`.
- **Tests:**
  - Stray track is dropped and logged, and a synthetic track 0 is removed.
  - Duplicate rows are dropped, and bib and team are correct.
  - Frame alignment: `iter_frames(every=5)` frame numbers match the encoded colour.
  - `torso_box`: padding, clamp, `min_aspect` on a side view, None on degenerate input.
  - `facing` cases and `max_iou` values.
  - `extract_game` with a `FakePose` on a synthetic video: rows, offsets and statuses correct, and crops cut at 720p scale.
  - `ViTPose` wrapper with a random tiny `VitPoseConfig` and default `VitPoseImageProcessor`, no download, output `(N, 17, 3)`.
  - Decoded-count mismatch raises, and `limit_frames` skips the check.
  - Challenge video refused.

### WP2: Classifier (parallel)

- **Owns:** `src/classifier/*.py`, `src/configs/classifier/{base,number_all,eligibility,number_eligible,number_eligible_resnet34,xcolor_no_yellow,xcolor_no_green,number_submit}.yaml`, `src/tests/{test_backbones,test_transforms,test_classifier_plugin}.py`.
- **Honours:** `LoopPlugin`, `predict_crops`, `training_rows`, `CropStore`.
- **Tests:**
  - Tiny DINO: output dim is `2 * hidden`, and `blocks()` has 4 entries.
  - `ResNet34Backbone(pretrained=False)` gives dim 512, and frozen BatchNorm buffers do not change during training.
  - Swapping backbone through config changes nothing else.
  - `configure_phase`: requires_grad matches `unfreeze_blocks`, LLRD ratios exact, no-decay group correct.
  - `mask_logits` restricts argmax to the roster.
  - Transform output shape `(3, 224, 224)`, aspect preserved.
  - **No flip:** across 200 draws on an asymmetric image, correlation to the original exceeds correlation to its mirror every time, and an AST scan finds no `cv2.flip`, `hflip` or `[:, ::-1]`.
  - Plugin on a synthetic store, tiny backbone: `run_training` finishes on CPU, accuracy above chance in 60 s or less, and `predict_crops` preserves row order and shape.

### WP3: Weak labels and evaluation (parallel)

- **Owns:** `src/weak/*.py`, `src/evaluate/*.py`, `src/configs/classifier/{weak_pool2,weak_pool3,weak_pool4}.yaml`, `src/tests/{test_weak,test_metrics,test_eval_cli}.py`.
- **Honours:** `run_training`, `predict_crops` (stub with monkeypatch in unit tests), `load_manifest`.
- **Tests:**
  - `fold_config`: held-out game moves to `test`, and training on it raises.
  - `assemble_labels`: each of the 5 label values on hand-built probabilities.
  - `covered_numbers` reproduces the §3 coverage table for all three pool sizes, from rosters. This is a required test.
  - `finalize_eligible`: `weak`, `head` and `fill` sources, overlap and ineligible never admitted, `fill_min_per_class` honoured.
  - Metrics on hand-computed values: masked accuracy where the 9-way is wrong and the 3-way is right, confusion, vote with windows, abstention, coverage curve monotonic in coverage, AUC.
  - Eval CLI with stubbed predictions: refuses a non-test game without `--allow-nontest`, and writes all outputs.

### WP4: Detector (parallel)

- **Owns:** `src/detector/*.py`, `src/configs/detector/rfdetr_small.yaml`, `src/tests/{test_coco,test_detector_plugin,test_detector_predict}.py`.
- **Honours:** `ModelPlugin`, `Run`, `load_labelled_boxes`, `iter_frames`.
- **Tests, without installing rfdetr, using a fake `rfdetr` module injected into `sys.modules`:**
  - COCO export on a synthetic video loads in `pycocotools.COCO`, boxes scaled and clamped, one category, valid block separate from train, stray dropped, test game refused.
  - `fit` passes only the allowed kwargs listed in §3, resumes when `last.ckpt` exists, copies the best file, and writes the manifest.
  - `predict`: class filter 1 for stock and 0 for fine-tuned, output format and full-resolution scale readable by `eval_detections.load_dets`.
  - `evaluate`: on GT-as-detections from a synthetic tree, recall is about 1 and foot-point error about 0.
  - Optional test, skipped unless rfdetr is present: `rfdetr.config.TrainConfig` accepts the kwargs.

### WP5: Colab, README, E2E (authoring parallel; E2E acceptance last)

- **Owns:** `colab/build_src_notebooks.py`, generated notebooks, `src/README.md`, `src/configs/smoke/*.yaml`, `src/tests/{test_notebooks,test_e2e_synthetic}.py`.
- **Tests:**
  - Both notebooks build and are valid JSON.
  - Bundle listing contains only allowed suffixes and paths, with no `data/`, `outputs/`, `docs/assets/`, or video/annotation filenames.
  - All four video IDs and both folder IDs are present.
  - The install cell includes `--no-install-package python-doctr` and `--group rfdetr` only for the detector notebook.
  - Unpacking the bundle into a temp dir and importing `src.core.config` works.
  - The final cell contains `files.download` and `flush_and_unmount`.
  - Every notebook command targets a module that exists.

## 6. Acceptance and smoke

**Per package:** all listed tests pass with no GPU, no video and no download, under `uv run python -m pytest src/tests -q`. Additionally:

| WP | Also requires |
|---|---|
| WP0 | `python -m src.train --config <toy>` runs a registered toy plugin end to end |
| WP1 | Real annotation load on all 8 games returns correct team and jersey, with stray count 1 on 0110 and 0 elsewhere (local annotations only) |
| WP2 | Switching to ResNet-34 needs one config line |
| WP3 | `coverage.md` from synthetic data shows the warning when a number is uncovered |
| WP4 | No import of `rfdetr` at module level |
| WP5 | Generated notebooks under 1 MB or so, with the existing notebook unchanged |

**End-to-end, tier 1 (`test_e2e_synthetic.py`, about 60-90 s, in pytest):**
1. Three synthetic training games plus one test game, with a `FakePose` and tiny DINO.
2. Run extract, `weak.folds`, eligibility training, finalize, number training and evaluation.
3. Assert all files exist, `done.json` is present, a rerun skips everything, `metrics.json` has every required key, and manifests have hashes.

**End-to-end, tier 2 (manual on the Mac with real weights and IMG_0106):**

```bash
# classifier_mac.yaml: games.train [IMG_0106], every 30, limit_frames 300, guard_frames 0,
# one head epoch, one fine-tune epoch, batch 8, workers 0
uv run python -m src.crops.extract --config src/configs/smoke/classifier_mac.yaml --games IMG_0106
uv run python -m src.train --config src/configs/smoke/classifier_mac.yaml
uv run python -m src.evaluate.classifier --game IMG_0106 --allow-nontest \
    --number smoke=src/configs/smoke/classifier_mac.yaml --out outputs/reports/smoke

# detector_mac.yaml: few frames, 1 epoch, batch 2, device cpu, no extra packages in the shared venv
uv run --isolated --group rfdetr python -m src.train --config src/configs/smoke/detector_mac.yaml
uv run --isolated --group rfdetr python -m src.detector.predict --weights stock --game IMG_0106 --every 30 --limit 5
```

**Pass criteria:**
- About 60 crops with median near 21x33.
- Most rows `ok`, with pose confidence around your measured median of 0.80.
- A visual check of 20 crops showing torsos.
- A real `dinov2-small` training finishes.
- The detector smoke test produces `best.pth` and a parseable detections file. This is the first real execution of the RF-DETR wrapper.

## 7. Risks, decisions, and what I could not verify

### Decisions needed from you

1. **IMG_0109 in the weak-label pool.** Default is yes. It does not touch the test game. Without it, labels exist only for 6 and 9. Set `weak_pool2.yaml` if you disagree, and expect a much weaker eligibility head.
2. **docTR and GroundingDINO in `uv.lock` and `.venv`.** They are pre-existing, never imported by `src/`, and not installed on Colab for these notebooks. A clean fix means dropping `trackers[detection]` for plain `trackers` plus `rfdetr`, and moving `tools/run_detector.py` off `inference_models`. That is the parallel session's code, so I did not plan it here. Also leave them out of the submission ZIP's install instructions if you want the same guarantee there.
3. **ViTPose provenance.** See §0 item 4. If your NDA covers this, a heuristic crop from fixed box fractions is the fallback, at a real accuracy cost. It is not built.
4. **Retraining on all games for the submission.** `number_submit.yaml` and `weak_pool4.yaml` add 0108 once evaluation is finished. Weak-pool 4 also closes the #11 gap. This means the submitted model has no held-out score of its own, which is the normal trade-off, so it is opt-in.

### Decisions I made

- **No feature cache for LOGO.** Cheap folds come from fewer frames and epochs. A cache would remove augmentation, including the greyscale needed against the colour shortcut, for little saving at this data size.
- **Letterboxed crops**, not stretched, to preserve digit shape.
- **`src` is not an installed package.** It is run with `python -m` from the repo root, so `pyproject.toml` needs no build changes.
- **Two checkpoints at inference**, one each for eligibility and number. This means two backbone passes, which is simple and fast enough. A shared backbone can come later.
- **CSV over parquet**, for no new dependency.
- **Default thresholds:** `max_iou` 0.1 drops about a third of crops, and `tau_pos` 0.7 and `tau_neg` 0.3 are starting values. All live in config.
- **Scope ends at metrics on the held-out game.** Connecting these models to a tracker for submission is not in this plan.

### Risks

- **Person and bib memory in weak labels.** Numbers on the same bib recur across games, so a held-out model can read a person rather than digits. Mitigations: 9-way unmasked scoring, greyscale, facing and `fold_saw_bib` diagnostics, contact sheets, cross-colour runs. Spend five minutes on the contact sheets before trusting the labels.
- **Frozen or short-budget fold models may be weak.** At 21x33 px, few crops may reach `tau_pos`. Then eligibility has few positives, the report warns, `fill_min_per_class` keeps training running, and `number_all` remains available. Compare both in the report before choosing.
- **The fine-tuned detector is tuned to the end-line camera.** Only 6 players are labelled, so officials and waiting players become negatives. The challenge camera is oblique with those same people nearby. Held-out 0108 scores use the training camera and cannot show this. Keep stock RF-DETR as a fallback.
  - Rules allow geometric warps as augmentation, but it needs `rfdetr[augment]`, so this plan leaves it out.
  - A cheap later proxy: warp 0108 frames into the challenge view using the two official homographies, and compare both detectors there. No challenge frames needed.
- **The classifier has the same camera shift.** Training crops are simulated 720p from a different angle, and challenge H.264 at a closer or more oblique view will differ. Nothing here measures that.
- **`gdown` quota failures** on 1.3 GB files are common. The notebook checks your Drive first and stops with clear instructions.
- **`files.download` needs the tab open**, and browsers may block the second download. Everything is also in Drive.
- **Colab disconnects.** Runs resume per epoch, finished steps are skipped, and crops are restored from Drive. RF-DETR resumes from `last.ckpt`.
- **Parallel session edits.** The `uv add` changes `pyproject.toml` and `uv.lock`, which that session owns. Coordinate timing, and rerun `uv lock --check` before building notebooks, because the bundle embeds `uv.lock` for `--frozen`.

### Licences

| Package or weights | Licence | How checked |
|---|---|---|
| rfdetr 1.11.2, RF-DETR Small weights | Apache 2.0 | Package metadata and README table |
| pytorch-lightning, torchmetrics, roboflow, pyDeprecate | Apache-2.0 | PyPI |
| hotcoco, simplejpeg | MIT | PyPI |
| vernier | MIT or Apache-2.0 | PyPI |
| ultrafast-pycocotools | BSD-2-Clause | PyPI |
| faster-coco-eval, torch-hungarian | Apache-2.0 | GitHub licence API |
| dinov2-small, vitpose-base-simple | Apache-2.0 | Hugging Face API tag |
| pydantic, PyYAML, triton | Believed MIT | Not checked |
| ResNet-34 ImageNet weights | torchvision BSD code; ImageNet data terms not checked | Disclose as a public checkpoint |

Grepping `rfdetr` for mm-family and banned OCR names found nothing, and its only Ultralytics references are to a dataset file format. The dependency is not installed or imported.

### Verified by running or reading

- `VitPose` call sequence and output shapes, on the real cached model with a synthetic image.
- Tiny `Dinov2Model` forward with position-embedding interpolation.
- `resnet34` structure.
- `TrainConfig` fields, including `extra="forbid"`.
- `checkpoint_best_total.pth` and `last.ckpt` names.
- Roboflow layout, category filtering and class ID semantics, from rfdetr source.
- Universal dependency resolve.
- `torch_hungarian` has a pure-Python wheel, and others have Linux x86_64 wheels.
- `--no-install-package` exists on `uv sync`.
- HF revisions, from the API.

### Not verified

- **RF-DETR training end to end has not been executed.** The source was read but it was never run. Tier 2 smoke is the first real test.
- `facebook/dinov2-small` real weights are not in the local cache. Its revision, file list and licence came from the HF API, but loading them has not been run.
- The `metrics.csv` column name for RF-DETR's mAP, which is why the plugin must not fail on it.
- `gdown.download_folder` against the official folders.
- All A100 timings, and the number of vCPUs on the Colab A100 machine.
- Colab's ffmpeg hardware decode for this HEVC.
- That no player wore two bibs across games. The cross-colour reading assumes different people, and the paper mentions 2 reserves.

### Critical Files for Implementation

- /Users/tanmaysharma/code/tracking_challenge/tools/trackid/io.py
- /Users/tanmaysharma/code/tracking_challenge/tools/eval_detections.py
- /Users/tanmaysharma/code/tracking_challenge/tools/run_detector.py
- /Users/tanmaysharma/code/tracking_challenge/colab/build_notebook.py
- /Users/tanmaysharma/code/tracking_challenge/.gitignore

## 8. Amendment: ViTSTR recogniser and all 8 videos (2026-11-01)

The user chose to include an OCR stage so the method reads well in the paper, and chose **ViTSTR-small**, fine-tuned on the official training set. Where this conflicts with an earlier section, this section wins.

### Pipeline
```
RF-DETR player box -> ViTPose shoulders + hips -> torso crop -> DINOv2 eligibility head
-> fine-tuned ViTSTR-small -> score the team's 3 roster numbers -> vote across the tracklet
```
The DINOv2 9-class number head stays as a baseline for the paper, not as the production reader.

### Data
All 8 training videos are now in `data/Training/vidoes/` on the Mac.
- Train: IMG_0104, 0105, 0106, 0107, 0109, 0110, 0111. Test: IMG_0108 only.
- Colab Drive IDs for the videos not listed above: IMG_0104 `1e0r3kqSBbXVCwjDj4hh3_K7Oh_KKsTa7`, IMG_0105 `1yaAoPXn-aTLwmb-LasGMdKdyXff2TndK`, IMG_0110 `1yNscs0JoERtgtt43oSGlO2SJebhtXMOJ`, IMG_0111 `1g6kidSDP4mo16dLsDWMeZXKNyIrqjCsD`.
- Crops are stored at 720p and cost about 30 MB per game, so `every: 5` still fits.

### Weak-label folds replace leave-one-game-out
Use 3 fixed game groups. Each held-out group's model trains on the other groups, and every number is covered in every fold, so the #11 gap in §3 goes away. Label `-2` (uncovered) remains in the code but should not occur, and `fill` should not be needed.

| Held out | Trained on |
|---|---|
| IMG_0104, IMG_0109 | IMG_0105, 0106, 0107, 0110, 0111 |
| IMG_0105, IMG_0110 | IMG_0104, 0106, 0107, 0109, 0111 |
| IMG_0106, IMG_0107, IMG_0111 | IMG_0104, 0105, 0109, 0110 |

- `WeakCfg` gains `folds: list[list[str]]`, which replaces `pool`, and `fold_plugin: Literal["recognizer", "classifier"] = "recognizer"`.
- `fold_config(cfg, held_out: list[str])` moves the whole group to `games.test`.
- A test must assert full 9-number coverage for these folds from the rosters.
- Replace `weak_pool2/3/4.yaml` with `weak_folds3.yaml`. The eligibility head and `number_eligible` train on the 7 training games.

### ViTSTR-small model
Upstream: `roatienza/deep-text-recognition-benchmark`, Apache 2.0. Weights: `https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224_aug.pth`, sha256 `3ff5c0490dabcb99b8540e66feb61d60e57821b55dabaacecbf31920824e42aa`, 86 MB.

Verified by loading the real file under `weights_only=True`, `timm` 1.0.30:
- 152 keys, prefixed `module.vitstr.`; strip the prefix. All keys matched this model:
  `VisionTransformer(img_size=224, patch_size=16, in_chans=1, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=96)`.
- **Greyscale input** (1 channel), so colour cannot be a shortcut for this model. Pixels are normalised with `(x / 255 - 0.5) / 0.5`.
- Upstream `forward`: patch embed, prepend CLS, add `pos_embed`, `pos_drop`, blocks, `norm`, take the first `seqlen` tokens, apply `head` to each. Write this as our own small subclass of timm's `VisionTransformer`, with an attribution comment. Do not vendor upstream modules or call their `create_model` registry.
- Token order: `['[GO]', '[s]'] + list(string.printable[:-6])`, 96 tokens. `[GO]` = 0, `[s]` = 1.

**Stock result, measured on 239 unfiltered IMG_0106 crops at 720p** (fixed torso fraction of the box, stretched to 224): free read 0.4% exact, digits-only read 8.4%, roster-limited 3-way 31.4%, which is chance (33.3%). So the stock model has no usable signal on these crops, and **fine-tuning has to supply it**. Report this result in the paper as the untrained baseline.

**Licence:** the repo is Apache 2.0. The release assets carry no separate licence, and the weights were trained on synthetic text datasets per the upstream paper; disclose both in `src/README.md` and the paper. The ImageNet DeiT start is not used, because we load the STR weights.

### Tokens and head
Replace the 96-way head with a 12-way head: `[GO]`, `[s]`, digits `0`-`9`. Copy the matching rows of the pretrained `head.weight` and `head.bias` so digit knowledge carries over.

Each number is read one digit at a time. With `seqlen = 4`, target positions are `[GO] d1 d2-or-[s] [s]-or-[GO]`, so `10` becomes `[GO] 1 0 [s]` and `5` becomes `[GO] 5 [s] [GO]`. Loss is cross-entropy with `ignore_index` = 0 (`[GO]`), which matches upstream: the leading position and all padding are ignored.

### New package: `src/recognizer/` (plugin name `recognizer`)
```
tokens.py      TOKENS, GO, EOS, encode_number(str) -> list[int], candidate targets for the 9 numbers
model.py       ViTSTR(VisionTransformer), build_vitstr(cfg), load and remap pretrained weights, blocks() for freezing
transforms.py  reuse the classifier's augmentations and no-flip rule; convert to greyscale, resize to 224, 1 channel
dataset.py     CropDataset with digit-token targets
plugin.py      RecognizerPlugin(LoopPlugin)
predict.py     predict_crops-compatible output, plus free decoding
```
- **Weights:** download through `torch.hub.download_url_to_file` into a local cache, check the sha256, load with `weights_only=True`. Record `WeightSource` with URL and hash. `pretrained=False` must build a random tiny model for tests with no download.
- **Resize:** config `resize: Literal["letterbox", "stretch"] = "letterbox"`. Upstream stretches. Digit shape favours letterbox, so run letterbox by default and include one `stretch` run in the report if time allows.
- **Phases:** first the head and last 2 blocks (3 epochs), then the last 6 blocks at a lower LR with layer-wise decay (8 epochs). Batch size 128, bf16 on CUDA.
- **Config:** add `RecognizerCfg` (`url`, `sha256`, `input_size: 224`, `seqlen: 4`, `resize`, `pretrained`, `config_overrides`). `Config` gains `recognizer: RecognizerCfg | None`.
- **Roster-constrained scoring:** one forward pass per crop. For each of the 9 numbers, sum the log-probabilities at its target positions through `[s]`, then softmax across the 9. Output shape is `(N, 9)` in `NUMBERS` order, so all `evaluate/` and `weak/` code works unchanged. Roster masking then applies as usual.
- **Free decoding:** take the argmax at each position after the first, cut at `[s]`, for the paper's unconstrained read rate.
- **Configs:** `src/configs/recognizer/{base,vitstr_all,vitstr_eligible,vitstr_xcolor_no_yellow,vitstr_xcolor_no_green}.yaml`.

### Reporting additions on IMG_0108
- Stock ViTSTR vs fine-tuned ViTSTR vs DINOv2 9-class vs ResNet-34, on the same crops.
- ViTSTR with and without the roster limit, and with and without eligibility filtering.
- Free-decoding read rate, plus how often it outputs a number not on the roster.
- Cross-colour runs for both ViTSTR and the 9-class head. ViTSTR is greyscale, so these mainly test person memory.

### Synthetic digits are left out of this build
The user asked to fine-tune on the given dataset. Digit-level labels do not remove person memory: `3` only ever appears on the #3 bib, and only `1` repeats across numbers. If the ViTSTR cross-colour results are weak, add synthetic digits on bibs as a follow-up and disclose them.

### Work-package changes
- **WP0:** partly written by a stopped agent and not reviewed. Review it, finish it, and run its tests. The `uv add` for `pydantic`, `pyyaml` and the `rfdetr` group is already applied, so do not repeat it. `src/tracker/` was not created by this plan: leave it alone. WP0 also adds `RecognizerCfg`, the `WeakCfg` changes, and `recognizer` in `PLUGINS`. `timm` is already in the environment through `trackers[detection]`.
- **WP6 (new, parallel):** owns `src/recognizer/*.py`, `src/configs/recognizer/*.yaml`, `src/tests/test_recognizer.py`. No download in unit tests: use a tiny random model. Tests cover digit targets for all 9 numbers including one-digit padding, head remapping of the digit rows, scoring shape and row order, roster-limited argmax, freeze groups, no flip, greyscale input, sha256 mismatch raises, and a plugin run above chance on the synthetic store.
- **WP3:** owns `weak_folds3.yaml` and uses the fold change above.
- **WP5:** notebook adds all 8 videos and the recogniser runs. **The classifier notebook estimate grows to about 3-4 h, unmeasured.** The fold step is the largest unknown, so report time per epoch early. Drop the ResNet-34 and classifier cross-colour runs first if time runs short.
- **Real-weights smoke (Mac, manual):** load the ViTSTR weights and run one short fine-tune on real IMG_0106 crops.
- **Banned name:** the repo must not mention the model the user ruled out, anywhere, in any letter case. Also keep MMDetection and its family out of the code, as in the hard requirements.

### Data check: IMG_0104 frame count
IMG_0104 has 11,400 video frames, but its labels stop at frame 11,376. The other 7 videos match exactly. The hard check in `crops/extract.py` would stop on this game, so allow labels to end early and log the gap. It is not known whether the extra 24 frames are at the end or whether the labels are shifted, so check alignment by overlaying boxes at the start, middle and end before training on this game. If they are shifted, leave IMG_0104 out and recheck fold coverage.

**Checked 2026-11-01:** overlays at labelled frames 1 and 11,376 show boxes tight on the players, so a 24-frame shift is unlikely. IMG_0104 stays in training, and the extra frames are treated as an unlabelled tail.

## WP0 deviations from plan

Signatures in §2 and §3 are kept. Additions and clarifications that other packages rely on:

- `core/paths.py`: `ROOT`, `OUTPUTS`, `outputs_dir() -> Path` (honours env `SRC_OUTPUTS`; `tests/conftest.py` sets it per test, so always use `outputs_dir()`, never `OUTPUTS`), `resolve_path(p) -> Path` (repo-relative to absolute), `add_tools_to_path()`.
- `core/config.py` (§8 items): `RecognizerCfg(url, sha256, input_size=224, seqlen=4, resize: Literal["letterbox","stretch"]="letterbox", pretrained=True, config_overrides={})` with the ViTSTR-small URL and sha256 as defaults; `Config.recognizer: RecognizerCfg | None`; `WeakCfg(name, folds: list[list[str]], fold_plugin: Literal["recognizer","classifier"]="recognizer", tau_pos, tau_neg, head_threshold, fill_min_per_class, fold_subsample_every, fold_phases)`. `pool` is gone. `folds` needs at least 2 non-empty groups, valid game ids, no game in two groups.
- `Config` validator: `plugin == "recognizer"` needs `recognizer` and `train` (same pattern as classifier). `train.phases` drives the generic loop for it too.
- `PLUGINS["recognizer"] = "src.recognizer.plugin:RecognizerPlugin"`; `src/recognizer/__init__.py` is empty. `get_plugin` also accepts any `"module:Class"` string.
- `LoopPlugin` has two optional hooks: `set_train_mode(model, phase)` (called after `model.train()` each epoch, e.g. to keep frozen BatchNorm in eval) and `checkpoint_meta() -> dict` (overrides `head`, `classes`, `backbone` in best.pt). Defaults: `classes` is `NUMBERS` for head `number`, `["ineligible","eligible"]` for `eligibility`; for a plugin without `cfg.model` they are empty, so the recognizer plugin should return its own `head`, `classes`, `backbone` from `checkpoint_meta()`.
- `fit_loop(plugin, run) -> FitResult` in `core/loop.py`; `FitResult.extra` has `epochs`, `best_epoch`, `stopped_early`, `device`. Early stop ends the whole run, not just the phase. `last.pt` also stores `stopped`, `best_epoch`; load it with `load_checkpoint(path, full=True)`.
- `core/run.py`: `Run.manifest() -> dict`, `Run.load_result() -> FitResult`, `file_sha256(path)`, `code_version()`. `open_run` on an existing dir records `resumed` and the new argv in the manifest.
- `dataio/schema.py`: `CROP_COLUMNS`, `crop_id(video, frame, track_id)`, `validate_manifest(df)`. String columns come back as pandas 3 `str` dtype, not `object`; never test `dtype == object`.
- `dataio/store.py` adds `CropWriter` as a context manager and `mark_done(video_dir, **meta)` (writes `done.json`; call it last). `save_manifest` validates first. `CropStore` pickles without its memmaps.
- `dataio/select.py`: the split is computed on the unfiltered frame range before filters, then `subsample_every` runs on dense frame rank, then guard rows are dropped.
- `tests/fixtures.py` also exports `frame_index(rgb)`, `draw_digit`, `ToyPlugin` (usable as plugin `"src.tests.fixtures:ToyPlugin"`; class attribute `crash_at` simulates a crash in `evaluate`), `toy_config_dict(run_name, phases, patience)` and `toy_config(...)`. `make_video` writes a lossless FFV1 `.mkv` (suffix forced unless `.mov`/`.mkv`) and returns the real path; frame i is the flat colour `(i%256, i//256, 77)`.
- `tests/test_compliance.py` scans every repo text file (`git ls-files -co --exclude-standard`) against a stored SHA-256 and imports whichever entry modules exist in a subprocess, asserting no `FORBIDDEN_MODULES` or `rfdetr` in `sys.modules`. A new module is checked once its name is in `ENTRY_MODULES` there.
