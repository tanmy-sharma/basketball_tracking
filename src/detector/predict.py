"""MOT-format detections from stock or fine-tuned RF-DETR Small, in full-resolution pixels.

  python -m src.detector.predict --weights stock|PATH --game IMG_0108 [--every 5] [--limit N]
Output: frame,-1,x,y,w,h,conf,-1,-1,-1 (1-based frames), readable by tools/eval_detections.py::load_dets.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.core.config import DetectorCfg, load_config
from src.core.paths import outputs_dir, resolve_path
from src.core.run import file_sha256
from src.dataio.video import frame_count, iter_frames, probe, video_path

STOCK_PERSON_ID = 1  # stock weights report COCO category ids
FINETUNED_PLAYER_ID = 0  # fine-tuned models report 0-based indices into class_names
HEIGHT = 720
DEFAULT_CONFIG = "src/configs/detector/rfdetr_small.yaml"


def load_model(weights: str):
    from rfdetr import RFDETRSmall  # lazy: optional dependency
    return RFDETRSmall() if weights == "stock" else RFDETRSmall.from_checkpoint(weights)


def output_path(game: str, weights: str, every: int) -> Path:
    tag = "stock" if weights == "stock" else f"ft-{file_sha256(weights)[:8]}"
    suffix = f"__{HEIGHT}p" + (f"__every{every}" if every > 1 else "")
    return outputs_dir() / "detections" / f"{game}__rfdetr-small-{tag}{suffix}.txt"


def _batches(it: Iterator[tuple[int, np.ndarray]], n: int) -> Iterator[list[tuple[int, np.ndarray]]]:
    buf: list[tuple[int, np.ndarray]] = []
    for item in it:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def predict_game(game: str, weights: str, every: int, limit: int | None, cfg: DetectorCfg) -> Path:
    path = video_path(game)
    src_w, src_h = probe(path)
    keep_id = STOCK_PERSON_ID if weights == "stock" else FINETUNED_PLAYER_ID
    model = load_model(weights)
    out = output_path(game, weights, every)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    n = 0
    with open(tmp, "w") as f:
        for batch in _batches(iter_frames(path, every, HEIGHT, limit), cfg.predict_batch):
            res = model.predict([img for _, img in batch], threshold=cfg.conf_floor, include_source_image=False)
            res = res if isinstance(res, list) else [res]
            for (frame, img), det in zip(batch, res):
                sx, sy = src_w / img.shape[1], src_h / img.shape[0]
                n += 1
                if det.class_id is None or not len(det):
                    continue
                m = det.class_id == keep_id
                for (x1, y1, x2, y2), c in zip(det.xyxy[m], det.confidence[m]):
                    f.write(f"{frame},-1,{x1 * sx:.2f},{y1 * sy:.2f},{(x2 - x1) * sx:.2f},{(y2 - y1) * sy:.2f},"
                            f"{c:.4f},-1,-1,-1\n")
    expected = -(-frame_count(path) // every)
    if limit is None and n != expected:
        tmp.unlink()
        sys.exit(f"decoded {n} frames but the video should give {expected}; frame numbers would not match the GT")
    tmp.rename(out)  # only a finished run appears under the final name
    print(f"{game}: {n} frames -> {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="'stock' or a path to a fine-tuned best.pth")
    ap.add_argument("--game", required=True)
    ap.add_argument("--every", type=int)
    ap.add_argument("--limit", type=int, help="stop after N processed frames (smoke test, skips the frame-count check)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    a = ap.parse_args(argv)
    cfg = load_config(resolve_path(a.config)).detector
    predict_game(a.game, a.weights, a.every or cfg.predict_every, a.limit, cfg)


if __name__ == "__main__":
    main()
