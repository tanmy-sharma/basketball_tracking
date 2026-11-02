"""Fine-tuned RF-DETR Small on every frame of any video, including the challenge videos. Inference only.

  python -m src.infer.detect_video --video PATH --weights best.pth [--previews 6] [--limit N]

Output matches tools/run_detector.py, so the trackers read it unchanged: MOT det format in full-resolution pixels,
1-based frames, frames downscaled to 720p first, conf floor 0.05, in outputs/detections/<stem>__<model>__720p.txt,
where <model> is rfdetr-small-ft-<first 8 hex of the weights' sha256>. Nothing here trains or writes training data.
--previews also saves that many evenly spaced frames with the boxes at conf >= 0.5 drawn, for a visual check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.core.paths import outputs_dir
from src.core.run import file_sha256
from src.dataio.video import frame_count, iter_frames, probe
from src.detector.predict import FINETUNED_PLAYER_ID, HEIGHT, _batches, load_model

CONF_FLOOR = 0.05  # same as tools/run_detector.py
PREVIEW_CONF = 0.5
BATCH = 8


def model_name(weights: str) -> str:
    return f"rfdetr-small-ft-{file_sha256(weights)[:8]}"


def output_path(video: Path, weights: str) -> Path:
    return outputs_dir() / "detections" / f"{video.stem}__{model_name(weights)}__{HEIGHT}p.txt"


def _draw(img: np.ndarray, boxes: list[tuple[float, float, float, float, float]], path: Path) -> None:
    import cv2
    out = np.ascontiguousarray(img[..., ::-1])  # RGB -> BGR for cv2
    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(out, f"{c:.2f}", (int(x1), max(int(y1) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


def detect_video(video: Path, weights: str, previews: int = 0, limit: int | None = None) -> Path:
    video = Path(video)
    src_w, _ = probe(video)
    n_packets = frame_count(video)
    model = load_model(weights)
    out = output_path(video, weights)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    total = min(n_packets, limit) if limit else n_packets
    want = {int(round(x)) for x in np.linspace(1, total, previews)} if previews else set()
    n = 0
    with open(tmp, "w") as f:
        for batch in _batches(iter_frames(video, 1, HEIGHT, limit), BATCH):
            res = model.predict([img for _, img in batch], threshold=CONF_FLOOR, include_source_image=False)
            res = res if isinstance(res, list) else [res]
            for (frame, img), det in zip(batch, res):
                n += 1
                s = src_w / img.shape[1]
                kept = []
                if det.class_id is not None and len(det):
                    m = det.class_id == FINETUNED_PLAYER_ID
                    for (x1, y1, x2, y2), c in zip(det.xyxy[m], det.confidence[m]):
                        f.write(f"{frame},-1,{x1 * s:.2f},{y1 * s:.2f},{(x2 - x1) * s:.2f},{(y2 - y1) * s:.2f},"
                                f"{c:.4f},-1,-1,-1\n")
                        kept.append((x1, y1, x2, y2, float(c)))
                if frame in want:
                    _draw(img, [b for b in kept if b[4] >= PREVIEW_CONF],
                          outputs_dir() / "previews" / f"{video.stem}__{model_name(weights)}__f{frame:06d}.jpg")
            if n % 2000 < BATCH:
                print(f"{video.stem}: {n} frames", flush=True)
    # As in tools/run_detector.py: more frames than packets would shift numbering; fewer can only be missing at the end.
    if not limit and n > n_packets:
        tmp.unlink()
        sys.exit(f"decoded {n} frames but the video has only {n_packets} packets; frame numbers would not match")
    if not limit and n < n_packets:
        print(f"WARNING: decoded {n} frames, {n_packets - n} fewer than the packet count. Kept: numbering is unaffected")
    tmp.rename(out)  # only a finished run appears under the final name
    print(f"{video.stem}: {n} frames -> {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--weights", required=True, help="fine-tuned best.pth")
    ap.add_argument("--previews", type=int, default=0)
    ap.add_argument("--limit", type=int, help="stop after N frames (smoke test, skips the frame-count check)")
    a = ap.parse_args(argv)
    detect_video(a.video, a.weights, a.previews, a.limit)


if __name__ == "__main__":
    main()
