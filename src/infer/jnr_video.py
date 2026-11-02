"""Jersey-number evidence for every detection of a video: torso crop, readability and number scores. Inference only.

  python -m src.infer.jnr_video --video PATH --dets DETS.txt --eligibility elig/best.pt --vitstr vitstr/best.pt
                                [--every 2] [--min-conf 0.3] [--limit N]

Same crop recipe as training (ViTPose torso box on the 720p frame, src/crops). For each detection with conf >= --min-conf
on every --every-th frame, one row in outputs/jnr/<dets stem>__jnr.csv.gz:
  det_row        0-based line of the detection in DETS.txt, to join back exactly
  frame,x,y,w,h,det_conf   copied from DETS.txt (full-resolution pixels)
  max_iou        overlap with the most-overlapping other kept detection on that frame
  pose_conf      min confidence of the 4 torso keypoints; facing: front/back/side/unknown
  status         ok | low_pose | degenerate (degenerate rows have no crop and NaN scores)
  crop_w,crop_h  crop size at 720p
  elig           DINOv2 eligibility head: P(number readable)
  p_1 .. p_12    ViTSTR: probability of each of the 9 numbers (log-prob sum per number, softmax across the 9).
                 Apply the team's roster yourself: renormalise over its 3 numbers.
  read           ViTSTR free read (no roster limit)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.core.config import Config, CropCfg
from src.core.constants import NUMBERS
from src.core.loop import pick_device
from src.core.paths import outputs_dir
from src.crops.geometry import MIN_KP_CONF, TORSO, facing, max_iou_per_box, torso_box
from src.dataio.video import frame_count, iter_frames, probe

COLUMNS = (["det_row", "frame", "x", "y", "w", "h", "det_conf", "max_iou", "pose_conf", "facing", "status", "crop_w",
            "crop_h", "elig"] + [f"p_{n}" for n in NUMBERS] + ["read"])
HEIGHT = 720


class Readers:
    """Eligibility head and ViTSTR, batched over crops."""

    def __init__(self, elig_pt: Path, vitstr_pt: Path, device: str) -> None:
        from src.classifier.model import build_model
        from src.classifier.transforms import build_transform
        from src.core.checkpoint import load_checkpoint
        from src.recognizer.predict import load_model
        from src.recognizer.transforms import eval_transform
        self.dev = pick_device(device)
        ck = load_checkpoint(Path(elig_pt))
        if ck.get("head") != "eligibility":
            raise ValueError(f"{elig_pt} is a {ck.get('head')!r} head, not the eligibility head")
        ecfg = Config.model_validate(ck["config"])
        self.elig = build_model(ecfg, pretrained=False)
        self.elig.load_state_dict(ck["state_dict"])
        self.elig.to(self.dev).eval()
        self.elig_tf = build_transform(ecfg.model.input_size, None, False)
        self.vit, rec = load_model(Path(vitstr_pt), None, self.dev)
        self.vit_tf = eval_transform(rec)

    @torch.no_grad()
    def __call__(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        from src.recognizer.scoring import score_numbers
        from src.recognizer.tokens import decode_free
        if not crops:
            return np.zeros(0, np.float32), np.zeros((0, len(NUMBERS)), np.float32), []
        x = torch.stack([self.elig_tf(c) for c in crops]).to(self.dev)
        elig = torch.softmax(self.elig(x, "eligibility").float(), 1)[:, 1].cpu().numpy()
        x = torch.from_numpy(np.stack([self.vit_tf(c, None) for c in crops])).to(self.dev)
        logits = self.vit(x).float()
        probs = score_numbers(logits).softmax(-1).cpu().numpy()
        reads = decode_free(logits.argmax(-1).cpu().numpy())
        return elig, probs, reads


def load_dets(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, header=None, usecols=[0, 2, 3, 4, 5, 6], names=["frame", "x", "y", "w", "h", "det_conf"])
    return d.reset_index(names="det_row")


def output_path(dets: Path) -> Path:
    return outputs_dir() / "jnr" / f"{Path(dets).stem}__jnr.csv.gz"


def jnr_video(video: Path, dets: Path, readers, pose, every: int = 2, min_conf: float = 0.3, limit: int | None = None,
              crops: CropCfg = CropCfg(), batch: int = 256) -> Path:
    video, dets = Path(video), Path(dets)
    d = load_dets(dets)
    d = d[d.det_conf >= min_conf]
    by_frame = {int(f): g for f, g in d.groupby("frame")}
    src_w, _ = probe(video)
    rows: list[dict] = []
    pending: list[tuple[int, np.ndarray]] = []  # (index into rows, crop)

    def flush() -> None:
        if not pending:
            return
        elig, probs, reads = readers([c for _, c in pending])
        for (i, _), e, p, r in zip(pending, elig, probs, reads):
            rows[i].update(elig=float(e), read=r, **{f"p_{n}": float(v) for n, v in zip(NUMBERS, p)})
        pending.clear()

    t0, n_frames, last = time.time(), 0, 0
    for frame, img in iter_frames(video, 1, HEIGHT, limit):
        n_frames, last = n_frames + 1, frame
        if (frame - 1) % every or (g := by_frame.get(frame)) is None:
            continue
        s = src_w / img.shape[1]
        b = g[["x", "y", "w", "h"]].to_numpy(np.float64) / s
        ious = max_iou_per_box(b)
        ok = (b[:, 2] >= 1) & (b[:, 3] >= 1)
        kps = np.zeros((len(g), 17, 3), np.float32)
        if ok.any():
            kps[ok] = pose(img, b[ok])
        for i, r in enumerate(g.itertuples()):
            tb = torso_box(kps[i], crops.pad_frac, crops.min_aspect, crops.min_side_px, img.shape[1], img.shape[0]) \
                if ok[i] else None
            conf = float(kps[i][list(TORSO), 2].min()) if ok[i] else 0.0
            row = {"det_row": r.det_row, "frame": frame, "x": r.x, "y": r.y, "w": r.w, "h": r.h, "det_conf": r.det_conf,
                   "max_iou": float(ious[i]), "pose_conf": conf, "facing": facing(kps[i]) if ok[i] else "unknown",
                   "status": "degenerate", "crop_w": 0, "crop_h": 0, "elig": np.nan, "read": "",
                   **{f"p_{n}": np.nan for n in NUMBERS}}
            if tb is not None:
                x0, y0, x1, y1 = tb
                row.update(status="ok" if conf >= MIN_KP_CONF else "low_pose", crop_w=x1 - x0, crop_h=y1 - y0)
                pending.append((len(rows), np.ascontiguousarray(img[y0:y1, x0:x1])))
            rows.append(row)
        if len(pending) >= batch:
            flush()
        if n_frames % 1000 == 0:
            print(f"{video.stem}: frame {frame}, {len(rows)} detections, {time.time() - t0:.0f}s", flush=True)
    flush()
    # Labels and detections are numbered by decode order, so only a frame count above the packet count is fatal.
    if not limit and n_frames > frame_count(video):
        sys.exit(f"decoded {n_frames} frames but the video has only {frame_count(video)} packets")
    if not limit and d.frame.max() > last:
        sys.exit(f"detections reach frame {int(d.frame.max())} but the video decoded only {last} frames")
    out = output_path(dets)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    pd.DataFrame(rows, columns=COLUMNS).to_csv(tmp, index=False, compression="gzip", float_format="%.5g")
    tmp.rename(out)  # only a finished run appears under the final name
    print(f"{video.stem}: {len(rows)} detections on {n_frames} frames (every {every}) -> {out}, {time.time() - t0:.0f}s")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--dets", type=Path, required=True)
    ap.add_argument("--eligibility", type=Path, required=True)
    ap.add_argument("--vitstr", type=Path, required=True)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--limit", type=int, help="stop after N decoded frames (smoke test)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    from src.crops.pose import ViTPose
    dev = str(pick_device(a.device))
    crops = CropCfg()
    jnr_video(a.video, a.dets, Readers(a.eligibility, a.vitstr, dev), ViTPose(crops.pose_model, crops.pose_revision, dev),
              a.every, a.min_conf, a.limit, crops)


if __name__ == "__main__":
    main()
