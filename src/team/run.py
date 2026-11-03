"""Team colours for one video: calibrate on random frames, label every box, check against GT, draw a video.

  python -m src.team.run IMG_0106 --boxes outputs/.../IMG_0106__rfdetr-large__720p.txt --video
  python -m src.team.run IMG_0108 --boxes gt --every 5 --seeds 20

Writes to outputs/team/: <video>__<boxes>.calibration.json, .background.png, .metrics.json and .mp4, and to outputs/labels/
the label file described in docs/13. GT is used only for scoring, and only on training videos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.core.constants import BIB_OF_ROSTER, CHALLENGE_VIDEOS
from src.core.paths import add_tools_to_path, outputs_dir
from src.dataio.video import frame_count, scaled_size
from src.team.colour import BIB_DRAW, BIB_REFERENCE, TeamColours, calibrate, crop_colour

add_tools_to_path()
from trackid import io  # noqa: E402

HEIGHT = 720
ALL_BIBS = tuple(BIB_REFERENCE)
OVERLAP = 0.2
PART_OF_PLAYER_IOU = 0.1  # a box this close to a GT player, but under the match threshold, isn't a non-player
CALIBRATION_FRAMES = 10
CALIBRATION_MIN_CONF = 0.5
LABEL_MIN_CONF = 0.1
COURT_MARGIN = 100.0  # cm


def _folder(video: str) -> Path:
    return io.FINAL if video in CHALLENGE_VIDEOS else io.TRAIN


def video_file(video: str) -> Path:
    folder = io.FINAL if video in CHALLENGE_VIDEOS else io.TRAIN / "vidoes"
    return next(p for p in sorted(folder.iterdir()) if p.stem.lower() == video.lower() and p.suffix.lower() in (".mov", ".mp4"))


def bibs_of(video: str) -> dict[str, str]:
    """bib -> team, from the roster alone."""
    roster = io.load_roster(_folder(video) / "games_roster.csv")[video]
    return {BIB_OF_ROSTER[frozenset(numbers)]: team for team, numbers in sorted(roster.teams.items())}


def decode(path: Path, frames: Iterable[int], height: int = HEIGHT) -> Iterator[tuple[int, np.ndarray]]:
    """(1-based frame, BGR image) for each requested frame, picked by index so numbering matches the annotations."""
    wanted = sorted(set(frames))
    width, height, _ = scaled_size(path, height)
    step = np.diff(wanted)
    if len(wanted) > 200 and len(set(step.tolist())) == 1:
        select = f"not(mod(n-{wanted[0] - 1},{int(step[0])}))*gte(n,{wanted[0] - 1})"
    elif len(wanted) > 200:
        selected = np.zeros(wanted[-1] + 1, bool)
        selected[wanted] = True
        select = None
    else:
        select = "+".join(f"eq(n,{f - 1})" for f in wanted)
    filters = f"select='{select}'," if select else ""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-hwaccel", "auto", "-i", str(path), "-vf",
                          f"{filters}scale={width}:{height}:flags=area", "-fps_mode", "passthrough", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size = width * height * 3
    try:
        if select:
            for f in wanted:
                buf = p.stdout.read(size)
                if len(buf) != size:
                    raise RuntimeError(f"{path.name}: video ended before frame {f}")
                yield f, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
        else:
            f = 0
            while f < wanted[-1]:
                buf = p.stdout.read(size)
                if len(buf) != size:
                    raise RuntimeError(f"{path.name}: video ended before frame {wanted[-1]}")
                f += 1
                if selected[f]:
                    yield f, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        p.kill()
        p.wait()


def load_boxes(video: str, source: str) -> pd.DataFrame:
    """frame, x, y, w, h, conf, det_row, source pixels. Detection boxes are limited to confident ones on or near the court."""
    if source == "gt":
        if video in CHALLENGE_VIDEOS:
            raise ValueError("GT is never used on challenge videos")
        b = io.load_bboxes(video).assign(conf=1.0).reset_index(drop=True)
        return b.assign(det_row=b.index)
    d = pd.read_csv(source, header=None, usecols=[0, 2, 3, 4, 5, 6], names=["frame", "x", "y", "w", "h", "conf"])
    d["det_row"] = d.index
    H, _ = io.load_homographies(_folder(video) / "court_registration.csv")
    feet = io.apply_h(H, np.c_[d.x + d.w / 2, d.y + d.h])
    near = (feet[:, 0] >= -COURT_MARGIN) & (feet[:, 0] <= io.COURT_W + COURT_MARGIN) \
        & (feet[:, 1] >= -COURT_MARGIN) & (feet[:, 1] <= io.COURT_H + COURT_MARGIN)
    return d[near & (d.conf >= LABEL_MIN_CONF)].reset_index(drop=True)


def pair_iou(box: np.ndarray) -> np.ndarray:
    """IoU between the xywh boxes of one frame, with 0 on the diagonal."""
    a1, a2 = box[:, :2], box[:, :2] + box[:, 2:]
    inter = (np.minimum(a2[:, None], a2) - np.maximum(a1[:, None], a1)).clip(0).prod(2)
    area = box[:, 2:].prod(1)
    iou = inter / (area[:, None] + area - inter + 1e-9)
    np.fill_diagonal(iou, 0)
    return iou


def background_image(path: Path, frames: Iterable[int]) -> np.ndarray:
    """Median of frames from a fixed camera: players move, so what remains is the empty court."""
    return np.median(np.stack([img for _, img in decode(path, frames)]), axis=0).astype(np.uint8)


def add_colours(boxes: pd.DataFrame, path: Path, frames: Iterable[int], background: np.ndarray | None = None) -> pd.DataFrame:
    """Rows for the requested frames, with status, a, b and overlap."""
    frames = set(frames) & set(boxes.frame)
    table = boxes[boxes.frame.isin(frames)].sort_values(["frame", "det_row"]).reset_index(drop=True)
    _, _, scale = scaled_size(path, HEIGHT)
    groups = table.groupby("frame").indices
    status, ab, over = np.full(len(table), "", object), np.full((len(table), 2), np.nan), np.zeros(len(table))
    xywh = table[["x", "y", "w", "h"]].to_numpy()
    for f, img in decode(path, frames):
        rows = groups[f]
        over[rows] = pair_iou(xywh[rows]).max(1)
        for i in rows:
            x, y, w, h = xywh[i] / scale
            cut = max(int(y), 0), max(int(y + h), 0), max(int(x), 0), max(int(x + w), 0)
            crop = img[cut[0]:cut[1], cut[2]:cut[3]]
            back = None if background is None else background[cut[0]:cut[1], cut[2]:cut[3]]
            status[i], value = crop_colour(crop, True, back)
            if value is not None:
                ab[i] = value
    return table.assign(status=status, a=ab[:, 0], b=ab[:, 1], overlap=over)


def calibration_frames(n_frames: int, seed: int, count: int = CALIBRATION_FRAMES) -> list[int]:
    """Random frames, leaving out the first and last 5% of the video."""
    low, high = int(n_frames * .05) + 1, int(n_frames * .95)
    return sorted(int(f) for f in np.random.default_rng(seed).choice(np.arange(low, high + 1), count, replace=False))


def fit(video: str, table: pd.DataFrame, frames: list[int]) -> TeamColours:
    """Calibrate on clear, confident, coloured boxes from the given frames."""
    use = table[table.frame.isin(frames) & (table.status == "ok") & (table.overlap < OVERLAP)
                & (table.conf >= CALIBRATION_MIN_CONF)]
    return calibrate(video, use[["a", "b"]].to_numpy(), bibs_of(video), frames)


def classify_table(table: pd.DataFrame, tc: TeamColours) -> pd.DataFrame:
    """Adds p_<bib>, p_none, colour and team. No-opinion columns stay NaN."""
    n = len(table)
    probs = {f"p_{b}": np.full(n, np.nan) for b in (*ALL_BIBS, "none")}
    colour, team = np.full(n, "unsure", object), np.full(n, "", object)
    for i, (status, a, b) in enumerate(zip(table.status, table.a, table.b)):
        if status == "no_colour":
            colour[i] = "none"
            probs["p_none"][i] = 1.0
            for bib in tc.centroids:
                probs[f"p_{bib}"][i] = 0.0
        elif status == "ok":
            r = tc.classify((a, b))
            colour[i], team[i] = r.colour, r.team or ""
            for key, value in r.probs.items():
                probs[f"p_{key}"][i] = value
    return table.assign(colour=colour, team=team, **probs)


def write_labels(table: pd.DataFrame, source: str, boxes_name: str, video: str) -> Path:
    out = outputs_dir() / "labels" / f"{video}__{boxes_name}.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["frame", "det_row", "x", "y", "w", "h", *[f"p_{b}" for b in (*ALL_BIBS, "none")], "status", "colour", "team"]
    table.assign(pose_min_conf=np.nan)[[*cols, "pose_min_conf"]].to_csv(out, index=False)
    if source != "gt":
        out.with_name(out.name + ".sha256").write_text(hashlib.sha256(Path(source).read_bytes()).hexdigest() + "\n")
    return out


def truth(video: str, table: pd.DataFrame, source: str) -> pd.DataFrame:
    """Adds true_team and player ('' and -1 for a box matching no player), and gt_iou (best IoU with any GT player)."""
    add_tools_to_path()
    from eval_detections import iou_matrix, match
    ids = io.track_identities(video)
    gt = io.load_bboxes(video)
    if source == "gt":
        tracks = gt.track_id.to_numpy()[table.det_row.to_numpy()]
        true = np.array([ids[int(t)][0] if int(t) in ids else "" for t in tracks], object)
        return table.assign(true_team=true, player=tracks, gt_iou=1.0)
    true, player = np.full(len(table), "", object), np.full(len(table), -1)
    positions = table.groupby("frame").indices
    near = np.zeros(len(table))
    xywh = table[["x", "y", "w", "h"]].to_numpy()
    known = gt[gt.track_id.isin(ids)]
    for f, g in known[known.frame.isin(set(table.frame))].groupby("frame"):
        rows = positions[f]
        players = g[["x", "y", "w", "h"]].to_numpy()
        idx, _ = match(players, xywh[rows])
        near[rows] = iou_matrix(xywh[rows], players).max(1)
        for track, k in zip(g.track_id, idx):
            if k >= 0:
                true[rows[k]], player[rows[k]] = ids[int(track)][0], track
    return table.assign(true_team=true, player=player, gt_iou=near)


def add_cross_team(table: pd.DataFrame) -> np.ndarray:
    """True where the most-overlapping box in the frame belongs to the other team."""
    cross = np.zeros(len(table), bool)
    xywh = table[["x", "y", "w", "h"]].to_numpy()
    true = table.true_team.to_numpy()
    for rows in table.groupby("frame").indices.values():
        if len(rows) < 2:
            continue
        iou = pair_iou(xywh[rows])
        best = iou.argmax(1)
        other = true[rows[best]]
        cross[rows] = (iou.max(1) >= OVERLAP) & (other != "") & (other != true[rows])
    return cross


def score(table: pd.DataFrame) -> dict:
    """Precision of team calls against GT. A correct call is the true team; unsure and none are counted separately."""
    players = table[table.true_team != ""]
    called = players[players.team != ""]
    right = called.team == called.true_team
    out = {"boxes": len(table), "player_boxes": len(players),
           "precision": float(right.mean()) if len(called) else None,
           "coverage": len(called) / len(players) if len(players) else None,
           "player_unsure": float((players.colour == "unsure").mean()) if len(players) else None,
           "player_none": float((players.colour == "none").mean()) if len(players) else None}
    for team in ("A", "B"):
        mine = called[called.team == team]
        out[f"precision_{team}"] = float((mine.team == mine.true_team).mean()) if len(mine) else None
    cross = add_cross_team(table)
    overlapped = table.overlap.to_numpy() >= OVERLAP
    kinds = {"clear": ~overlapped, "same_team_overlap": overlapped & ~cross, "cross_team_overlap": cross}
    for name, mask in kinds.items():
        c = table[mask & (table.true_team != "") & (table.team != "")]
        total = table[mask & (table.true_team != "")]
        out[f"{name}_precision"] = float((c.team == c.true_team).mean()) if len(c) else None
        out[f"{name}_coverage"] = len(c) / len(total) if len(total) else None
    if len(called):
        fifth = pd.cut(called.frame, 5, labels=False)
        out["precision_by_fifth"] = [float(v) for v in right.groupby(fifth).mean()]
    # Whole-player vote: the most common team call for each player
    if len(called):
        votes = called.groupby("player").team.agg(lambda s: s.value_counts().index[0])
        truth_of = called.groupby("player").true_team.first()
        out["player_vote_correct"] = int((votes == truth_of).sum())
        out["players"] = len(votes)
    # Boxes matching no player, split into real non-players and loose boxes around a player. Only the first are false calls
    others = table[(table.true_team == "") & (table.gt_iou < PART_OF_PLAYER_IOU)]
    out["non_player_boxes"] = len(others)
    if len(others):
        out["non_player_rejected"] = float((others.team == "").mean())
    out["team_calls_by_min_conf"] = {}
    for conf in (.1, .3, .5, .7):
        kept = table[table.conf >= conf]
        calls = kept[kept.team != ""]
        if len(calls):
            matched = calls.true_team != ""
            out["team_calls_by_min_conf"][str(conf)] = {
                "calls": len(calls), "right_team": float((matched & (calls.team == calls.true_team)).mean()),
                "wrong_team": float((matched & (calls.team != calls.true_team)).mean()),
                "part_of_player": float((~matched & (calls.gt_iou >= PART_OF_PLAYER_IOU)).mean()),
                "non_player": float((~matched & (calls.gt_iou < PART_OF_PLAYER_IOU)).mean()),
                "players_found": float((kept.true_team != "").sum() and ((kept.true_team != "") & (kept.team != "")).sum() / (table.true_team != "").sum())}
    return out


def draw_video(path: Path, table: pd.DataFrame, tc: TeamColours, out: Path, n: int) -> None:
    """Frames 1 to n. Only team A and team B boxes are drawn, in the bib's colour."""
    width, height, scale = scaled_size(path, HEIGHT)
    called = table[table.team != ""]
    by_frame = {f: g for f, g in called.groupby("frame")}
    names = {team: bib for bib, team in tc.teams.items()}
    legend = f"Team A: {names['A']}   Team B: {names['B']}"
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                            "-r", "30000/1001", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                            "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for f, img in decode(path, range(1, n + 1)):
        img = img.copy()
        g = by_frame.get(f)
        if g is not None:
            for x, y, w, h, team, bib in zip(g.x / scale, g.y / scale, g.w / scale, g.h / scale, g.team, g.colour):
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(img, (x, y), (x + w, y + h), BIB_DRAW[bib], 2)
                cv2.putText(img, team, (x, max(y - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, team, (x, max(y - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, .7, BIB_DRAW[bib], 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(img, f"{legend}   frame {f}", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError(f"ffmpeg could not write {out}")


def seed_spread(video: str, table: pd.DataFrame, n_seeds: int, gt: bool) -> list[dict]:
    """How much the outcome depends on which random frames were drawn, reusing frames already decoded."""
    frames = table.frame.unique()
    rows = []
    for seed in range(n_seeds):
        tc = fit(video, table, [int(f) for f in np.random.default_rng(1000 + seed).choice(frames, CALIBRATION_FRAMES, replace=False)])
        row = {"seed": seed, "centroids": tc.centroids, "sigma": tc.sigma, "warnings": tc.warnings}
        if gt:
            s = score(classify_table(table.drop(columns=["colour", "team"], errors="ignore"), tc))
            row |= {k: s[k] for k in ("precision", "coverage", "clear_precision", "cross_team_overlap_precision")}
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--boxes", required=True, help="gt, or a detection file")
    ap.add_argument("--every", type=int, default=1, help="label every Nth frame. The video and the label file for the tracker need 1")
    ap.add_argument("--seed", type=int, default=0, help="chooses the calibration frames")
    ap.add_argument("--seeds", type=int, default=0, help="also repeat calibration on this many other random draws")
    ap.add_argument("--video", action="store_true", dest="draw", help="draw the team video")
    ap.add_argument("--video-path", type=Path, help="the video file; default: found by name under data/")
    a = ap.parse_args()
    if a.draw and a.every != 1:
        sys.exit("--video needs --every 1")

    path = a.video_path or video_file(a.video)
    boxes = load_boxes(a.video, a.boxes)
    # The packet count can exceed the frames that decode (IMG_0104: 11400 and 11376), so don't ask for frames past the last box
    n_frames = min(frame_count(path), int(boxes.frame.max()))
    name = "gt" if a.boxes == "gt" else Path(a.boxes).stem.removeprefix(f"{a.video}__")
    chosen = calibration_frames(n_frames, a.seed)
    labelled = set(range(1, n_frames + 1, a.every))
    out = outputs_dir() / "team"
    out.mkdir(parents=True, exist_ok=True)
    background = background_image(path, chosen)
    cv2.imwrite(str(out / f"{a.video}__{name}.background.png"), background)
    table = add_colours(boxes, path, labelled | set(chosen), background)
    tc = fit(a.video, table, chosen)
    print(f"{a.video}: calibrated on frames {chosen}\n  colours {tc.centroids}\n  teams {tc.teams}  sigma {tc.sigma:.1f}  samples {tc.samples}")
    for w in tc.warnings:
        print(f"  WARNING: {w}")
    result = classify_table(table, tc)
    tc.save(out / f"{a.video}__{name}.calibration.json")
    labels = write_labels(result, a.boxes, name, a.video)
    print(f"  labels -> {labels}")
    metrics = {"video": a.video, "box_source": name, "every": a.every, "calibration": dict(tc.__dict__)}
    use_gt = a.video not in CHALLENGE_VIDEOS
    if use_gt:
        scored = truth(a.video, result, a.boxes)
        metrics |= score(scored[~scored.frame.isin(chosen)])  # calibration frames aren't scored
        print("  " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items() if k != "calibration"}))
    if a.seeds:
        base = truth(a.video, table, a.boxes) if use_gt else table
        metrics["seed_runs"] = seed_spread(a.video, base, a.seeds, use_gt)
    (out / f"{a.video}__{name}.metrics.json").write_text(json.dumps(metrics, indent=1))
    if a.draw:
        target = out / f"{a.video}__{name}.mp4"
        draw_video(path, result, tc, target, n_frames)
        print(f"  video -> {target}")


if __name__ == "__main__":
    main()
