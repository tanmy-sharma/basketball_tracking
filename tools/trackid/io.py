"""Loaders/writers for the TrackID3x3 data. Court coordinates are in cm."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
TRAIN = DATA / "Training"
FINAL = DATA / "Final challenge"
GT_COURT = TRAIN / "full player annotations" / "court position"
GT_BBOX = TRAIN / "full player annotations" / "Image-space bounding boxes"
COURT_W, COURT_H = 1505.0, 1105.0

Rows = list[tuple[int, float, float, str]]  # (frame, x, y, attr) as used by the metric


@dataclass
class Roster:
    video: str
    teams: dict[str, list[str]]  # {"A": ["4","10","12"], "B": [...]}
    game_start: int | None = None
    track_jersey: dict[int, str] = field(default_factory=dict)
    offense: list[int] = field(default_factory=list)
    defense: list[int] = field(default_factory=list)

    @property
    def identities(self) -> set[tuple[str, str]]:
        return {(t, j) for t, js in self.teams.items() for j in js}


def _ints(s: str) -> list[int]:
    return [int(v) for v in re.findall(r"\d+", s)]


def parse_teams(s: str) -> dict[str, list[str]]:
    """'A: 4, 10, 12; B: 6, 9, 11' -> {'A': ['4','10','12'], 'B': [...]} (space tolerant)."""
    out = {}
    for part in s.split(";"):
        if ":" in part:
            name, rest = part.split(":", 1)
            out[name.strip()] = [str(v) for v in _ints(rest)]
    return out


def load_roster(path: Path | str) -> dict[str, Roster]:
    """Works for both Training (rich) and Final challenge (video, 'team: jersey') rosters."""
    df = pd.read_csv(path, dtype=str, skipinitialspace=True).fillna("")
    out = {}
    for _, r in df.iterrows():
        ro = Roster(r["video"].strip(), parse_teams(r["team: jersey"]))
        if "track_id: jersey" in r:
            ro.game_start = int(r["game_start_frame"])
            ro.track_jersey = {int(a): b for a, b in re.findall(r"(\d+)\s*:\s*(\d+)", r["track_id: jersey"])}
            ro.offense, ro.defense = _ints(r["offense_track_ids"]), _ints(r["defense_track_ids"])
        out[ro.video] = ro
    return out


def load_homographies(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """court_registration.csv -> (H image->court, H court->image), 3x3."""
    df = pd.read_csv(path)
    img = df[["image_x", "image_y"]].to_numpy(np.float64)
    crt = df[["court_x", "court_y"]].to_numpy(np.float64)
    H, _ = cv2.findHomography(img, crt, 0)
    return H, np.linalg.inv(H)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def load_court_gt(video: str) -> pd.DataFrame:
    """Columns: frame, x, y, team, jersey (jersey is str)."""
    return pd.read_csv(GT_COURT / f"{video}.txt", header=None, names=["frame", "x", "y", "team", "jersey"],
                       dtype={"team": str, "jersey": str})


def load_bboxes(video: str) -> pd.DataFrame:
    """Columns: frame, track_id, x, y, w, h (top-left, 4K pixels)."""
    df = pd.read_csv(GT_BBOX / f"{video}.txt", header=None,
                     names=["frame", "track_id", "x", "y", "w", "h", "conf", "a", "b"])
    return df.drop(columns=["conf", "a", "b"])


def track_identities(video: str, roster: Roster | None = None, n_check: int = 200) -> dict[int, tuple[str, str]]:
    """track_id -> (team, jersey). Jersey comes from the roster; team is resolved by matching the
    homography-projected bbox bottom-centre to the nearest GT court position with that jersey."""
    roster = roster or load_roster(TRAIN / "games_roster.csv")[video]
    H, _ = load_homographies(TRAIN / "court_registration.csv")
    bb, gt = load_bboxes(video), load_court_gt(video)
    out = {}
    for tid, jersey in roster.track_jersey.items():
        b = bb[bb.track_id == tid].head(n_check)
        pos = apply_h(H, np.c_[b.x + b.w / 2, b.y + b.h])
        g = gt[gt.jersey == jersey].set_index("frame")
        votes: dict[str, int] = {}
        for f, p in zip(b.frame, pos):
            if f not in g.index:
                continue
            c = g.loc[[f]]
            team = c.team.iloc[int(np.hypot(c.x - p[0], c.y - p[1]).argmin())]
            votes[team] = votes.get(team, 0) + 1
        out[tid] = (max(votes, key=votes.get), jersey)
    return out


def gt_rows(df: pd.DataFrame) -> Rows:
    """GT/pred DataFrame (frame,x,y,team,jersey) -> metric rows with attr = team+jersey."""
    return list(zip(df.frame.astype(int), df.x.astype(float), df.y.astype(float), df.team + df.jersey))


def write_submission(path: Path | str, rows: list[tuple[int, float, float, str, str]]) -> None:
    """rows: (frame, x_cm, y_cm, team, jersey). No header, no spaces."""
    with open(path, "w") as f:
        for fr, x, y, t, j in rows:
            f.write(f"{int(fr)},{x:.2f},{y:.2f},{t},{j}\n")


def read_submission(path: Path | str) -> Rows:
    """Mirrors the official parser: split(',') with no per-field strip; attr = team + jersey."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                p = line.split(",")
                rows.append((int(p[0]), float(p[1]), float(p[2]), p[3] + p[4]))
    return rows
