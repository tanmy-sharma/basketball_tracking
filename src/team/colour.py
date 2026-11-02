"""Bib colour of a player crop: calibrate on a few frames of a video, then classify any crop.

Colour is compared in Lab a and b only, so brightness changes (sun, shadow) matter less. No model or weights are involved.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Median a, b of each bib across the 8 training games. These only say which group is which and rule out other colours;
# the colours actually used for classifying are measured on each video by calibrate().
BIB_REFERENCE = {"green": (-18.0, 22.0), "yellow": (-7.0, 30.0), "pink": (21.0, -1.0), "purple": (15.0, -21.0)}
BIB_DRAW = {"green": (60, 200, 60), "yellow": (0, 230, 255), "pink": (180, 105, 255), "purple": (200, 50, 130)}  # BGR

MIN_CHROMA = 15.0  # below this a pixel is black, white or grey
MIN_VIVID = 0.15  # share of the person's pixels that must be vivid, or the box has no bib colour
BACKGROUND_DIFF = 20.0  # Lab distance from the empty-court image above which a pixel belongs to the person
MIN_PERSON = 0.15  # share of the region that must differ from the background
MIN_PIXELS = 30
COURT_BLUE_HUE = (100, 124)  # OpenCV hue. Purple bibs sit above this
COURT_BLUE_SAT = 70
MAX_HUE_GAP = 35.0  # degrees from the nearest expected bib, in calibration
MIN_SIGMA = 3.0
MIN_SAMPLES = 10  # per colour
MIN_GAP = 8.0  # between the two colours


def upper_body(crop: np.ndarray) -> np.ndarray:
    """The middle of the chest from a whole-player crop, without needing a pose model."""
    h, w = crop.shape[:2]
    return crop[int(.15 * h):int(.5 * h), int(.25 * w):int(.75 * w)]


def crop_colour(crop: np.ndarray, whole_player: bool = True, background: np.ndarray | None = None
                ) -> tuple[str, np.ndarray | None]:
    """BGR crop -> (status, [a, b]). Status is ok, no_colour (nothing vivid, such as a black shirt) or degenerate.

    background is the same box cut from the empty-court image. It removes grass and court, which otherwise pass for a bib.
    """
    region = upper_body(crop) if whole_player else crop
    if region.size == 0 or region.shape[0] * region.shape[1] < MIN_PIXELS:
        return "degenerate", None
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    ab = lab[:, 1:] - 128
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    person = (hsv[:, 0] < COURT_BLUE_HUE[0]) | (hsv[:, 0] > COURT_BLUE_HUE[1]) | (hsv[:, 1] <= COURT_BLUE_SAT)
    if background is not None:
        back = upper_body(background) if whole_player else background
        back = cv2.cvtColor(back, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
        person &= np.linalg.norm(lab - back, axis=1) > BACKGROUND_DIFF
    if person.sum() < MIN_PERSON * len(ab) or person.sum() < 5:
        return "no_colour", None
    vivid = ab[person & (np.hypot(ab[:, 0], ab[:, 1]) > MIN_CHROMA)]
    if len(vivid) < max(MIN_VIVID * person.sum(), 5):
        return "no_colour", None
    return "ok", np.median(vivid, axis=0)


@dataclass
class ColourResult:
    colour: str  # a bib colour, none (not a player's colour) or unsure
    team: str | None  # A or B, only when colour is a bib colour
    probs: dict[str, float] = field(default_factory=dict)  # one per bib colour plus none. Uncalibrated
    status: str = "ok"
    ab: tuple[float, float] | None = None


@dataclass
class TeamColours:
    video: str
    centroids: dict[str, list[float]]  # bib -> [a, b], measured on this video
    teams: dict[str, str]  # bib -> A or B
    sigma: float
    frames: list[int] = field(default_factory=list)
    samples: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    gate_sigmas: float = 4.0  # further than this from both colours means not a player's colour
    min_prob: float = 0.8  # a weaker winner is unsure

    def classify(self, ab: np.ndarray) -> ColourResult:
        ab = np.asarray(ab, np.float64)
        bibs = list(self.centroids)
        d = np.array([np.linalg.norm(ab - self.centroids[b]) for b in bibs]) / self.sigma
        weights = np.exp(-.5 * np.r_[d, self.gate_sigmas] ** 2)
        total = weights.sum()
        p = weights / total if total > 0 else np.r_[np.zeros(len(bibs)), 1.0]
        probs = {b: float(v) for b, v in zip(bibs, p)} | {"none": float(p[-1])}
        best = int(p.argmax())
        pair = (float(ab[0]), float(ab[1]))
        if best == len(bibs):
            return ColourResult("none", None, probs, ab=pair)
        if p[best] < self.min_prob:
            return ColourResult("unsure", None, probs, ab=pair)
        return ColourResult(bibs[best], self.teams[bibs[best]], probs, ab=pair)

    def run(self, crop: np.ndarray, whole_player: bool = True, background: np.ndarray | None = None) -> ColourResult:
        """BGR crop -> colour and team. Pass whole_player=False if the crop is already a torso, such as from ViTPose.

        background is the same box cut from the video's empty-court image, and is strongly recommended.
        """
        status, ab = crop_colour(crop, whole_player, background)
        if status == "degenerate":
            return ColourResult("unsure", None, status=status)
        if status == "no_colour":
            return ColourResult("none", None, {b: 0.0 for b in self.centroids} | {"none": 1.0}, status)
        return self.classify(ab)

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path | str) -> "TeamColours":
        return cls(**json.loads(Path(path).read_text()))


def _hue(ab: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(ab[..., 1], ab[..., 0]))


def calibrate(video: str, samples: np.ndarray, teams: dict[str, str], frames: list[int] | None = None) -> TeamColours:
    """samples: Nx2 a, b of bib-coloured boxes. teams: bib -> A or B, from the roster.

    Boxes are grouped by the nearest expected hue, which also drops skin and bibs of teams not in this game. Then the two
    colours are refined on this video, so they reflect its own camera and light.
    """
    bibs = list(teams)
    samples = np.asarray(samples, np.float64).reshape(-1, 2)
    expected = _hue(np.array([BIB_REFERENCE[b] for b in bibs]))
    gaps = np.abs((_hue(samples)[:, None] - expected + 180) % 360 - 180)
    keep = gaps.min(1) <= MAX_HUE_GAP
    points, group = samples[keep], gaps[keep].argmin(1)
    warnings = []
    centres = np.array([np.median(points[group == i], axis=0) if (group == i).any() else BIB_REFERENCE[b]
                        for i, b in enumerate(bibs)], np.float64)
    for _ in range(5):
        if not len(points):
            break
        group = np.linalg.norm(points[:, None] - centres, axis=2).argmin(1)
        centres = np.array([np.median(points[group == i], axis=0) if (group == i).any() else centres[i]
                            for i in range(len(bibs))])
    counts = {b: int((group == i).sum()) for i, b in enumerate(bibs)}
    for b, n in counts.items():
        if n < MIN_SAMPLES:
            warnings.append(f"only {n} samples of {b}")
    gap = float(np.linalg.norm(centres[0] - centres[1]))
    if gap < MIN_GAP:
        warnings.append(f"colours are only {gap:.1f} apart")
    if len(points):
        dist = np.linalg.norm(points - centres[group], axis=1)
        sigma = max(MIN_SIGMA, 1.4826 * float(np.median(dist)))
    else:
        sigma = MIN_SIGMA
    return TeamColours(video, {b: centres[i].tolist() for i, b in enumerate(bibs)}, teams, sigma, sorted(frames or []),
                       counts, warnings)
