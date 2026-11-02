"""Bridge to tools/trackid: labelled player boxes with team, jersey and bib. The only module that touches tools/."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.core.constants import BIB_OF_ROSTER, CHALLENGE_VIDEOS, NUMBER_TO_IDX, NUMBERS
from src.core.paths import add_tools_to_path

log = logging.getLogger(__name__)
COLUMNS = ["frame", "track_id", "x", "y", "w", "h", "team", "jersey", "number_idx", "bib", "in_game"]


def _io():
    add_tools_to_path()
    from trackid import io
    return io


def _roster(video: str):
    if video in CHALLENGE_VIDEOS:
        raise ValueError(f"{video} is a challenge video and has no labels")
    io = _io()
    rosters = io.load_roster(io.TRAIN / "games_roster.csv")
    if video not in rosters:
        raise KeyError(f"{video} not in games_roster.csv")
    return io, rosters[video]


def load_labelled_boxes(video: str) -> pd.DataFrame:
    """Boxes (4K pixels, top-left xywh) for tracks on the roster; in_game means frame >= game_start_frame."""
    io, ro = _roster(video)
    ident = io.track_identities(video, ro)
    bb = io.load_bboxes(video)
    stray = ~bb.track_id.isin(list(ident))
    if stray.any():
        ids = sorted(bb.track_id[stray].unique().tolist())
        log.warning("%s: dropped %d boxes on %d off-roster track ids %s", video, stray.sum(), len(ids), ids)
    bb = bb[~stray]
    dup = bb.duplicated(["frame", "track_id"], keep="first")
    if dup.any():
        log.warning("%s: dropped %d duplicate (frame, track_id) rows", video, dup.sum())
    bb = bb[~dup].copy()
    bb["team"] = bb.track_id.map(lambda t: ident[t][0])
    bb["jersey"] = bb.track_id.map(lambda t: ident[t][1])
    bb["number_idx"] = bb.jersey.map(NUMBER_TO_IDX).astype("int64")
    bb["bib"] = bb.team.map(lambda t: BIB_OF_ROSTER.get(frozenset(ro.teams[t]), ""))
    bb["in_game"] = bb.frame >= (ro.game_start or 0)
    return bb[COLUMNS].sort_values(["frame", "track_id"], ignore_index=True)


def roster_mask(video: str, team: str) -> np.ndarray:
    """Bool (9,) in NUMBERS order: the numbers on `team`'s roster in `video`."""
    _, ro = _roster(video)
    if team not in ro.teams:
        raise KeyError(f"{video} has no team {team!r}")
    return np.array([n in ro.teams[team] for n in NUMBERS])
