
# From UCMCTrack tracker/ucmc.py (MIT, see LICENSE.UCMCTrack). Changes: no CMC, unmatched tracks always coast, plus the options listed on the class.
from __future__ import print_function

import numpy as np
from lap import lapjv


from .kalman import KalmanTracker,TrackStatus


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

BLOCKED = 1e6  # cost of a pair the gate rejects; far above any ranking limit
CLEAR_EVIDENCE = 4.0  # box evidence above this (call right > 98%) counts as a clear box for the veto
SURE_TEAM = 3.0  # track log-odds above this (95%) counts as sure of its team for the birth rule
SLOTS_PER_TEAM = 3


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class UCMCTrack(object):
    """Upstream's association, with these additions (each is off at its default):

    gate:       reject pairs above this squared Mahalanobis distance, separately from the a1/a2 ranking limit
    ambiguity:  if a match beats a competing pair by less than this cost, don't commit: the track coasts and the box is dropped
    recency:    match tracks seen last frame before tracks already coasting

    Team (needs det.team, +1/-1/0, and det.evidence, the logit of that call being right):
    Every track keeps team_belief, log-odds of team A, summed from the boxes it matched and capped at +-team_cap.
    team_scale: the association cost is a -2 log-likelihood, so a box adds team_scale * -2 log(P(its call | track) / 0.5).
                0 for a box with no call or a track with no belief, up to about 2 logit(q) for a sure track of the other team.
    team_veto:  a clear box can't join a track at least this sure (log-odds) of the other team
    team_birth: no new track from a box whose team already has 3 sure tracks matched this frame
    team_lock_frames: after this many matched frames in a row sure of one team, the track's team never changes again
    """

    def __init__(self,a1,a2,wx, wy,vmax, max_age, fps, high_score, gate=0.0, ambiguity=0.0, recency=False, birth_hits=3,
                 team_scale=0.0, team_cap=10.0, team_decay=1.0, team_veto=0.0, team_birth=False, team_learn_overlap=1.0,
                 team_lock_frames=0):
        self.wx = wx
        self.wy = wy
        self.vmax = vmax
        self.high_score = high_score
        self.max_age = max_age
        self.a1 = a1
        self.a2 = a2
        self.dt = 1.0/fps
        self.gate = gate
        self.ambiguity = ambiguity
        self.recency = recency
        self.birth_hits = birth_hits
        self.team_scale = team_scale
        self.team_cap = team_cap
        self.team_decay = team_decay
        self.team_veto = team_veto
        self.team_birth = team_birth
        self.team_learn_overlap = team_learn_overlap
        self.team_lock_frames = team_lock_frames

        self.trackers = []
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []
        self.stats = dict(births=0, confirmed=0, abstained=0)


    def update(self, dets,frame_id):

        self.data_association(dets,frame_id)

        self.associate_tentative(dets)

        self.initial_tentative(dets)

        self.delete_old_trackers()

        self.update_status(dets)

    def cost(self, det, trk):
        mahalanobis, logdet = trk.distance_terms(det.y, det.R)
        if self.gate and mahalanobis > self.gate:
            return BLOCKED
        team = getattr(det, "team", 0)
        if team and (self.team_scale or self.team_veto):
            agree = team * trk.team_belief
            if self.team_veto and agree <= -self.team_veto and det.evidence >= CLEAR_EVIDENCE:
                return BLOCKED
            if self.team_scale:
                return mahalanobis + logdet + self.team_scale * self.team_penalty(det.evidence, agree)
        return mahalanobis + logdet

    @staticmethod
    def team_penalty(evidence, agree):
        """-2 log(P(call | track) / 0.5), floored at 0: no reward for agreeing, so team alone never pulls a box in."""
        q, pi = sigmoid(evidence), sigmoid(agree)
        return max(0.0, -2.0 * np.log(2.0 * (q * pi + (1.0 - q) * (1.0 - pi))))

    def add_team(self, trk, det):
        if getattr(trk, "team_locked", False):
            return
        team = getattr(det, "team", 0)
        if team:
            # Overlapped calls are wrong 6% of the time, and in runs: a few in a row could flip a sure track
            w = self.team_learn_overlap if getattr(det, "overlapped", False) else 1.0
            b = self.team_decay * trk.team_belief + team * det.evidence * w
            trk.team_belief = float(np.clip(b, -self.team_cap, self.team_cap))
        if self.team_lock_frames:
            self.update_lock(trk)

    def update_lock(self, trk):
        """Count matched frames in a row that the track is sure (|belief| >= SURE_TEAM) of the same team; at team_lock_frames
        its team is fixed at +-team_cap for the rest of its life. Frames with no call count if the belief stays sure."""
        sign = 1 if trk.team_belief >= SURE_TEAM else -1 if trk.team_belief <= -SURE_TEAM else 0
        if sign and sign == getattr(trk, "lock_sign", 0):
            trk.lock_run += 1
        else:
            trk.lock_sign, trk.lock_run = sign, 1 if sign else 0
        if trk.lock_run >= self.team_lock_frames:
            trk.team_locked = True
            trk.team_belief = sign * self.team_cap

    def assign(self, dets, detidx, trackidx, limit, abstain=True):
        """One Hungarian pass. Returns (matched pairs, unmatched det indices, unmatched track indices, abstained det indices)."""
        if len(detidx)*len(trackidx) == 0:
            return [], list(detidx), list(trackidx), []
        cost_matrix = np.zeros((len(detidx), len(trackidx)))
        for i, det_idx in enumerate(detidx):
            for j, trk_idx in enumerate(trackidx):
                cost_matrix[i,j] = self.cost(dets[det_idx], self.trackers[trk_idx])
        matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix, limit)

        pairs, abstained = [], []
        rest_d, rest_t = [detidx[i] for i in unmatched_a], [trackidx[j] for j in unmatched_b]
        for i,j in matched_indices:
            if abstain and self.ambiguity and self.is_ambiguous(cost_matrix, i, j, limit):
                abstained.append(detidx[i])
                rest_t.append(trackidx[j])
                self.stats["abstained"] += 1
            else:
                pairs.append((detidx[i], trackidx[j]))
        return pairs, rest_d, rest_t, abstained

    def is_ambiguous(self, cost_matrix, i, j, limit):
        """Another acceptable track for this box, or box for this track, costs nearly the same."""
        best = cost_matrix[i,j]
        rivals = np.r_[np.delete(cost_matrix[i,:], j), np.delete(cost_matrix[:,j], i)]
        rivals = rivals[rivals < limit]
        return len(rivals) > 0 and rivals.min() - best < self.ambiguity

    def commit(self, dets, det_idx, trk_idx):
        trk = self.trackers[trk_idx]
        trk.update(dets[det_idx].y, dets[det_idx].R)
        trk.death_count = 0
        trk.detidx = det_idx
        trk.status = TrackStatus.Confirmed
        dets[det_idx].track_id = trk.id
        self.add_team(trk, dets[det_idx])

    def data_association(self, dets,frame_id):
        # Separate detections into high score and low score
        detidx_high = []
        detidx_low = []
        for i in range(len(dets)):
            if dets[i].conf >= self.high_score:
                detidx_high.append(i)
            else:
                detidx_low.append(i)

        # Predcit new locations of tracks
        for track in self.trackers:
            track.predict()

        for trk in self.trackers:
            trk.detidx = -1

        # Associate high score detections with tracks, optionally the most recently seen tracks first
        stages = [self.confirmed_idx, self.coasted_idx] if self.recency else [self.confirmed_idx + self.coasted_idx]
        det_left, trackidx_remain = detidx_high, []
        for trackidx in stages:
            pairs, det_left, trk_left, _ = self.assign(dets, det_left, trackidx, self.a1)
            for det_idx, trk_idx in pairs:
                self.commit(dets, det_idx, trk_idx)
            trackidx_remain += trk_left
        self.detidx_remain = det_left

        # Associate low score detections with remain tracks
        pairs, _, trk_left, _ = self.assign(dets, detidx_low, trackidx_remain, self.a2)
        for det_idx, trk_idx in pairs:
            self.commit(dets, det_idx, trk_idx)
        # Upstream only did this when low-score boxes existed, leaving other unmatched tracks Confirmed and undeletable
        for trk_idx in trk_left:
            self.trackers[trk_idx].status = TrackStatus.Coasted
            self.trackers[trk_idx].detidx = -1


    def associate_tentative(self, dets):
        # New tracks have no identity to protect yet, and abstaining here would stop nearby tracks from ever confirming
        pairs, det_left, trk_left, _ = self.assign(dets, self.detidx_remain, self.tentative_idx, self.a1, abstain=False)
        for det_idx, trk_idx in pairs:
            trk = self.trackers[trk_idx]
            trk.update(dets[det_idx].y, dets[det_idx].R)
            trk.death_count = 0
            trk.birth_count += 1
            trk.detidx = det_idx
            dets[det_idx].track_id = trk.id
            self.add_team(trk, dets[det_idx])
            if trk.birth_count >= self.birth_hits - 1:
                trk.birth_count = 0
                trk.status = TrackStatus.Confirmed
                self.stats["confirmed"] += 1
        for trk_idx in trk_left:
            self.trackers[trk_idx].detidx = -1
        self.detidx_remain = det_left


    def full_teams(self):
        """Teams (+1/-1) that already have 3 tracks sure of that team, confirmed and matched this frame."""
        n = {1: 0, -1: 0}
        for trk in self.trackers:
            if trk.status == TrackStatus.Confirmed and trk.detidx >= 0 and abs(trk.team_belief) >= SURE_TEAM:
                n[1 if trk.team_belief > 0 else -1] += 1
        return {s for s, k in n.items() if k >= SLOTS_PER_TEAM}

    def initial_tentative(self,dets):
        full = self.full_teams() if self.team_birth else set()
        for i in self.detidx_remain:
            if getattr(dets[i], "team", 0) in full:
                self.stats["births_blocked"] = self.stats.get("births_blocked", 0) + 1
                continue
            self.trackers.append(KalmanTracker(dets[i].y,dets[i].R,self.wx,self.wy,self.vmax, dets[i].bb_width,dets[i].bb_height,self.dt))
            self.trackers[-1].status = TrackStatus.Tentative
            self.trackers[-1].detidx = i
            self.trackers[-1].team_belief = 0.0
            self.add_team(self.trackers[-1], dets[i])
            self.stats["births"] += 1
        self.detidx_remain = []

    def delete_old_trackers(self):
        # death_count is frames since the last observation; upstream also counted the frame of the match itself
        kept = []
        for trk in self.trackers:
            if trk.detidx < 0:
                trk.death_count += 1
            dead = (trk.status == TrackStatus.Coasted and trk.death_count > self.max_age) or (
                trk.status == TrackStatus.Tentative and trk.death_count >= 1)
            if not dead:
                kept.append(trk)
        self.trackers = kept

    def update_status(self,dets):
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []
        for i in range(len(self.trackers)):

            detidx = self.trackers[i].detidx
            if detidx >= 0 and detidx < len(dets):
                self.trackers[i].h = dets[detidx].bb_height
                self.trackers[i].w = dets[detidx].bb_width

            if self.trackers[i].status == TrackStatus.Confirmed:
                self.confirmed_idx.append(i)
            elif self.trackers[i].status == TrackStatus.Coasted:
                self.coasted_idx.append(i)
            elif self.trackers[i].status == TrackStatus.Tentative:
                self.tentative_idx.append(i)
