"""UCMCTrack (https://github.com/corfyi/UCMCTrack, MIT, commit 7851a49) adapted to a static camera and a known court.

kalman.py and ucmc.py come from upstream tracker/; mapper.py replaces upstream detector/mapper.py. Changes:
- The camera model is the homography from court_registration.csv, not a camera parameter file.
- Camera motion compensation is removed: the camera is static, and UCMCTrack+ is UCMCTrack plus that step.
- Foot-point noise scales with box height, without upstream's fixed pixel limits, so it carries over between resolutions.
- Tracks with no low-score detections to compete for are now marked coasted (see ucmc.py).
- death_count means frames since the last observation, and how long a track lives is separate from what is written out.
- Measurement noise depends on detection confidence (mapper.CONF_NOISE).
- A match that barely beats a rival isn't committed, so an ambiguous box can't corrupt a track's motion state (ucmc.py).
- Optional and off: an independent Mahalanobis gate, recently-seen-first matching. Neither helped; see docs/11.
"""
