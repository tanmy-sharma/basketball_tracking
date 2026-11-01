# TrackID3x3 starter tools

Setup: `uv sync` from the repo root, then prefix commands with `uv run`. Run scripts from this directory. Data is read from `../data` (never written).
All court coordinates are in cm (court 1505 x 1105). Frames are 1-based.

- `trackid/io.py` - roster parsing (train + final), registration -> homographies (`load_homographies` gives image->court and court->image), GT court positions, bboxes, `track_identities(video)` (track_id -> (team, jersey)), submission read/write.
- `trackid/metric.py` - `ti_hota_reference(gt_rows, pred_rows)` (port of the official code) and `ti_hota_fast` (same result, ~vectorized). Rows are `(frame, x, y, "A10")`; convert DataFrames with `io.gt_rows`. Returns `TI-HOTA`, `DetA`, `AssA` plus `*_alphas` arrays. Per video; average across videos yourself.
- `validate_submission.py DIR_OR_ZIP [--make-zip out.zip] [--frames IMG_0114=N ...]` - format checks against the roster and video frame counts; builds a root-level ZIP only when error-free.
- `sensitivity.py` - how much each error type costs in score; results in `sensitivity_results.md`.
- `viz_overlay.py VIDEO REGISTRATION.csv FRAME [--points file.txt] [--out out.png] [--scale 0.5]` - court lines and labelled points on a frame. Use the training registration for training video and the challenge one for 1280x720 videos.
- Tests: `python -m pytest tests` (about 30 s; compares fast vs reference metric).
