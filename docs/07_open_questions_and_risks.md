# Open Questions and Risks

## Questions to email the organizers (trackid3x3-challenge-2026@g.sp.m.is.nagoya-u.ac.jp)

1. **Offline post-processing: answered, allowed** if automatic (`competition/organizer_emails.md`, Email 5). Covers
   whole-video tracklet-to-identity assignment, bidirectional smoothing and gap interpolation using future frames.
2. **Challenge GT provenance**: is the challenge GT, like the training GT, the bottom-centre of boxes from the synchronized
   end-line camera projected with that camera's homography? Or were boxes annotated in the oblique view? (It may not
   be answered, but asking is legitimate.)
3. **Frames where a player is out of the GT camera's view**: will they have no GT row, as in some training frames?
4. **Score aggregation**: is TI-HOTA averaged per video (as in the notebook) or computed over the pooled frames of both videos?
5. **Loader for the 5-column format**: confirm that the attribute is built as team + jersey (e.g. `A10`) and whether
   whitespace is stripped.
6. **Learned priors**: may a colour→team mapping learned from the Training set (the same teams appear there) be used at inference?
7. **The "three main challenge papers"** to cite: which ones?
8. Tutorial Zoom password (if needed); FAQ content (it is in images on the site).

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Missing the registration deadline (Oct 27) | — | Disqualification | Register today |
| Team letter A/B swapped in one video | Medium | −50 % of that video | Automatic roster-consistency check (jersey sets); validator assertion |
| Coordinates in m, frame offset, or spaces in fields | Low | Catastrophic | `tools/validate_submission.py` before every upload |
| Systematic localization bias from the convention mismatch | High | −10 to −30 % | §2 of the strategy doc; geometric estimate of the bias |
| Identity exchange after a crossing persists for minutes | Medium | Large | Tracklet splitting at occlusions plus global assignment |
| Scorekeeper (purple #10) or referee labelled as a player | Medium | Several % | Colour gating; never more than 6 rows |
| Rule violation (e.g. using public Outdoor GT or pose labels) | Low if careful | Disqualification | Keep them out of the repo; compliance checklist |
| Late upload or form issues | Low | Total | Upload a valid baseline by Oct 27–29 and keep overwriting |
| Final code and weights not reproducible | Medium | Paper rejection / DQ | Tag and hash on every upload; run from a clean env on Nov 2 |
| HEVC 4K decode too slow on the laptop | Medium | Lost time | Pre-extract downscaled frames once; use ffmpeg hardware decode (VideoToolbox) |
| Overfitting the smoother or offset to the end-line view | Medium | Some | Keep the model low-capacity and physically motivated; geometric sanity checks |
