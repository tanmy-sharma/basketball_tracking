# Competition Rules & Logistics

| File | What's in it |
|---|---|
| [rules_and_data_policy.md](rules_and_data_policy.md) | Eligibility, allowed and prohibited data/models, inference rules, grey zones, evaluation rules, completion requirements, prizes, citation, compliance checklist |
| [submission_and_deliverables.md](submission_and_deliverables.md) | Timeline with countdown, exact results format, silent-zero pitfalls, coordinate frame, code/paper deliverables, freeze discipline |
| [organizer_emails.md](organizer_emails.md) | All 4 organizer emails verbatim, plus a change log of what each one superseded |
| [official_links.md](official_links.md) | Every official URL: site pages, forms, Drive folder IDs, tutorial, paper, repo, contacts |

**Precedence:** latest organizer email > earlier emails > Registration page > other website pages > paper.

## Five rules most likely to cost you the win
1. **Register by Tue Oct 27.** Without registration you can't submit.
2. **Never train on anything except the official Training set** (plus public checkpoints and allowed synthetic data).
   In particular, keep the public TrackID3x3 Outdoor GT out of the project.
3. **No manual identity work on challenge frames.** Calibration landmarks are the only manual input allowed.
4. **Exact format**: cm, frames 1…N, `A`/`B`, no spaces, two files at the ZIP root.
5. **Freeze code and weights for the final upload.** The paper must reproduce exactly that submission.
