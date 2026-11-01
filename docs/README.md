# TrackID3x3 Challenge (ABC 2027): Documentation

Compiled 2026-10-25 from the challenge website, all 4 organizer emails, the TrackID3x3 paper (arXiv v2), the official
evaluation code, and hands-on analysis of the data in `data/`.

> ⏰ **Register by Tue Oct 27** · results due **Tue Nov 3, 23:59 AoE** (Wed Nov 4, 17:29 IST) · paper Nov 10

## Start here
- [00_brief.md](00_brief.md): a 5-minute summary of everything

## Rules and logistics → [competition/](competition/README.md)
- [rules_and_data_policy.md](competition/rules_and_data_policy.md): what's allowed or prohibited, grey zones, compliance checklist
- [submission_and_deliverables.md](competition/submission_and_deliverables.md): exact file format, pitfalls, timeline, paper/code deliverables
- [organizer_emails.md](competition/organizer_emails.md): the emails verbatim plus a change log
- [official_links.md](competition/official_links.md): every URL, Drive folder ID and contact

## Technical deep dives
| Doc | Contents |
|---|---|
| [01_task_overview.md](01_task_overview.md) | What you predict, where the data comes from, what is hard and what is easy |
| [02_evaluation_metric.md](02_evaluation_metric.md) | TI-HOTA exactly as implemented, tolerance tables, the "wrong row costs 2×" algebra, measured sensitivity |
| [03_dataset.md](03_dataset.md) | Every file and format, **how GT positions were generated (bbox bottom-centre)**, bib colours, distractors, calibration |
| [04_paper_and_baseline.md](04_paper_and_baseline.md) | The paper's subsets, the baseline pipeline step by step, the 71.03 score and why it's not comparable |
| [05_related_work.md](05_related_work.md) | SoccerNet GSR winners, jersey OCR, team clustering, detectors, trackers, licences |
| [06_strategy_and_ideas.md](06_strategy_and_ideas.md) | Where the points are, a 6-identity global formulation, convention-matching localization, validation plan, 9-day schedule |
| [07_open_questions_and_risks.md](07_open_questions_and_risks.md) | Questions to email the organizers; risk register |

## Reference material
- [reference/compute_TI-HOTA_official.py](reference/compute_TI-HOTA_official.py): official metric code (verbatim)
- [reference/TrackID3x3_arXiv2503.18282v2.pdf](reference/TrackID3x3_arXiv2503.18282v2.pdf): the dataset paper
- [assets/](assets/): annotated frames (calibration overlays, bib close-ups)

## Tooling
`../tools/` holds loaders, the official-faithful plus a fast TI-HOTA, a submission validator and ZIP builder, a
sensitivity study, and an overlay visualizer. See `../tools/README.md`.
