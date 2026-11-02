The current outputs are **not ready for blind submission**. Their file structure and provenance appear sound, and several sampled identities agree with visible bibs, but the evidence round verifies a serious mechanism that promotes unreadable nonplayer tracks into roster identities and suppresses better-supported player tracks. This is repairable automatically without Challenge ground truth or manual identity corrections.

1. **Highest priority: identity evidence is being amplified incorrectly, with a verified downstream consequence.**

   I inspected IMG_0115’s raw and overlay frames around 64.7–68.7 seconds. A black-shirt person consistently carries predicted B1 while a pink-bibbed player remains anonymous. The parent’s exact artifact trace makes the conclusion stronger than visual suspicion: at frames 1999–2001, the black track follows original track 13 → piece 13 → known ID 4 → submitted B1; the pink track follows original track 19 → eligible piece 19 → conflict/unknown 104 → no submitted row.

   The black track’s frame-1999 read is especially diagnostic. Eligibility is approximately 0.102, its raw number probabilities favor 12 at 0.826, and raw probability for 1 is only 0.046. Normalizing separately within team B’s roster converts that small B-roster mass into almost certain B1. Its color observation is NONE with probability 1.0. This is not a reliable B1 reading: roster conditioning, persistent team belief, and fallback aggregation manufacture apparent confidence from unsuitable evidence.

   Meanwhile, the pink track has eligibility approximately 0.944 and raw probability for 1 approximately 0.9999. Nevertheless, its piece loses the shared identity reservation because the unreadable piece’s accumulated strength is 985.8 versus 550.8. The source confirms that eligible and ALL-fallback pieces compete in the same length-weighted priority pool.

   This causal chain would be weakened by a broken frame/box join or mistaken lineage. The supplied exact frame mapping, visually distinct boxes, and consistent neighboring reads strongly disfavor those alternatives here. They do not establish the correctness of every join elsewhere.

2. **Exporter guessing and contradictory bridging independently defeat abstention.**

   The source excerpts establish two additional failure paths. A conflicting piece can receive a bridge identity without checking whether its own strong number evidence contradicts that identity; piece 32’s strong B1 evidence followed by a B5 bridge is an actual artifact counterexample. Separately, the exporter assigns jerseys to unknown pieces even when their own jersey scores are unavailable. Identity uniqueness after deduplication therefore does not mean identity correctness.

   The supplied accounting is consequential: IMG_0115’s 11,552 B1 rows comprise 7,659 named ALL-fallback rows, 486 rows guessed from an unnamed ALL-source piece, 3,405 guessed rows from eligible conflict pieces, and two fill rows. The first two groups total the user’s reported 8,145 rows exactly. That arithmetic verifies provenance, **not visual nonplayer status for every row**; only the inspected subset was visually checked. Similarly, IMG_0114’s 235 resurrected ALL/NONE rows substantiate the export mechanism without independently certifying all 235 visual identities.

   A direct falsifier for the proposed remedy is straightforward: if rejected evidence can still produce a named export through another branch, the remedy is incomplete.

3. **The selected IMG_0114 low-count interval does not demonstrate a detector collapse. It reveals a court-filter confound.**

   I inspected the raw/overlay sequence before, during, and after 243–247 seconds, then requested cached candidate overlays and projected coordinates. At frame 7314, four active bibbed players are visible and all four have boxes; the other pair has moved toward or beyond the right image boundary. At frames 7284 and 7404, high-confidence detections of the moving pair exist inside the image but project outside the court and are excluded by input/output court limits.

   Thus, “fewer than six on-court detections” mixes visibility, physical movement beyond the painted boundary, filtering, and possibly other failures. It cannot establish missing or merged in-view players. The supplied Training-only analysis also shows official targets outside the nominal court rectangle, so strict court clipping is an implementation choice, not a demonstrated competition requirement.

   This warrants a separate automatic ablation: retain roster-supported established tracks beyond the court boundary, with bounded continuity, while restricting unsupported births. Simply expanding the court for every person could increase nonplayer contamination. Shrinking it to exclude the black-shirt person would address neither the verified evidence failure nor the rule-compliant generalization requirement.

4. **There is limited positive evidence, but the global accuracy claims remain unverified.**

   I inspected predetermined quarter, half, and three-quarter IMG_0114 samples, including neighboring frames. Visible yellow/green 6s around frame 5768 and green/yellow 9s around frame 8652 agree with their overlays. The sampled short trajectories show no obvious team flip. That supports those observations; it does not establish that shared 6/9 identities never swap across the video.

   Full frame counts, bounded row counts, zero duplicate identities, roster coverage, and reproducible weight/cache provenance are useful structural assurances. Color separation, number-color agreement, and high selected OCR confidence are weaker accuracy evidence because color and OCR can share the same contaminated crop, and roster conditioning can hide contrary evidence.

   Two explanations should explicitly be retired: the apparent threefold player-size difference compares incompatible resolutions, and the very low raw confidence median in IMG_0115 does not describe the tracker-qualified detections. Viewpoint and occlusion remain plausible difficulties, but neither supplies a measured Challenge accuracy estimate.

5. **The smallest worthwhile repair is an end-to-end identity-admission contract, followed by Training validation.**

   Keep the existing detections and geometric tracks initially. Change identity handling together:

   - Zero-eligible ALL-fallback OCR must not directly reserve a roster identity. Admit candidates using readable evidence and consistent bib/team support; preserve raw cross-team number mass instead of treating separately normalized roster probabilities as calibrated certainty.
   - Allow unreadable spans to inherit an established identity through sufficiently constrained continuity. Absence of a readable number alone must not delete a true player.
   - Prevent bridging from overriding strong, admitted contradictory evidence. If identity remains unresolved, retain an internal unknown.
   - Make export preserve that unknown status; unused roster slots are not evidence for a jersey assignment.

   This is smaller and better supported than retraining models or rewriting the motion tracker. A birth/player-admission improvement may also help, but it should use repeated actual bib observations rather than counting no-call matches as fresh confirmation. Eligibility itself is a readability score, not a player classifier.

   Validate each change separately and together on official Training: exact metric, per-video identity precision/recall, losses among genuinely unreadable player spans, conflict resolutions, and nonplayer false assignments. Treat video 108 as model-held-out but development-used, not pristine. Tune thresholds on Training, then freeze the policy.

   Challenge checks should remain automatic consistency diagnostics: every named identity must have admissible provenance; unknowns must not become named solely at export; bridges must respect contradiction checks; joins must be unique and auditable; and boundary crossings must have explained disposition. Reduced coverage may accompany better precision, so coverage alone must not select the repair.

No Challenge GT was used, and no replacement predictions were generated in this review. The evidence supports repairing identity admission, bridging, and export before submission; it does not support a numeric Challenge score prediction or a blanket claim that all remaining identities are correct.