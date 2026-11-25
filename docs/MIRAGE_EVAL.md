MIRAGE Evaluation (Day 11)

Evaluated on the 40-image synthetic invoice set (phantom/mirage/make_samples.py, 20 benign/suspicious pairs across 3 template families). Document-level prediction: "suspicious" if any non-quality finding is suspicious or inconclusive.

python -m phantom.mirage.evaluate --samples data/samples

Results (all templates)
Metric	Value	v1 target (SCOPE.md)
Documents	40	—
True positives	9	—
False positives	7	—
False negatives	11	—
True negatives	13	—
Precision	0.562	—
Recall	0.45	—
F1	0.50	≥ 0.70
False-positive rate	0.35	—
Mean runtime / document	0.54 s	≤ 5 s/page


F1 is below the v1 target. These are the measured results from the regenerated sample set. Runtime is within the project's per-page budget.

What's driving the gap
- field_alignment uses a MAD-based z-score on amount-column left edges. It may still flag benign documents; the current false-positive rate is 0.35.
- region_inconsistency may rarely fire on this synthetic set: the generator paints over a field with white before redrawing it, so the patched region's background matches the page. Test this check on non-synthetic manipulated documents.
- font_consistency uses ink-density outliers to detect font mismatches. It may miss altered amounts when replacement digits use the same font; a numeric cross-check against the invoice total could help.

Next steps (Week 3)
- Add a numeric consistency check: do the item amounts sum to the stated total?
- Re-run the evaluation against non-synthetic manipulated documents.
- Evaluate generalization by holding out one template family from threshold tuning.