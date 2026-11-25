# MIRAGE Evaluation

## Headline: real forged receipts (Find it again!, ICDAR 2023)

[Find it again!](https://l3i-share.univ-lr.fr/2023Finditagain/) contains 988 real scanned
receipts (from SROIE); 163 of them were forged by people using Paint, GIMP and similar tools,
and the edited regions are annotated with boxes. We tuned only on the official train/val
splits and ran the **test split once**, on the frozen pipeline:

```
python -m phantom.mirage.eval_findit --split test --json-out docs/mirage_eval_findit_test.json
```

| Split | Receipts (forged) | Precision | Recall | F1 | False-positive rate | Flagged forgeries localised correctly |
|---|---|---|---|---|---|---|
| val, Week 2/3 pipeline before receipt work | 192 (33) | 0.14 | 0.64 | 0.23 | 0.80 | 5 / 21 |
| train, final (tuning) | 577 (94) | 0.93 | 0.59 | 0.72 | 0.01 | 55 / 55 |
| val, final (tuning) | 192 (33) | 0.93 | 0.42 | 0.58 | 0.01 | 14 / 14 |
| **test, final (untuned, run once)** | **218 (35)** | **1.00** | **0.69** | **0.81** | **0.00** | **24 / 24** |

- **0 false alarms on 183 genuine test receipts**, including the 98 with handwritten notes.
  (The old layout checks raised false alarms on 75% of handwritten val receipts.)
- Every flagged forgery was flagged at the place the forger edited.
- The test set is small: 35 forgeries, so recall 0.69 = 24/35. Treat the numbers as
  ±~0.15, not as exact.
- One labelled val image (`X51006619709.png`) is missing from the published zip and is excluded.

### What made the difference

1. **New `copy_move` check** (`phantom/mirage/copy_move.py`). Most real forgeries (`CPI`,
   copy-paste inside the same image) clone a digit from elsewhere on the receipt. Printed and
   scanned characters never match pixel-for-pixel because each has its own noise, but a pasted
   copy does. The check compares character-sized blobs at the original resolution. On train,
   forged clones measured ≤ 0.15 grey levels mean difference and the closest genuine pairs
   ≥ 0.56, so the threshold is 0.3. Born-digital images (no scan noise) are detected and
   skipped, not flagged.
2. **Routing by document type.** On real receipts the invoice checks (arithmetic, alignment,
   baseline, font, background) fired on genuine receipts about 10× more often than on forged
   ones. The causes were OCR misreads ("471.00" for "41.00"), mixed fonts and handwriting. Pages
   with till-receipt vocabulary (CASH, CHANGE, ROUNDING, CASHIER…) are now judged only by
   checks validated on receipts. The others are still reported as reviewer notes.
3. **Corroboration rule.** A layout or visual hint only becomes a finding when a *different*
   check flags the same field. Lone hints are listed as notes.
4. **Unreadable arithmetic is a note, not a flag.** Failing to read a number is not evidence of
   tampering.

### What it still misses (11 of 35 test forgeries)

Forgeries that leave no copy-paste trace: characters painted pixel by pixel (`PIX`),
imitation text typed in a similar font (`IMI`), cut/erase (`CUT`), and pastes from another
document (`CPO`). Some copy-pastes also blend into their neighbours and miss the clone
threshold. Next steps for these:
- Error Level Analysis / JPEG-grid consistency for pasted-from-elsewhere regions.
- A near-clone search tolerant to partial overlap.
- A small trained detector on the train split (e.g. a lightweight CNN on character patches),
  if the team wants a learned component.

### Trade-off on our synthetic set

The corroboration rule lowered recall on our synthetic held-out invoices. That set's hardest
attacks leave exactly one weak visual trace.

| Synthetic hard held-out (60 docs) | Precision | Recall | F1 | FP rate |
|---|---|---|---|---|
| First Week 3 version | 1.00 | 0.67 | 0.80 | 0.00 |
| Final pipeline | 1.00 | 0.50 | 0.67 | 0.00 |

We chose the final version because real-data precision matters more than synthetic recall: a
tool that flags 80% of genuine receipts is unusable. Both numbers are reported.

---

## Synthetic invoice evaluation (Week 3, first version)

Document-level prediction: "suspicious" if any content check (not input-quality
notes) is suspicious or inconclusive. Inconclusive counts as flagged, since the
goal is to route documents to a human reviewer, not to pass them silently.

```
python -m phantom.mirage.make_hard_samples --out data/dev_hard --n 20 --seed 11
python -m phantom.mirage.make_hard_samples --out data/holdout_hard --n 30 --seed 2026
python -m phantom.mirage.evaluate --samples data/holdout_hard --json-out docs/mirage_eval_holdout_hard.json
```

## Week 3 results

| Set | Docs | Precision | Recall | F1 | FP rate | Runtime / doc |
|---|---|---|---|---|---|---|
| Day 6 synthetic set (`data/samples`), Week 2 pipeline (as reported) | 40 | 0.67 | 0.60 | 0.63 | 0.30 | 0.30 s |
| Day 6 synthetic set, Week 2 pipeline, re-run on OpenCV 5.1 | 40 | 0.56 | 0.45 | 0.50 | 0.35 | 0.58 s |
| Day 6 synthetic set, Week 3 pipeline (**tuning set**) | 40 | 1.00 | 1.00 | 1.00 | 0.00 | 0.42 s |
| Hard dev set (`data/dev_hard`, seed 11, **tuning set**) | 40 | 1.00 | 0.85 | 0.92 | 0.00 | 0.60 s |
| **Hard held-out set (`data/holdout_hard`, seed 2026, run once, untuned)** | **60** | **1.00** | **0.67** | **0.80** | **0.00** | **0.63 s** |

v1 target (SCOPE.md): F1 ≥ 0.70, runtime ≤ 5 s/page. **Met on the held-out set.**
Full per-document results are recorded in `docs/mirage_eval_*.json`. Measured on
Windows 11, Python 3.14, OpenCV 5.1, Tesseract 5.4.0.

The 1.00 on the Day 6 set is not a meaningful accuracy claim. Every manipulation
in that set changes one amount without touching the total, so the new
arithmetic check catches all of them, and thresholds were tuned on it. The
held-out row is the number to quote.

### Held-out recall by attack type

| Attack | Arithmetic still adds up? | Caught |
|---|---|---|
| altered_amount | no | 5 / 5 |
| pasted_patch (amount pasted from another scan) | no | 6 / 6 |
| font_mismatch | no | 4 / 4 |
| font_mismatch / misaligned_field, edited **after** scanning | yes | 2 / 2 |
| consistent_edit (item **and** totals rewritten), edited after scanning | yes | 2 / 4 |
| font_mismatch / misaligned_field, edited **before** scanning | yes | 1 / 4 |
| consistent_edit, edited before scanning | yes | 0 / 5 |

**Summary:** MIRAGE catches every edit that breaks the invoice's arithmetic
(15/15), and 5/15 of edits that keep it consistent. When a forger also fixes the
totals *and* the document is printed and re-scanned afterwards, the scan
re-textures the whole page and erases most visual traces. This is the main
known limitation.

## What changed in Week 3 and why

1. **Ingestion bug: pages rotated 90°.** `estimate_skew` assumed OpenCV's old
   `minAreaRect` angle convention, and under OpenCV ≥ 4.5 an upright page came
   back as −90°. `requirements.txt` doesn't pin OpenCV, so any fresh install
   OCR'd every page sideways, and every geometric check ran on rotated text.
   This is why the Week 2 numbers drop to F1 0.50 on a fresh install. Replaced
   with a projection-profile skew search, which also fixed ~0.5° of residual
   skew on noisy scans. A regression test was added.
2. **New `numeric_consistency` check** (`phantom/mirage/consistency.py`). Line
   items must add up to SUBTOTAL/TOTAL, and SUBTOTAL + tax must equal TOTAL. If
   an amount it needs is unreadable or read with low OCR confidence, it returns
   *inconclusive* instead of accusing the document.
3. **Checks now compare amount fields only** (`phantom/mirage/fields.py`).
   Previously any token containing a digit (`INV-5561`, the `0` in "TEMPLATE 0",
   a hyphen) entered the statistics. That was the main false-positive source.
4. **`font_consistency` uses stroke width** (2 × ink area / contour length)
   instead of ink density. Ink density mostly measured *which* characters a
   word contained.
5. **New `row_baseline` check.** An amount must share a baseline with its row
   label, relative to the page's typical offset so residual tilt cancels out.
6. **`region_inconsistency` measures background level and fine texture** (a
   high-pass residual) around each amount. This catches whiteout and pasted
   patches on scanned pages. It stays quiet on clean digital pages by design.
7. **`nearly_blank` quality flag** now means "almost no ink" rather than "mean
   brightness > 250", which flagged every normal invoice.

## Known limitations

- All evaluation data is still synthetic. The hard set imitates scanning
  (paper tone, noise, skew, blur, JPEG), but real scanned or photographed
  forgeries are the next test before relying on these numbers.
- Arithmetic-consistent edits made before printing and scanning are mostly
  undetected (1/9 on held-out).
- The checks are tuned to invoice-like tables: a labelled row per amount and a
  TOTAL row. Other document types get only the visual checks.
- Confidence scores are module-local heuristics and uncalibrated.

## Next steps

- Combine weak indicators on the same field. One dev-set miss had both
  thinner strokes (−14%) and a cleaner background (0.5×), each under its own
  threshold.
- Error Level Analysis on the original (pre-normalisation) image for
  after-scan edits that were re-saved as JPEG.
- Test on real scanned forgeries.
