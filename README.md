# PHANTOM AI

Multimodal investigation prototype: **MIRAGE** (document forensics), **ECHO** (voice incident
intelligence), **BLACKBOX** (controlled browser investigation with a learned action ranker), joined by
a bounded agentic coordinator. Outputs are evidence-linked and always flagged for human review.

## Quick start
System dependencies: [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (MIRAGE)
and ffmpeg (ECHO). On Windows, Tesseract is found automatically at
`C:\Program Files\Tesseract-OCR`; otherwise put it on PATH or set `TESSERACT_CMD`.

```bash
pip install -r requirements.txt
pytest -q
uvicorn phantom.api.main:app --reload        # demo UI: http://127.0.0.1:8000/  · API docs: /docs
```

## MIRAGE demo
Open http://127.0.0.1:8000/ after starting the API. The case library has six real receipts
from the Find it again! test split (shown only if the dataset is downloaded locally) and seven
synthetic invoices (`data/demo/`). You can also upload your own PNG/JPG/TIFF/PDF. The page shows the verdict, every finding
drawn on the analysed page, evidence IDs with integrity checks, and limitations. For demo
cases it also shows the planted edit (ground truth). The **Evaluation** tab shows the recorded
held-out numbers.

## MIRAGE evaluation
**Real forged receipts** ([Find it again!](https://l3i-share.univ-lr.fr/2023Finditagain/), ICDAR 2023,
test split, run once): **precision 1.00, recall 0.69, F1 0.81, 0 false alarms on 183 genuine
receipts**. Download the dataset into `data/raw/findit2/`, then:
```bash
python -m phantom.mirage.eval_findit --split test
```
Synthetic invoices:
```bash
python -m phantom.mirage.make_hard_samples --out data/holdout_hard --n 30 --seed 2026
python -m phantom.mirage.evaluate --samples data/holdout_hard   # precision/recall/F1 + recall by attack
python -m phantom.mirage.sample_report data/demo/03_font_swap.png
```
Synthetic held-out with the final pipeline: precision 1.00, recall 0.50, F1 0.67, FP rate 0.00.
See `docs/MIRAGE_EVAL.md` for both benchmarks, the per-attack breakdown and limitations.

## Layout
```
docs/            SCOPE.md, DATASET_AUDIT.md, MIRAGE_EVAL.md
schemas/         tool_result.schema.json (generated from phantom/common/schemas.py)
phantom/common/  shared ToolResult contract + evidence store
phantom/mirage/  ingestion, OCR, field extraction, numeric consistency check, visual checks,
                 analyze pipeline, evaluation, sample generators and report renderer
phantom/api/static/  single-page MIRAGE demo UI (served at /)
data/demo/       7 held-out demo cases + manifest with ground truth
phantom/echo/    scope defined (docs/ECHO_SCOPE.md); audio ingestion live (Phase 3)
phantom/blackbox/(Phase 4)
phantom/api/     FastAPI app: /health, /mirage/analyze (full pipeline),
                 /echo/analyze (mock), /blackbox/investigate (mock), /evidence/{id}
tests/           pytest suite (52 tests)
```

See `docs/MIRAGE_EVAL.md` for MIRAGE's current accuracy numbers and known
limitations — reported as measured, not tuned to hit the v1 target.

## Team
| Member | Primary module |
|---|---|
| Maanickavasagam | MIRAGE |
| Aadhityan | ECHO |
| Aswin Karthik | BLACKBOX |

See `CONTRIBUTING.md` for branch and PR rules.
