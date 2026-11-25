# PHANTOM AI

Multimodal investigation prototype: **MIRAGE** (document forensics), **ECHO** (voice incident
intelligence), **BLACKBOX** (controlled browser investigation with a learned action ranker), joined by
a bounded agentic coordinator. Outputs are evidence-linked and always flagged for human review.

## Quick start
```bash
pip install -r requirements.txt
pytest -q
python -m phantom.mirage.make_samples --out data/samples --n 40
python -m phantom.mirage.evaluate --samples data/samples    # precision/recall/F1
python -m phantom.mirage.sample_report data/samples/inv_000_suspicious.png
uvicorn phantom.api.main:app --reload        # open http://127.0.0.1:8000/docs
```

## Layout
```
docs/            SCOPE.md, DATASET_AUDIT.md, MIRAGE_EVAL.md
schemas/         tool_result.schema.json (generated from phantom/common/schemas.py)
phantom/common/  shared ToolResult contract + evidence store
phantom/mirage/  ingestion, OCR, visual checks, analyze pipeline, evaluation,
                 sample generator and report renderer
phantom/echo/    scope defined (docs/ECHO_SCOPE.md); audio ingestion live (Phase 3)
phantom/blackbox/(Phase 4)
phantom/api/     FastAPI app: /health, /mirage/analyze (full pipeline),
                 /echo/analyze (mock), /blackbox/investigate (mock), /evidence/{id}
tests/           pytest suite (30 tests)
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
