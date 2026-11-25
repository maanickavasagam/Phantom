# PHANTOM AI

Multimodal investigation prototype: **MIRAGE** (document forensics), **ECHO** (voice incident
intelligence), **BLACKBOX** (controlled browser investigation with a learned action ranker), joined by
a bounded agentic coordinator. Outputs are evidence-linked and always flagged for human review.

## Quick start
System dependencies:
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for MIRAGE. On Windows it's
  found automatically at `C:\Program Files\Tesseract-OCR`; otherwise put it on PATH or set
  `TESSERACT_CMD`.
- ffmpeg for ECHO. Put it on PATH or set `FFMPEG_BINARY`.
- ECHO downloads the Whisper `base.en` model (~145 MB) on first use. Set `ECHO_MODEL_DIR` to
  choose where it's cached, and `ECHO_WHISPER_MODEL` to pick another model (e.g. `small.en`).

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

## ECHO: call analysis
Open the **ECHO · Calls** page in the demo. You can pick one of 7 scripted calls
(`data/demo_echo/`) or upload a WAV/MP3/FLAC/M4A. ECHO transcribes it locally with Whisper,
extracts amounts, deadlines, payment and credential requests, and who the caller claims to
be. It then flags strong indicators (credential requests, gift cards / crypto / remote-access
apps) and combinations of pressure tactics. Claims are marked on the audio timeline, and
every finding links to its timestamp.

**Public benchmarks.**
- Scam detection on independent scam-dialogue datasets ([BothBosu](https://huggingface.co/BothBosu),
  official test halves, rules tuned on train halves only): **F1 0.89 and 0.94**.
- Real scam calls from scam-baiter videos: 6/20 with rules alone.
- A learned classifier (trained on the public train halves only) can only accuse when rule
  evidence backs it up; on its own it flags a call for human review. Together they reach
  F1 0.95 / 0.99 on the test halves, catch 12/20 real scam calls, and flag all 20 for review,
  with no new false alarms on our own calls. Details: `docs/ECHO_EVAL.md` §2b.
- Speech recognition: **WER 5.3% on LibriSpeech test-clean and 10.7% on test-other**.

Held-out scripted calls: **claim F1 0.85 on clean audio and 0.80 with noise or over a simulated
phone line; all scam calls caught with 0 false alarms in every condition**. Speech recognition:
**WER 5.3% on LibriSpeech test-clean** (target ≤ 15%). The phone-line result comes from a
second test run after a dev-only VAD fix; both runs are in `docs/ECHO_EVAL.md`.
```bash
python -m phantom.echo.make_calls --out data/echo_calls --fills 3
python -m phantom.echo.evaluate calls --split test
python -m phantom.echo.evaluate librispeech --root data/raw/LibriSpeech/test-clean --n 200
```
See `docs/ECHO_EVAL.md` for the per-condition table, label audit and limitations.

## Layout
```
docs/            SCOPE.md, DATASET_AUDIT.md, MIRAGE_EVAL.md, ECHO_SCOPE.md, ECHO_EVAL.md
schemas/         tool_result.schema.json (generated from phantom/common/schemas.py)
phantom/common/  shared ToolResult contract + evidence store
phantom/mirage/  ingestion, OCR, field extraction, numeric consistency check, visual checks,
                 analyze pipeline, evaluation, sample generators and report renderer
phantom/echo/    ingestion, Whisper transcription, claim extraction, indicators, analyze
                 pipeline, scripted-call generator, evaluation (WER, claim F1, detection)
phantom/api/static/  single-page demo UI for MIRAGE + ECHO + evaluation (served at /)
data/demo/       7 synthetic invoice demo cases + manifest with ground truth
data/demo_echo/  7 scripted demo calls + manifest with ground truth
phantom/blackbox/(Phase 4)
phantom/api/     FastAPI app: /health, /mirage/analyze, /echo/analyze (full pipelines),
                 /blackbox/investigate (mock), /evidence/{id}[/image|/audio]
tests/           pytest suite
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
