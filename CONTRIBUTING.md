# Contributing to PHANTOM AI

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# OCR engine: install Tesseract (apt install tesseract-ocr / brew install tesseract / Windows installer)
pytest
uvicorn phantom.api.main:app --reload    # http://127.0.0.1:8000/docs
```

## Branches and PRs
- `main` is protected: no direct pushes.
- Branch names: `feat/<module>-<short-name>`, `fix/<module>-<short-name>`, `docs/<topic>`.
- Every PR needs: a linked issue, passing CI (ruff + pytest), and one approving review from another member.
- Squash-merge; the commit title follows `<module>: <what changed>` (e.g. `mirage: add rotation correction`).

## Code style
- Python 3.11+, formatted and linted with `ruff` (`ruff check . && ruff format .`).
- Every module returns a `ToolResult` (see `phantom/common/schemas.py`); never return raw dicts.
- No live network targets in tests or code paths; BLACKBOX targets must be on the allowlist.

## Issue board columns
Backlog → This week → In progress → In review → Done
