"""FastAPI app (Day 12). MIRAGE is a complete pipeline; ECHO and BLACKBOX are mocks."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from phantom.common.evidence_store import EvidenceStore
from phantom.common.schemas import ModuleName, ToolError, ToolResult
from phantom.echo.ingest import IngestError as EchoIngestError
from phantom.echo.ingest import ingest as echo_ingest
from phantom.mirage.analyze import analyze_document, overall_outcome

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
app = FastAPI(title="PHANTOM AI", version="0.3.0")
REPO_ROOT = Path(__file__).resolve().parents[2]
store = EvidenceStore(REPO_ROOT / "evidence_store")
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEMO_DIR = REPO_ROOT / "data" / "demo"
EVAL_FILES = {
    "holdout_hard": REPO_ROOT / "docs" / "mirage_eval_holdout_hard.json",
    "holdout_hard_final": REPO_ROOT / "docs" / "mirage_eval_holdout_hard_final.json",
    "dev_hard": REPO_ROOT / "docs" / "mirage_eval_dev_hard.json",
    "samples": REPO_ROOT / "docs" / "mirage_eval_samples.json",
    "findit_val_baseline": REPO_ROOT / "docs" / "mirage_eval_findit_val_baseline.json",
    "findit_val": REPO_ROOT / "docs" / "mirage_eval_findit_val.json",
    "findit_test": REPO_ROOT / "docs" / "mirage_eval_findit_test.json",
}
# Real receipts are third-party data (Find it again!): never committed, served only
# from a local download (data/raw/findit2), listed by docs/findit_demo_cases.json.
REAL_DIR = REPO_ROOT / "data" / "raw" / "findit2" / "test"
REAL_MANIFEST = REPO_ROOT / "docs" / "findit_demo_cases.json"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_ui() -> HTMLResponse:
    """MIRAGE investigation demo (single page, no build step)."""
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/demo/cases")
def demo_cases() -> dict:
    manifest = DEMO_DIR / "manifest.json"
    if not manifest.exists():
        return {"cases": []}
    return json.loads(manifest.read_text(encoding="utf-8"))


@app.get("/demo/cases/{name}")
def demo_case_file(name: str) -> FileResponse:
    path = (DEMO_DIR / name).resolve()
    if path.parent != DEMO_DIR.resolve() or not path.is_file():
        raise HTTPException(404, "demo case not found")
    return FileResponse(path)


@app.get("/demo/real-cases")
def demo_real_cases() -> dict:
    if not REAL_MANIFEST.exists():
        return {"cases": [], "note": "No real-receipt demo cases configured."}
    manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    cases = [c for c in manifest.get("cases", []) if (REAL_DIR / c["file"]).is_file()]
    note = None if cases else (
        "Real receipts are not on this machine. Download Find it again! into data/raw/findit2 "
        "(see phantom/mirage/eval_findit.py) to enable these cases."
    )
    return {**manifest, "cases": cases, "note": note}


@app.get("/demo/real-cases/{name}")
def demo_real_case_file(name: str) -> FileResponse:
    allowed = {c["file"] for c in demo_real_cases()["cases"]}
    path = (REAL_DIR / name).resolve()
    if name not in allowed or path.parent != REAL_DIR.resolve() or not path.is_file():
        raise HTTPException(404, "case not found")
    return FileResponse(path)


@app.get("/mirage/eval")
def mirage_eval() -> dict:
    """Recorded evaluation results (see docs/MIRAGE_EVAL.md), without per-row detail."""
    out = {}
    for name, path in EVAL_FILES.items():
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            out[name] = {k: v for k, v in data.items() if k != "rows"}
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modules": [m.value for m in ModuleName]}


@app.post("/mirage/analyze", response_model=ToolResult)
async def mirage_analyze(file: UploadFile = File(...)) -> ToolResult:
    """Full MIRAGE pipeline: ingest, OCR, visual checks, evidence-linked findings.

    See docs/MIRAGE_EVAL.md for accuracy numbers and known limitations.
    """
    data = await file.read()
    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
    try:
        return analyze_document(tmp.name, store, source=file.filename or "upload")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/mirage/verdict")
def mirage_verdict(result: ToolResult) -> dict:
    """Document-level verdict for a MIRAGE ToolResult (same rule as the evaluation)."""
    return {"verdict": overall_outcome(result).value}


def _mock(module: ModuleName) -> ToolResult:
    return ToolResult(
        module=module,
        limitations=["mock endpoint: module not implemented yet"],
        needs_human_review=True,
    )


@app.post("/echo/analyze", response_model=ToolResult)
async def echo_analyze(file: UploadFile = File(...)) -> ToolResult:
    """Audio is validated and normalized; claim extraction is not built yet
    (Week 3), so this always reports needs_human_review with no findings."""
    data = await file.read()
    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
    result = ToolResult(module=ModuleName.ECHO)
    try:
        audio = echo_ingest(tmp.name)
        evidence_id = store.put(data, kind="audio", source=file.filename or "upload")
        result.evidence_ids.append(evidence_id)
        if audio.quality.issues:
            result.limitations.append(f"audio quality issues: {', '.join(audio.quality.issues)}")
    except EchoIngestError as e:
        result.errors.append(ToolError(code="ingest_failed", message=str(e)))
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    result.limitations.append("claim extraction not implemented yet: ingestion/validation only")
    return result


@app.post("/blackbox/investigate", response_model=ToolResult)
def blackbox_investigate() -> ToolResult:
    return _mock(ModuleName.BLACKBOX)


@app.get("/evidence/{evidence_id}")
def evidence(evidence_id: str) -> dict:
    try:
        _, record = store.get(evidence_id)
    except KeyError as e:
        raise HTTPException(404, "evidence not found") from e
    return {**record, "integrity_ok": store.verify(evidence_id)}


@app.get("/evidence/{evidence_id}/image")
def evidence_image(evidence_id: str) -> Response:
    """Render a stored page_region evidence item (the normalized page MIRAGE
    analysed) as PNG, so finding bboxes can be drawn on exactly what was checked."""
    try:
        blob, record = store.get(evidence_id)
    except KeyError as e:
        raise HTTPException(404, "evidence not found") from e
    shape = record.get("meta", {}).get("shape")
    if record.get("kind") != "page_region" or not shape:
        raise HTTPException(400, "evidence item is not a rendered page")
    img = np.frombuffer(blob, dtype=np.uint8).reshape(shape)
    ok, png = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "could not encode page image")
    return Response(png.tobytes(), media_type="image/png")
