"""FastAPI app (Day 12). MIRAGE is a complete pipeline; ECHO and BLACKBOX are mocks."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from phantom.common.evidence_store import EvidenceStore
from phantom.common.schemas import ModuleName, ToolError, ToolResult
from phantom.echo.ingest import IngestError as EchoIngestError
from phantom.echo.ingest import ingest as echo_ingest
from phantom.mirage.analyze import analyze_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
app = FastAPI(title="PHANTOM AI", version="0.2.0")
store = EvidenceStore()


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
