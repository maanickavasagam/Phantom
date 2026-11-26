"""FastAPI app for MIRAGE, ECHO, BLACKBOX, and the bounded coordinator."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from phantom.blackbox.investigate import SCENARIOS, evaluate, investigate, scenario_html
from phantom.common.evidence_store import EvidenceStore
from phantom.common.schemas import ModuleName, ToolResult
from phantom.coordinator.engine import coordinate
from phantom.echo.analyze import analyze_audio
from phantom.echo.analyze import overall_outcome as echo_overall_outcome
from phantom.mirage.analyze import analyze_document, overall_outcome

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
app = FastAPI(title="PHANTOM AI", version="0.3.0")
REPO_ROOT = Path(__file__).resolve().parents[2]
store = EvidenceStore(REPO_ROOT / "evidence_store")
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEMO_DIR = REPO_ROOT / "data" / "demo"
ECHO_DEMO_DIR = REPO_ROOT / "data" / "demo_echo"
EVAL_FILES = {
    "holdout_hard": REPO_ROOT / "docs" / "mirage_eval_holdout_hard.json",
    "holdout_hard_final": REPO_ROOT / "docs" / "mirage_eval_holdout_hard_final.json",
    "dev_hard": REPO_ROOT / "docs" / "mirage_eval_dev_hard.json",
    "samples": REPO_ROOT / "docs" / "mirage_eval_samples.json",
    "findit_val_baseline": REPO_ROOT / "docs" / "mirage_eval_findit_val_baseline.json",
    "findit_val": REPO_ROOT / "docs" / "mirage_eval_findit_val.json",
    "findit_test": REPO_ROOT / "docs" / "mirage_eval_findit_test.json",
    "echo_calls_dev": REPO_ROOT / "docs" / "echo_eval_calls_dev.json",
    "echo_calls_test": REPO_ROOT / "docs" / "echo_eval_calls_test.json",
    "echo_librispeech": REPO_ROOT / "docs" / "echo_eval_librispeech.json",
    "echo_librispeech_other": REPO_ROOT / "docs" / "echo_eval_librispeech_other.json",
    "echo_public": REPO_ROOT / "docs" / "echo_eval_public.json",
    "echo_public_frozen": REPO_ROOT / "docs" / "echo_eval_public_frozen.json",
    "echo_hybrid": REPO_ROOT / "docs" / "echo_eval_hybrid.json",
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
            if isinstance(data, list):  # e.g. one entry per public dataset
                out[name] = [{k: v for k, v in d.items() if k not in ("rows", "examples")}
                             for d in data]
            else:
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


@app.post("/echo/analyze", response_model=ToolResult)
async def echo_analyze(file: UploadFile = File(...)) -> ToolResult:
    """Full ECHO pipeline: validate audio, transcribe locally (Whisper), extract
    claims, detect social-engineering indicators. Transcript and claims are in
    result.artifacts; see docs/ECHO_EVAL.md for accuracy and limitations."""
    data = await file.read()
    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
    try:
        return analyze_audio(tmp.name, store, source=file.filename or "upload")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/echo/verdict")
def echo_verdict(result: ToolResult) -> dict:
    return {"verdict": echo_overall_outcome(result).value}


@app.get("/demo/echo-cases")
def demo_echo_cases() -> dict:
    manifest = ECHO_DEMO_DIR / "manifest.json"
    return json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {"cases": []}


@app.get("/demo/echo-cases/{name}")
def demo_echo_case_file(name: str) -> FileResponse:
    path = (ECHO_DEMO_DIR / name).resolve()
    if path.parent != ECHO_DEMO_DIR.resolve() or not path.is_file():
        raise HTTPException(404, "demo case not found")
    return FileResponse(path)


@app.get("/evidence/{evidence_id}/audio")
def evidence_audio(evidence_id: str) -> Response:
    """Stream a stored audio evidence item, so the UI plays exactly what was analysed."""
    try:
        blob, record = store.get(evidence_id)
    except KeyError as e:
        raise HTTPException(404, "evidence not found") from e
    if record.get("kind") != "audio":
        raise HTTPException(400, "evidence item is not audio")
    ext = Path(record.get("source", "")).suffix.lower()
    media = {".mp3": "audio/mpeg", ".flac": "audio/flac", ".m4a": "audio/mp4",
             ".mp4": "audio/mp4", ".aac": "audio/aac", ".ogg": "audio/ogg",
             ".opus": "audio/ogg", ".webm": "audio/webm", ".amr": "audio/amr",
             ".3gp": "audio/3gpp", ".wma": "audio/x-ms-wma"}.get(ext, "audio/wav")
    return Response(blob, media_type=media)


class BlackboxRequest(BaseModel):
    scenario: str = "fees"
    strategy: str = "learned"
    action_budget: int = Field(default=2, ge=1, le=10)


@app.get("/synthetic/{scenario}/{page}", response_class=HTMLResponse)
def synthetic_page(scenario: str, page: str) -> HTMLResponse:
    try:
        return HTMLResponse(scenario_html(scenario, page))
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@app.get("/blackbox/scenarios")
def blackbox_scenarios() -> dict:
    return {"scenarios": list(SCENARIOS)}


@app.post("/blackbox/investigate", response_model=ToolResult)
def blackbox_investigate(options: BlackboxRequest, request: Request) -> ToolResult:
    if options.scenario not in SCENARIOS:
        raise HTTPException(400, "unknown synthetic scenario")
    if options.strategy not in {"fixed", "random", "learned"}:
        raise HTTPException(400, "unknown strategy")
    url = str(request.base_url).rstrip("/") + f"/synthetic/{options.scenario}/start"
    try:
        return investigate(url, store, options.strategy, options.action_budget)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(503, str(error)) from error


@app.get("/blackbox/eval")
def blackbox_eval(request: Request) -> dict:
    url = str(request.base_url).rstrip("/")
    return evaluate(url, store)


class CoordinatorRequest(BaseModel):
    goal: str
    document_evidence_id: str | None = None
    audio_evidence_id: str | None = None
    scenario: str | None = None
    max_calls: int = Field(default=3, ge=1, le=3)


@app.post("/coordinator/investigate")
def coordinator_investigate(options: CoordinatorRequest, request: Request) -> dict:
    if options.scenario is not None and options.scenario not in SCENARIOS:
        raise HTTPException(400, "unknown synthetic scenario")
    url = (
        str(request.base_url).rstrip("/") + f"/synthetic/{options.scenario}/start"
        if options.scenario else None
    )
    return coordinate(
        options.goal, store, options.document_evidence_id,
        options.audio_evidence_id, url, options.max_calls
    )


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
