"""Goal routing, tool execution, evidence trace, and unified human-review report."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from phantom.blackbox.investigate import investigate
from phantom.common.evidence_store import EvidenceStore
from phantom.common.schemas import ToolResult
from phantom.echo.analyze import analyze_audio
from phantom.mirage.analyze import analyze_document


def coordinate(
    goal: str,
    store: EvidenceStore,
    document_id: str | None = None,
    audio_id: str | None = None,
    target_url: str | None = None,
    max_calls: int = 3,
    max_seconds: float = 120,
) -> dict:
    if not 1 <= max_calls <= 3 or not 1 <= max_seconds <= 600:
        raise ValueError("invalid coordinator budget")
    start = time.monotonic()
    results: list[ToolResult] = []
    trace: list[dict] = []
    errors: list[dict] = []
    attempts = 0
    goal_text = goal.lower()
    requested = [
        ("mirage", document_id, ("document", "invoice", "receipt", "image", "pdf")),
        ("echo", audio_id, ("audio", "voice", "call", "speech")),
        ("blackbox", target_url, ("web", "website", "browser", "site")),
    ]
    for name, supplied, keywords in requested:
        relevant = any(word in goal_text for word in keywords) or goal_text.strip() in {
            "investigate", "all"
        }
        if not relevant or not supplied:
            continue
        if attempts >= max_calls:
            trace.append({"module": name, "status": "skipped", "reason": "call budget"})
            continue
        if time.monotonic() - start >= max_seconds:
            trace.append({"module": name, "status": "skipped", "reason": "time budget"})
            break
        attempts += 1
        try:
            if name == "blackbox":
                result = investigate(str(supplied), store, action_budget=2)
            else:
                blob, record = store.get(str(supplied))
                suffix = Path(record["source"]).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                    temporary.write(blob)
                    path = Path(temporary.name)
                try:
                    result = (
                        analyze_document(path, store, source=record["source"])
                        if name == "mirage"
                        else analyze_audio(path, store, source=record["source"])
                    )
                finally:
                    path.unlink(missing_ok=True)
            results.append(result)
            trace.append({"module": name, "status": "completed", "run_id": result.run_id})
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            trace.append({"module": name, "status": "failed", "error": str(error)})
            errors.append({"module": name, "code": "tool_failed", "message": str(error)})
    missing = [
        name for name, supplied, words in requested
        if any(word in goal_text for word in words) and not supplied
    ]
    if not results and not trace:
        missing.append("no relevant supplied input")
    findings = [
        {"module": result.module.value, **finding.model_dump(mode="json")}
        for result in results for finding in result.findings
    ]
    return {
        "goal": goal,
        "needs_human_review": True,
        "findings": findings,
        "evidence_ids": sorted({eid for result in results for eid in result.evidence_ids}),
        "results": [result.model_dump(mode="json") for result in results],
        "trace": trace,
        "missing_inputs": missing,
        "limitations": [
            note for result in results for note in result.limitations
        ],
        "errors": [
            {"module": result.module.value, **error.model_dump()}
            for result in results for error in result.errors
        ] + errors,
        "tool_calls": attempts,
        "elapsed_s": round(time.monotonic() - start, 3),
    }
