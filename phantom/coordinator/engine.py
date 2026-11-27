"""Goal routing, tool execution, evidence trace, and unified human-review report.

Safeguards (docs/COORDINATOR.md):

1. Malformed calls: the goal, budgets, evidence IDs (format, existence, kind,
   SHA-256 integrity) and URL are validated before any tool runs; a bad input
   is reported as an error for that module, never guessed at.
2. Tool failures: any exception, or a result that does not satisfy the
   ToolResult contract, is caught, reported and counted against the budget.
   Two failures in a row stop the run (circuit breaker).
3. Timeouts: each tool runs under a hard per-tool timeout (capped by the time
   left in the overall budget). A tool that overruns is abandoned and its
   result discarded, even if it finishes later.
4. Prompt injection: routing is decided ONLY from the user's goal and the
   inputs they supplied, before any tool runs. Text produced by tools (OCR,
   transcripts, page text) is data: it is scanned for instruction-like phrases
   and flagged as a warning, but it can never add, remove or reorder calls or
   change a verdict.
5. Repeated actions: each (module, input) pair runs at most once per run, and
   the same evidence item cannot be submitted as both document and audio.
6. Budgets: at most 3 calls and max_seconds overall; the report says why the
   run stopped (status / stop_reason).
"""

from __future__ import annotations

import re
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from phantom.blackbox.investigate import investigate
from phantom.blackbox.urltool import MAX_URL_LENGTH, check_url_tool
from phantom.common.evidence_store import EvidenceStore
from phantom.common.schemas import ToolResult
from phantom.echo.analyze import analyze_audio
from phantom.echo.analyze import overall_outcome as echo_outcome
from phantom.mirage.analyze import analyze_document
from phantom.mirage.analyze import overall_outcome as mirage_outcome

MAX_CALLS = 3
MAX_GOAL_LENGTH = 2000
DEFAULT_TOOL_TIMEOUT = 90.0
MAX_CONSECUTIVE_FAILURES = 2
EVIDENCE_ID = re.compile(r"^ev_[0-9a-f]{16}$")
EXPECTED_KIND = {"mirage": "document", "echo": "audio"}
KEYWORDS = {
    "mirage": ("document", "invoice", "receipt", "image", "pdf", "bill"),
    "echo": ("audio", "voice", "call", "speech", "recording"),
    "blackbox": ("web", "website", "browser", "site", "url", "link", "domain"),
}
def _asks_for(goal_text: str, module: str) -> bool:
    """Whole-word (prefix) keyword match: "calls" counts, "invoice" is not "voice"."""
    return any(re.search(r"\b" + re.escape(w), goal_text) for w in KEYWORDS[module])


ALL_GOALS = {"investigate", "all", "everything", "investigate all"}

# Instruction-like phrases that have no business inside a receipt, a call
# transcript or a web address. Matching never changes what the coordinator does;
# it only warns the reviewer that the content tries to address an AI system.
INJECTION_PATTERNS = [
    (re.compile(p, re.I), label) for p, label in (
        (r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier|all)\b.{0,20}"
         r"\b(instructions?|rules|prompts?|messages?)\b", "override_instructions"),
        (r"\b(system|developer)\s*(prompt|message|instructions?)\b", "mentions_system_prompt"),
        (r"\byou are (now )?(an?|the) (ai|assistant|model|agent)\b", "role_reassignment"),
        (r"\b(mark|classify|label|report|treat)\b.{0,30}\b(this|it|document|call|site)\b.{0,30}"
         r"\b(as )?(benign|safe|genuine|legitimate|official|not (a )?scam)\b", "verdict_request"),
        (r"\b(call|run|invoke|use|execute)\b.{0,20}\b(tool|function|mirage|echo|blackbox|"
         r"coordinator)\b", "tool_request"),
        # "flag"/"escalate" only: scammers tell VICTIMS "do not tell anyone / don't
        # report it" (a scam sign ECHO already reports, not an instruction to an AI;
        # found as a false alarm in the end-to-end evaluation)
        (r"\b(do not|don't|never)\b.{0,20}\b(flag|escalate)\b", "suppression_request"),
        (r"<\s*/?\s*(system|assistant|instructions?)\s*>|\[\s*(system|inst)\s*\]",
         "fake_prompt_markup"),
    )
]


class CoordinatorInputError(ValueError):
    """The coordinator call itself is malformed (bad goal or budget)."""


def scan_for_injection(text: str) -> list[str]:
    """Labels of instruction-like phrases found in tool-produced text."""
    return [label for pattern, label in INJECTION_PATTERNS if pattern.search(text)]


def _texts(obj, limit: int = 200_000) -> str:
    """All strings inside a tool result (findings, artifacts), joined and capped."""
    out: list[str] = []
    size = 0

    def walk(o):
        nonlocal size
        if size > limit:
            return
        if isinstance(o, str):
            out.append(o)
            size += len(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list | tuple):
            for v in o:
                walk(v)

    walk(obj)
    return "\n".join(out)[:limit]


def _run_with_timeout(fn: Callable[[], ToolResult], timeout: float) -> ToolResult:
    """Run fn in a daemon thread; raise TimeoutError if it overruns.

    Python cannot kill a thread, so an overrunning tool keeps running in the
    background, but its result is discarded and the coordinator moves on.
    """
    box: dict = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as error:  # noqa: BLE001 - re-raised in the caller
            box["error"] = error

    worker = threading.Thread(target=target, daemon=True, name="phantom-tool")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"tool did not finish within {timeout:.1f} s")
    if "error" in box:
        raise box["error"]
    return box["result"]


def _validate_evidence(store: EvidenceStore, evidence_id: str, module: str) -> tuple[bytes, dict]:
    if not EVIDENCE_ID.match(evidence_id):
        raise ValueError(f"malformed evidence ID {evidence_id[:40]!r} (expected ev_ + 16 hex)")
    try:
        blob, record = store.get(evidence_id)
    except KeyError as error:
        raise ValueError(f"evidence {evidence_id} not found") from error
    kind = record.get("kind")
    if kind != EXPECTED_KIND[module]:
        raise ValueError(f"evidence {evidence_id} is a {kind!r} item, {module} needs "
                         f"{EXPECTED_KIND[module]!r}")
    if not store.verify(evidence_id):
        raise ValueError(f"evidence {evidence_id} failed its SHA-256 integrity check")
    return blob, record


def _analyze_stored(name: str, blob: bytes, record: dict, store: EvidenceStore) -> ToolResult:
    suffix = Path(record.get("source", "")).suffix[:10]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(blob)
        path = Path(temporary.name)
    try:
        analyze = analyze_document if name == "mirage" else analyze_audio
        return analyze(path, store, source=record["source"])
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:  # Windows: a parser may still hold the file; never mask the real error
            pass


def module_verdict(result: ToolResult) -> dict:
    """Each module's own headline verdict (the same rule its tab and evaluation use)."""
    try:
        if result.module.value == "blackbox":
            check = result.artifacts.get("url_check") or {}
            if check:
                return {"module": "blackbox", "verdict": check.get("verdict"),
                        "brand": check.get("brand")}
        rule = mirage_outcome if result.module.value == "mirage" else echo_outcome
        if result.module.value in ("mirage", "echo"):
            return {"module": result.module.value, "verdict": rule(result).value}
    except Exception:  # noqa: BLE001 - a verdict summary must never break the report
        pass
    outcomes = {f.outcome.value for f in result.findings}
    worst = next((o for o in ("suspicious", "inconclusive", "benign") if o in outcomes),
                 "inconclusive")
    return {"module": result.module.value, "verdict": worst}


def coordinate(
    goal: str,
    store: EvidenceStore,
    document_id: str | None = None,
    audio_id: str | None = None,
    target_url: str | None = None,
    max_calls: int = MAX_CALLS,
    max_seconds: float = 120,
    website_url: str | None = None,
    url_lookups: bool = False,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
    on_event: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Run the bounded investigation. on_event receives progress events
    ({"type": "plan" | "start" | "step", ...}) for the live UI; should_stop is
    checked before every tool call (a running tool is not interrupted)."""
    # ---- 1. validate the call itself (fail fast, nothing has run yet)
    if not isinstance(goal, str) or not goal.strip():
        raise CoordinatorInputError("goal must be a non-empty string")
    if len(goal) > MAX_GOAL_LENGTH:
        raise CoordinatorInputError(f"goal longer than {MAX_GOAL_LENGTH} characters")
    if not 1 <= max_calls <= MAX_CALLS or not 1 <= max_seconds <= 600:
        raise CoordinatorInputError("invalid coordinator budget")
    if not 1 <= tool_timeout <= 600:
        raise CoordinatorInputError("invalid per-tool timeout")
    goal_clean = re.sub(r"[\x00-\x1f\x7f]+", " ", goal).strip()
    goal_text = goal_clean.lower()

    start = time.monotonic()
    results: list[ToolResult] = []
    def emit(event: dict) -> None:
        if on_event:
            try:
                on_event(event)
            except Exception:  # noqa: BLE001 - a broken listener must not break the run
                pass

    class _Trace(list):  # every trace entry is also a progress event
        def append(self, item):
            super().append(item)
            emit({"type": "step", "step": item})

    trace: list[dict] = _Trace()
    errors: list[dict] = []
    warnings: list[dict] = []
    attempts = 0
    consecutive_failures = 0
    stop_reason = "completed"
    seen: set[tuple[str, str]] = set()

    def fail(name: str, code: str, message: str, status: str = "failed", counted=True):
        nonlocal consecutive_failures
        trace.append({"module": name, "status": status, "error": message, "code": code,
                      "elapsed_s": round(time.monotonic() - start, 3)})
        errors.append({"module": name, "code": code, "message": message})
        if counted:
            consecutive_failures += 1

    # ---- 2. the plan is fixed here, from the user's goal and inputs only
    supplied = {"mirage": document_id, "echo": audio_id, "blackbox": website_url or target_url}
    named = [name for name in ("mirage", "echo", "blackbox") if _asks_for(goal_text, name)]
    if goal_text in ALL_GOALS:
        routing = "all_requested"
    elif named:
        routing = "keywords"
    else:
        # The goal names no kind of evidence ("Was the person who rang me trying to
        # trick me?"). Keyword routing cannot tell what is meant, so check everything
        # the user supplied rather than nothing (found in the end-to-end evaluation:
        # these goals ran no tool at all). Still bounded by the same budgets.
        routing = "no_evidence_named"
    run_all = routing != "keywords"
    plan = [name for name in ("mirage", "echo", "blackbox")
            if supplied[name] and (run_all or name in named)]
    if routing == "no_evidence_named" and plan:
        warnings.append({"code": "goal_names_no_evidence",
                         "message": "The goal does not say which evidence to check, so all "
                                    "supplied evidence was checked."})
    if document_id and audio_id and document_id == audio_id:
        # one item cannot be both; the kind check below rejects the wrong one
        warnings.append({"code": "duplicate_input",
                         "message": "the same evidence ID was given as document and audio"})

    emit({"type": "plan", "plan": plan, "budget": {"max_calls": max_calls,
          "max_seconds": max_seconds, "tool_timeout": tool_timeout}})
    for name in plan:
        if should_stop and should_stop():
            trace.append({"module": name, "status": "skipped", "reason": "stopped by user"})
            stop_reason = "stopped_by_user"
            continue
        key = (name, str(supplied[name]))
        if key in seen:
            trace.append({"module": name, "status": "skipped", "reason": "repeated action"})
            continue
        seen.add(key)
        if attempts >= max_calls:
            trace.append({"module": name, "status": "skipped", "reason": "call budget"})
            stop_reason = "call_budget"
            continue
        remaining = max_seconds - (time.monotonic() - start)
        if remaining <= 0:
            trace.append({"module": name, "status": "skipped", "reason": "time budget"})
            stop_reason = "time_budget"
            continue
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            trace.append({"module": name, "status": "skipped",
                          "reason": "stopped after repeated failures"})
            stop_reason = "repeated_failures"
            continue

        # input validation for this module (malformed input: reported, not run)
        try:
            if name == "blackbox":
                url = str(supplied[name])
                if len(url) > MAX_URL_LENGTH or any(c in url for c in "\r\n\x00"):
                    raise ValueError("malformed URL (too long or contains control characters)")
                if website_url:
                    job = lambda u=url: check_url_tool(u, store, lookups=url_lookups)  # noqa: E731
                else:
                    job = lambda u=url: investigate(u, store, action_budget=2)  # noqa: E731
            else:
                blob, record = _validate_evidence(store, str(supplied[name]), name)
                job = lambda n=name, b=blob, r=record: _analyze_stored(n, b, r, store)  # noqa: E731
        except ValueError as error:
            fail(name, "invalid_input", str(error), counted=False)
            continue

        attempts += 1
        timeout = max(0.1, min(tool_timeout, remaining))
        emit({"type": "start", "module": name, "timeout_s": round(timeout, 1),
              "elapsed_s": round(time.monotonic() - start, 3)})
        t0 = time.monotonic()
        try:
            result = _run_with_timeout(job, timeout)
        except TimeoutError as error:
            fail(name, "tool_timeout", str(error), status="timed_out")
            continue
        except Exception as error:  # noqa: BLE001 - any tool crash is reported, not raised
            fail(name, "tool_failed", f"{type(error).__name__}: {error}")
            continue
        if not isinstance(result, ToolResult) or result.module.value != name:
            fail(name, "malformed_result", "tool returned something other than a "
                 f"{name} ToolResult ({type(result).__name__})")
            continue
        try:  # re-validate: a tool could build an object that breaks the contract
            result = ToolResult.model_validate(result.model_dump())
        except Exception as error:  # noqa: BLE001
            fail(name, "malformed_result", f"result failed contract validation: {error}")
            continue

        consecutive_failures = 0
        results.append(result)
        hits = sorted(set(scan_for_injection(_texts(
            [f.model_dump(mode="json") for f in result.findings] + [result.artifacts]))))
        if hits:
            warnings.append({
                "module": name, "code": "instructions_in_content", "patterns": hits,
                "message": f"{name} content contains instruction-like text ({', '.join(hits)}). "
                           "It was treated as data only: it did not change which tools ran "
                           "or any verdict. A reviewer should look at it: it can be a sign of "
                           "deliberate manipulation."})
        trace.append({"module": name, "status": "completed", "run_id": result.run_id,
                      "seconds": round(time.monotonic() - t0, 3),
                      "findings": len(result.findings)})

    missing = [name for name in ("mirage", "echo", "blackbox")
               if _asks_for(goal_text, name) and not supplied[name]]
    if not plan:
        missing.append("no relevant supplied input")
        stop_reason = "nothing_to_do"
    elif not results and stop_reason == "completed":
        stop_reason = "all_failed"
    if stop_reason == "completed" and errors:
        stop_reason = "completed_with_errors"
    findings = [
        {"module": result.module.value, **finding.model_dump(mode="json")}
        for result in results for finding in result.findings
    ]
    return {
        "goal": goal_clean,
        "status": "completed" if stop_reason == "completed" else
                  "partial" if results else "failed" if plan else "nothing_to_do",
        "stop_reason": stop_reason,
        "plan": plan,
        "routing": routing,
        "needs_human_review": True,
        "verdicts": [module_verdict(result) for result in results],
        "findings": findings,
        "evidence_ids": sorted({eid for result in results for eid in result.evidence_ids}),
        "results": [result.model_dump(mode="json") for result in results],
        "trace": trace,
        "missing_inputs": missing,
        # supplied evidence the goal did not ask about: listed, not analysed
        "ignored_inputs": [name for name in ("mirage", "echo", "blackbox")
                           if supplied[name] and name not in plan],
        "warnings": warnings,
        "limitations": [note for result in results for note in result.limitations],
        "errors": [
            {"module": result.module.value, **error.model_dump()}
            for result in results for error in result.errors
        ] + errors,
        "tool_calls": attempts,
        "budget": {"max_calls": max_calls, "max_seconds": max_seconds,
                   "tool_timeout": tool_timeout},
        "elapsed_s": round(time.monotonic() - start, 3),
    }
