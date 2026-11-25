"""BLACKBOX investigation: one explorer run on a local test site -> ToolResult.

Only the locally generated synthetic sites (served on 127.0.0.1) can be
investigated; live websites are out of scope by design (docs/SCOPE.md).
Every finding carries a screenshot of the page where the behaviour was seen
and the action trace (the clicks/submits that led there) as evidence.
"""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path

from phantom.blackbox.browser import SafeBrowser
from phantom.blackbox.detectors import detect
from phantom.blackbox.explore import FixedRuleStrategy, RandomStrategy, run_episode
from phantom.common.evidence_store import EvidenceStore
from phantom.common.schemas import (
    EvidenceRef,
    Finding,
    ModuleName,
    Outcome,
    ToolError,
    ToolResult,
    Uncertainty,
)

LABELS = {
    "credential_form": "Credential-harvesting form",
    "payment_demand": "Upfront fee demand (UPI / gift card)",
    "fake_urgency": "Fake countdown / artificial scarcity",
    "forced_download": "Pushes an executable download",
    "deceptive_redirect": "Silent redirect to a login page",
    "exfil_form": "Form sends personal data to another site",
    "support_popup": "Scareware pop-up with a phone number",
}
_lock = threading.Lock()  # one headless browser run at a time
_server: dict = {}


def _site_server(root: Path) -> str:
    from phantom.blackbox.sites import serve

    if "base" not in _server:
        _server["srv"], _server["base"] = serve(root)
    return _server["base"]


def list_sites(root: Path) -> list[dict]:
    idx = root / "index.json"
    return json.loads(idx.read_text()) if idx.exists() else []


def make_strategy(name: str):
    if name == "learned":
        from phantom.blackbox.ranker import LearnedStrategy

        return LearnedStrategy()
    return {"random": RandomStrategy, "fixed_rule": FixedRuleStrategy}[name]()


def investigate(root: Path, site_id: str, store: EvidenceStore, strategy: str = "learned",
                budget: int = 12, seed: int = 0) -> ToolResult:
    result = ToolResult(module=ModuleName.BLACKBOX)
    if site_id not in {s["id"] for s in list_sites(root)}:
        result.errors.append(ToolError(code="unknown_site", recoverable=False,
                                       message="only the local synthetic test sites can be "
                                               "investigated (live sites are out of scope)"))
        return result
    budget = max(1, min(int(budget), 30))
    base = _site_server(root)
    shots: dict[str, str] = {}
    with _lock, SafeBrowser(base) as b:
        def capture(state, actions, ctx, found):
            # screenshot every page where a behaviour is first seen (evidence)
            for d in detect(state, base):
                key = f"{d.type}:{d.page}"
                if key not in shots:
                    shots[key] = store.put(b.screenshot(), kind="screenshot",
                                           source=f"{site_id}{state.path}",
                                           meta={"url": state.url, "behaviour": d.type})

        ep = run_episode(b, base, site_id, make_strategy(strategy), budget,
                         random.Random(seed), on_state=capture)
        # the final page has no further decision point: capture it too
        last = b.page.url
        for d in detect(b.load(last), base) if last.startswith(base) else []:
            key = f"{d.type}:{d.page}"
            if key not in shots:
                shots[key] = store.put(b.screenshot(), kind="screenshot",
                                       source=f"{site_id}{d.page}", meta={"behaviour": d.type})
        blocked = list(b.blocked)
    trace = [s.__dict__ for s in ep.steps]
    trace_id = store.put(json.dumps({"site": site_id, "strategy": ep.strategy, "budget": budget,
                                     "steps": trace, "blocked_requests": blocked},
                                    indent=1).encode(),
                         kind="action_trace", source=f"{site_id}#trace")
    result.evidence_ids.append(trace_id)
    for (btype, page), step in sorted(ep.found.items(), key=lambda kv: kv[1]):
        sid = shots.get(f"{btype}:/{site_id}/{page}") or next(
            (v for k, v in shots.items() if k.startswith(btype + ":") and k.endswith(page)), None)
        refs = [EvidenceRef(evidence_id=trace_id, kind="action_trace",
                            locator={"step": step, "page": page})]
        if sid:
            result.evidence_ids.append(sid)
            refs.insert(0, EvidenceRef(evidence_id=sid, kind="screenshot", locator={"page": page}))
        step_rec = next((s for s in ep.steps if s.step == step), None)
        reason = next((d["reason"] for d in (step_rec.detections if step_rec else [])
                       if d.get("type") == btype), "observed on the start page")
        result.findings.append(Finding(
            check=btype, outcome=Outcome.SUSPICIOUS,
            description=f"{LABELS.get(btype, btype)} on {page} (found after {step} action"
                        f"{'s' if step != 1 else ''}): {reason}",
            evidence=refs,
            uncertainty=Uncertainty(confidence=0.9, notes="rule-based detector; validated with "
                                    "0 false positives on decoys of the synthetic sites"),
        ))
    if not result.findings:
        result.findings.append(Finding(
            check="overall", outcome=Outcome.BENIGN,
            description=f"No planted behaviour found within {budget} actions.",
            evidence=[EvidenceRef(evidence_id=trace_id, kind="action_trace")],
            uncertainty=Uncertainty(confidence=0.5, notes="a limited budget may miss deep pages")))
    result.artifacts = {"site": site_id, "strategy": ep.strategy, "budget": budget,
                        "actions_used": len(ep.steps), "pages_visited": sorted(ep.pages_visited),
                        "trace": trace, "blocked_requests": blocked,
                        "runtime_s": round(ep.runtime_s, 2)}
    result.limitations += [
        "Synthetic local test sites only; live websites are not browsed (scope guardrail).",
        "Detectors are rules written for these sites' planted behaviours; real sites would "
        "need broader detectors.",
        f"Exploration was limited to {budget} actions; behaviours deeper than that can be missed.",
    ]
    return result
