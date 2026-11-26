"""Bounded exploration of the project's local, synthetic sites."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from phantom.common.evidence_store import EvidenceStore
from phantom.common.schemas import (
    EvidenceRef,
    Finding,
    ModuleName,
    Outcome,
    ToolResult,
    Uncertainty,
)

SCENARIOS = {
    "fees": ("pricing", "A fee appears only after opening the payment details."),
    "privacy": ("privacy", "The form reveals an unexpected data-sharing notice."),
    "redirect": ("redirect", "A local checkout link changes its stated destination."),
    "urgency": ("deadline", "A synthetic timer pressures the visitor to act."),
    "subscription": ("terms", "The terms disclose a recurring charge."),
    "identity": ("identity", "The contact page gives conflicting operator details."),
}
MODEL_PATH = Path(__file__).with_name("action_ranker.joblib")
HINTS = ("fee", "privacy", "detail", "terms", "identity", "deadline")


def scenario_html(scenario: str, page: str = "start") -> str:
    if scenario not in SCENARIOS or page not in {"start", "safe", "clue"}:
        raise ValueError("unknown synthetic page")
    clue, description = SCENARIOS[scenario]
    if page == "clue":
        body = f'<p data-behaviour="{clue}">{description}</p>'
    elif page == "safe":
        body = "<p>General information; no planted behaviour here.</p>"
    else:
        body = (
            f'<a href="/synthetic/{scenario}/safe">Overview</a> '
            f'<a href="/synthetic/{scenario}/clue">Review {clue} details</a>'
        )
    return f"<!doctype html><title>Phantom synthetic {scenario}</title><main>{body}</main>"


def allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and parsed.port is not None
        and parsed.path.startswith("/synthetic/")
        and not parsed.username
        and not parsed.password
    )


def action_features(text: str) -> list[float]:
    text = text.lower()
    return [float(word in text) for word in HINTS] + [float("overview" in text)]


def train_ranker(path: Path = MODEL_PATH) -> dict:
    """Train a small neural ranker on labelled synthetic actions."""
    from joblib import dump
    from sklearn.neural_network import MLPClassifier

    samples = [
        ("Overview", 0), ("General information", 0), ("Home", 0),
        ("Review fee details", 1), ("Review privacy details", 1),
        ("Review deadline details", 1), ("Review more details", 1),
    ]
    x = [action_features(text) for text, _ in samples] * 12
    y = [label for _, label in samples] * 12
    model = MLPClassifier(
        hidden_layer_sizes=(8,), solver="lbfgs", random_state=26, max_iter=500
    )
    model.fit(x, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
    return {"samples": len(x), "training_accuracy": float(model.score(x, y))}


def _choose(actions: list[tuple[str, str]], strategy: str, rng: random.Random) -> tuple[str, str]:
    if strategy == "random":
        return rng.choice(actions)
    if strategy == "fixed":
        return actions[0]
    if strategy != "learned":
        raise ValueError("strategy must be fixed, random, or learned")
    from joblib import load

    if not MODEL_PATH.is_file():
        raise RuntimeError("train the action ranker before using the learned strategy")
    model = load(MODEL_PATH)
    scores = model.predict_proba([action_features(text) for text, _ in actions])[:, 1]
    return actions[int(scores.argmax())]


def investigate(
    target_url: str,
    store: EvidenceStore,
    strategy: str = "learned",
    action_budget: int = 2,
    seed: int = 26,
) -> ToolResult:
    """Visit only local synthetic pages, recording screenshots and actions."""
    if not allowed_url(target_url):
        raise ValueError("target must be an allowlisted local synthetic URL")
    if not 1 <= action_budget <= 10:
        raise ValueError("action budget must be between 1 and 10")
    from playwright.sync_api import sync_playwright

    result = ToolResult(module=ModuleName.BLACKBOX)
    trace: list[dict] = []
    origin = urlparse(target_url).netloc
    rng = random.Random(seed)
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Exception:
            try:
                browser = playwright.chromium.launch(channel="msedge", headless=True)
            except Exception as error:
                raise RuntimeError(
                    "Playwright could not start Edge or Chromium; install its browser runtime"
                ) from error
        try:
            page = browser.new_page()
            page.goto(target_url, wait_until="domcontentloaded", timeout=10000)
            for step in range(action_budget + 1):
                url = page.url
                if not allowed_url(url) or urlparse(url).netloc != origin:
                    raise ValueError("browser left the allowed local origin")
                screenshot = page.screenshot(timeout=10000)
                evidence_id = store.put(
                    screenshot, kind="screenshot", source=url, meta={"step": step}
                )
                result.evidence_ids.append(evidence_id)
                for marker in page.locator("[data-behaviour]").all():
                    finding = Finding(
                        check="planted_behaviour",
                        description=marker.inner_text(),
                        outcome=Outcome.SUSPICIOUS,
                        evidence=[EvidenceRef(
                            evidence_id=evidence_id, kind="screenshot",
                            locator={"url": url, "step": step}
                        )],
                        uncertainty=Uncertainty(confidence=1.0, notes="synthetic ground truth"),
                    )
                    result.findings.append(finding)
                if result.findings or step == action_budget:
                    break
                actions = []
                for link in page.locator("a[href]").all():
                    text = link.inner_text()
                    href = urljoin(url, link.get_attribute("href") or "")
                    if allowed_url(href) and urlparse(href).netloc == origin:
                        actions.append((text, href))
                if not actions:
                    break
                text, next_url = _choose(actions, strategy, rng)
                trace.append({"step": step + 1, "action": text, "from": url, "to": next_url})
                page.goto(next_url, wait_until="domcontentloaded", timeout=10000)
        finally:
            browser.close()
    trace_id = store.put(
        json.dumps(trace).encode(), kind="action_trace", source=target_url,
        meta={"strategy": strategy, "budget": action_budget}
    )
    result.evidence_ids.append(trace_id)
    result.artifacts = {"trace": trace, "strategy": strategy, "actions": len(trace)}
    if not result.findings:
        result.limitations.append("No planted behaviour found within the action budget.")
    return result


def evaluate(base_url: str, store: EvidenceStore, repeats: int = 10) -> dict:
    """Compare all policies on held-out scenarios under identical budgets."""
    if not 1 <= repeats <= 100:
        raise ValueError("repeats must be between 1 and 100")
    report = {}
    for strategy in ("fixed", "random", "learned"):
        outcomes = []
        actions_to_discovery = []
        runtimes = []
        false_positives = 0
        for scenario in ("subscription", "identity"):
            for seed in range(repeats):
                started = time.monotonic()
                result = investigate(
                    f"{base_url}/synthetic/{scenario}/start", store,
                    strategy=strategy, action_budget=1, seed=seed
                )
                runtimes.append(time.monotonic() - started)
                found = bool(result.findings)
                outcomes.append(found)
                if found:
                    actions_to_discovery.append(result.artifacts["actions"])
                false_positives += sum(
                    not finding.evidence[0].locator.get("url", "").endswith(
                        f"/synthetic/{scenario}/clue"
                    )
                    for finding in result.findings
                )
        report[strategy] = {
            "discovery_rate": sum(outcomes) / len(outcomes),
            "coverage": sum(outcomes) / len(outcomes),
            "false_positives": false_positives,
            "mean_actions_to_discovery": (
                sum(actions_to_discovery) / len(actions_to_discovery)
                if actions_to_discovery else None
            ),
            "mean_runtime_s": round(sum(runtimes) / len(runtimes), 3),
            "trials": len(outcomes), "action_budget": 1,
        }
    return report
