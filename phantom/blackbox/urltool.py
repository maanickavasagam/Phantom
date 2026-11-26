"""BLACKBOX URL check as a shared-contract ToolResult (API + coordinator)."""

from __future__ import annotations

import json

from phantom.blackbox.brands import BRANDS
from phantom.blackbox.urlcheck import check_url, explain
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

_OUTCOME = {"official": Outcome.BENIGN, "likely_fake": Outcome.SUSPICIOUS,
            "suspicious": Outcome.INCONCLUSIVE, "no_brand_link": Outcome.BENIGN}
_SIGNAL_OUTCOME = {"strong": Outcome.SUSPICIOUS, "medium": Outcome.INCONCLUSIVE,
                   "weak": Outcome.INCONCLUSIVE}
MAX_URL_LENGTH = 2048
_STRENGTH_CONF = {"strong": 0.8, "medium": 0.55, "weak": 0.3}


def check_url_tool(url: str, store: EvidenceStore, lookups: bool = False) -> ToolResult:
    result = ToolResult(module=ModuleName.BLACKBOX)
    if not url or len(url) > MAX_URL_LENGTH:
        result.errors.append(ToolError(code="invalid_url", recoverable=False,
                                       message=f"URL missing or longer than "
                                               f"{MAX_URL_LENGTH} characters"))
        return result
    report = check_url(url, lookups=lookups)
    if report.verdict == "invalid":
        result.errors.append(ToolError(code="invalid_url", message=report.summary,
                                       recoverable=False))
        return result
    ev = store.put(json.dumps(report.to_dict(), indent=1, default=str).encode(),
                   kind="url_analysis", source=url[:200],
                   meta={"verdict": report.verdict, "brand": report.brand,
                         "lookups": bool(lookups)})
    result.evidence_ids.append(ev)
    ref = [EvidenceRef(evidence_id=ev, kind="url_analysis",
                       locator={"url": url[:200], "registrable": report.registrable})]
    conf = {"official": 0.95, "likely_fake": 0.85, "suspicious": 0.5, "no_brand_link": 0.4}
    result.findings.append(Finding(
        check=f"verdict_{report.verdict}", description=report.summary,
        outcome=_OUTCOME[report.verdict], evidence=ref,
        uncertainty=Uncertainty(confidence=conf[report.verdict],
                                notes="rule-based URL analysis; the page itself was not opened")))
    for s in report.signals:
        brand = f" [{BRANDS[s.brand].name}]" if s.brand else ""
        result.findings.append(Finding(
            check=s.name, description=s.detail + brand,
            outcome=Outcome.BENIGN if report.verdict == "official" else _SIGNAL_OUTCOME[s.strength],
            evidence=ref,
            uncertainty=Uncertainty(confidence=_STRENGTH_CONF[s.strength],
                                    notes=f"{s.strength} signal")))
    result.artifacts = {"url_check": report.to_dict(), "explain": explain(report, url)}
    result.limitations += [
        "Judged from the URL (and, if enabled, DNS + domain registration date) only; the page "
        "was never opened, so a fake with a clean-looking domain, or a compromised official "
        "page, cannot be detected.",
        "Only the brands in the registry are recognised (12 Indian brands); others get "
        "'no brand link', which is not a safety guarantee.",
    ]
    if lookups:
        result.limitations.append("Safe lookups sent the domain name (not the full URL) to the "
                                  "DNS resolver and rdap.org.")
    return result
