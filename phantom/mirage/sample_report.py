"""Generate a human-readable MIRAGE report from a ToolResult (Day 12).

    python -m phantom.mirage.sample_report data/samples/inv_000_suspicious.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phantom.common.evidence_store import EvidenceStore
from phantom.mirage.analyze import analyze_document


def render(result) -> str:
    lines = [
        f"MIRAGE report — run {result.run_id}",
        f"Started: {result.started_at.isoformat()}",
        f"Human review required: {'yes' if result.needs_human_review else 'no'}",
        "",
    ]
    if result.errors:
        lines.append("ERRORS:")
        for e in result.errors:
            lines.append(f"  [{e.code}] {e.message}")
        return "\n".join(lines)

    by_outcome = {"suspicious": [], "inconclusive": [], "benign": []}
    for f in result.findings:
        by_outcome[f.outcome.value].append(f)

    for outcome in ("suspicious", "inconclusive", "benign"):
        items = by_outcome[outcome]
        if not items:
            continue
        lines.append(f"{outcome.upper()} ({len(items)}):")
        for f in items:
            loc = f.evidence[0].locator if f.evidence else {}
            lines.append(f"  - [{f.check}] {f.description}")
            eid = f.evidence[0].evidence_id if f.evidence else "none"
            lines.append(f"      evidence: {eid} {loc}")
            lines.append(f"      confidence: {f.uncertainty.confidence} ({f.uncertainty.notes})")
        lines.append("")

    if result.limitations:
        lines.append("LIMITATIONS:")
        for note in result.limitations:
            lines.append(f"  - {note}")

    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--evidence-dir", default="/tmp/mirage_report_evidence")
    args = ap.parse_args()
    store = EvidenceStore(args.evidence_dir)
    result = analyze_document(args.path, store, source=Path(args.path).name)
    print(render(result))
    sys.exit(0)
