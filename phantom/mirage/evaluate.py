"""MIRAGE evaluation (Day 11).

Runs analyze_document over the labelled sample set and reports
precision, recall, F1, false-positive rate and per-document runtime.
A document is predicted "suspicious" if any non-quality finding is
suspicious or inconclusive (inconclusive counts toward recall since the
project's goal is to flag for human review, not to silently pass).

    python -m phantom.mirage.evaluate --samples data/samples --holdout-template 2
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from phantom.common.evidence_store import EvidenceStore
from phantom.common.schemas import Outcome
from phantom.mirage.analyze import analyze_document, overall_outcome


@dataclass
class EvalRow:
    file: str
    label: str
    predicted_suspicious: bool
    runtime_s: float
    attack: str | None = None  # manipulation type (+ variant) for suspicious docs
    checks_fired: tuple[str, ...] = ()


def _attack_name(entry: dict) -> str | None:
    """Manipulation type from either label format (make_samples / make_hard_samples)."""
    if entry["label"] != "suspicious":
        return None
    m = entry.get("manipulation")
    if m:
        variant = "consistent" if m.get("arithmetic_consistent") else "inconsistent"
        return f"{m['type']} ({m['stage']}, {variant})"
    regions = entry.get("regions") or [{}]
    return regions[0].get("type", "unknown")


def _recall_by_attack(rows: list[EvalRow]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        if r.attack is None:
            continue
        d = out.setdefault(r.attack, {"n": 0, "caught": 0})
        d["n"] += 1
        d["caught"] += int(r.predicted_suspicious)
    for d in out.values():
        d["recall"] = round(d["caught"] / d["n"], 3)
    return dict(sorted(out.items()))


def _predict(result) -> bool:
    return overall_outcome(result) != Outcome.BENIGN


def evaluate(
    samples_dir: str | Path, holdout_template: int | None, evidence_dir: str | Path
) -> dict:
    samples_dir = Path(samples_dir)
    labels = json.loads((samples_dir / "labels.json").read_text())
    if holdout_template is not None:
        labels = [entry for entry in labels if entry["template"] == holdout_template]
    store = EvidenceStore(evidence_dir)

    rows: list[EvalRow] = []
    for entry in labels:
        start = time.perf_counter()
        result = analyze_document(samples_dir / entry["file"], store, source=entry["file"])
        elapsed = time.perf_counter() - start
        fired = tuple(sorted({f.check for f in result.findings
                              if f.outcome != Outcome.BENIGN and f.check != "overall"}))
        rows.append(EvalRow(entry["file"], entry["label"], _predict(result), round(elapsed, 3),
                            _attack_name(entry), fired))

    tp = sum(1 for r in rows if r.label == "suspicious" and r.predicted_suspicious)
    fn = sum(1 for r in rows if r.label == "suspicious" and not r.predicted_suspicious)
    fp = sum(1 for r in rows if r.label == "benign" and r.predicted_suspicious)
    tn = sum(1 for r in rows if r.label == "benign" and not r.predicted_suspicious)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    mean_runtime = sum(r.runtime_s for r in rows) / len(rows) if rows else 0.0

    return {
        "n_documents": len(rows),
        "holdout_template": holdout_template,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positive_rate": round(false_positive_rate, 3),
        "mean_runtime_s": round(mean_runtime, 3),
        "recall_by_attack": _recall_by_attack(rows),
        "false_positive_files": [r.file for r in rows
                                 if r.label == "benign" and r.predicted_suspicious],
        "rows": [r.__dict__ for r in rows],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--holdout-template", type=int, default=None,
                     help="evaluate only this template family, simulating a held-out source")
    ap.add_argument("--evidence-dir", default=str(Path(tempfile.gettempdir()) / "mirage_eval"))
    ap.add_argument("--json-out", default=None, help="also write full metrics (with rows) here")
    args = ap.parse_args()
    metrics = evaluate(args.samples, args.holdout_template, args.evidence_dir)
    print(json.dumps({k: v for k, v in metrics.items() if k != "rows"}, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(metrics, indent=2))
