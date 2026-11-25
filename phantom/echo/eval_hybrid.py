"""Compare rules-only, classifier-only and hybrid scam detection (Week 4).

Held-out data only: the classifier was trained on the BothBosu train halves.
  - BothBosu single/multi-agent TEST halves
  - BothBosu Scammer-Conversation (other generator) and youtube (20 real scams)
  - our own scripted calls (Indian scam styles the classifier never saw), scored
    on the Whisper transcripts already recorded in docs/echo_eval_calls_{dev,test}.json

    python -m phantom.echo.eval_hybrid --json-out docs/echo_eval_hybrid.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phantom.echo import scam_classifier
from phantom.echo.claims import extract_claims
from phantom.echo.eval_public import _rows, dialogue_to_transcript
from phantom.echo.indicators import decide, detect_indicators

MODES = ("rules", "classifier", "hybrid", "corroborated", "flag_for_review")


def predict_all(text: str) -> dict[str, bool]:
    tr = dialogue_to_transcript(text)
    d = decide(detect_indicators(tr, extract_claims(tr)))
    s = scam_classifier.score(text)
    clf = bool(s and s.suspicious)
    return {"rules": d.suspicious, "classifier": clf, "hybrid": d.suspicious or clf,
            # classifier may only convict when the rules found supporting evidence
            # (the live pipeline's verdict; see analyze.py)
            "corroborated": d.suspicious or (clf and d.supports_classifier),
            # everything that would at least be routed to a reviewer as "inconclusive"
            "flag_for_review": d.suspicious or clf}


def _metrics(items: list[tuple[int, dict[str, bool]]]) -> dict:
    out = {}
    for mode in MODES:
        tp = sum(1 for y, p in items if y and p[mode])
        fp = sum(1 for y, p in items if not y and p[mode])
        fn = sum(1 for y, p in items if y and not p[mode])
        tn = sum(1 for y, p in items if not y and not p[mode])
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0
        out[mode] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(pr, 3),
                     "recall": round(rc, 3), "f1": f1,
                     "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else None}
    return out


def evaluate(root: Path, calls_docs: Path) -> dict:
    results = {}
    for name, split in (("single_agent", "test"), ("multi_agent", "test"),
                        ("scammer_conversation", None), ("youtube_real", None)):
        items = [(y, predict_all(t)) for t, y, _ in _rows(root, name, split)]
        results[f"{name}" + (f"_{split}" if split else "")] = {"n": len(items), **_metrics(items)}
    for split in ("dev", "test"):
        f = calls_docs / f"echo_eval_calls_{split}.json"
        if not f.exists():
            continue
        rows = [r for r in json.loads(f.read_text())["rows"] if r["condition"] == "clean"]
        items = [(int(r["label"] == "scam"), predict_all(r["hypothesis"])) for r in rows]
        per_family = {}
        for r, (_, p) in zip(rows, items, strict=True):
            fam = f"{r['family']} ({r['label']})"
            per_family.setdefault(fam, {m: 0 for m in MODES} | {"n": 0})
            per_family[fam]["n"] += 1
            for m in MODES:
                per_family[fam][m] += int(p[m])
        results[f"own_scripted_{split}"] = {"n": len(items), **_metrics(items),
                                            "flagged_by_family": per_family}
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/bothbosu")
    ap.add_argument("--calls-docs", default="docs")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    res = evaluate(Path(a.root), Path(a.calls_docs))
    for name, r in res.items():
        line = " | ".join(f"{m}: P {r[m]['precision']:.2f} R {r[m]['recall']:.2f} "
                          f"F1 {r[m]['f1']:.2f} FPR {r[m]['false_positive_rate']}" for m in MODES)
        print(f"{name:26} n={r['n']:5}  {line}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=2))
