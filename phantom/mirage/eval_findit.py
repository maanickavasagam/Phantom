"""Evaluate MIRAGE on the "Find it again!" receipt forgery dataset (Week 3).

Real scanned receipts (from SROIE) with forgeries made by people, plus
forgery bounding boxes. Download from
https://l3i-share.univ-lr.fr/2023Finditagain/ and unzip so that
data/raw/findit2/{train,val,test}/ and {train,val,test}.txt exist.

Rules for an honest number: tune only on train/val; run --split test once
and report it as-is (see docs/MIRAGE_EVAL.md).

    python -m phantom.mirage.eval_findit --split val
    python -m phantom.mirage.eval_findit --split test --json-out docs/mirage_eval_findit_test.json

Besides document-level precision/recall/F1 it reports localisation: of the
forged receipts MIRAGE flagged, how many had at least one flagged region
overlapping a region the forger actually modified.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import tempfile
import time
from pathlib import Path

import cv2

from phantom.common.evidence_store import EvidenceStore
from phantom.common.schemas import Outcome
from phantom.mirage.analyze import INPUT_CHECKS, analyze_document, overall_outcome
from phantom.mirage.ingest import TARGET_WIDTH

DEFAULT_ROOT = Path("data/raw/findit2")


def load_split(root: Path, split: str) -> list[dict]:
    rows = []
    with open(root / f"{split}.txt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            regions = []
            ann = ast.literal_eval(r["forgery annotations"]) if r["forged"] == "1" else None
            if isinstance(ann, dict):
                for reg in ann.get("regions", []):
                    attrs = reg.get("region_attributes", {})
                    if attrs.get("Original area") == "yes":  # copy source, not a modification
                        continue
                    s = reg["shape_attributes"]
                    if s.get("name") == "rect":
                        regions.append({
                            "bbox": [s["x"], s["y"], s["width"], s["height"]],
                            "entity": attrs.get("Entity type", "?"),
                        })
            rows.append({
                "file": r["image"],
                "path": root / split / r["image"],
                "forged": r["forged"] == "1",
                "handwritten": r["handwritten annotation"] == "1",
                "digital_annotation": r["digital annotation"] == "1",
                "regions": regions,
            })
    return rows


def _overlaps(a, b, margin: int = 4) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + margin < bx or bx + bw + margin < ax
                or ay + ah + margin < by or by + bh + margin < ay)


def _flagged_boxes(result, scale: float) -> list[list[float]]:
    """Bboxes of non-benign content findings, mapped back to original pixels.
    MIRAGE reports boxes on the normalized page (TARGET_WIDTH wide); small
    deskew rotations are ignored, hence the overlap margin."""
    boxes = []
    for f in result.findings:
        if f.outcome == Outcome.BENIGN or f.check in INPUT_CHECKS:
            continue
        for ev in f.evidence:
            bbox = ev.locator.get("bbox")
            if bbox:
                boxes.append([v * scale for v in bbox])
    return boxes


def evaluate(root: Path, split: str, limit: int | None = None) -> dict:
    all_rows = load_split(root, split)
    # The published zip is missing at least one labelled image (val/X51006619709.png).
    missing = [r["file"] for r in all_rows if not r["path"].exists()]
    rows = [r for r in all_rows if r["path"].exists()][:limit]
    store = EvidenceStore(Path(tempfile.gettempdir()) / "mirage_findit_eval")
    out = []
    for r in rows:
        t0 = time.perf_counter()
        result = analyze_document(r["path"], store, source=r["file"])
        elapsed = time.perf_counter() - t0
        verdict = overall_outcome(result)
        width = cv2.imread(str(r["path"]), cv2.IMREAD_GRAYSCALE).shape[1]
        boxes = _flagged_boxes(result, width / TARGET_WIDTH)
        hit = any(_overlaps(b, g["bbox"]) for b in boxes for g in r["regions"])
        out.append({
            "file": r["file"], "forged": r["forged"], "handwritten": r["handwritten"],
            "predicted": verdict.value, "localised": hit, "runtime_s": round(elapsed, 3),
            "entities": sorted({g["entity"] for g in r["regions"]}),
            "checks_fired": sorted({f.check for f in result.findings
                                    if f.outcome != Outcome.BENIGN
                                    and f.check not in (*INPUT_CHECKS, "overall")}),
            "errors": [e.code for e in result.errors],
        })

    def flagged(x):
        return x["predicted"] != "benign"

    tp = sum(1 for x in out if x["forged"] and flagged(x))
    fn = sum(1 for x in out if x["forged"] and not flagged(x))
    fp = sum(1 for x in out if not x["forged"] and flagged(x))
    tn = sum(1 for x in out if not x["forged"] and not flagged(x))
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    benign_hw = [x for x in out if not x["forged"] and x["handwritten"]]
    by_check: dict[str, dict] = {}
    for x in out:
        for c in x["checks_fired"]:
            d = by_check.setdefault(c, {"on_forged": 0, "on_genuine": 0})
            d["on_forged" if x["forged"] else "on_genuine"] += 1
    return {
        "split": split, "n_documents": len(out), "n_forged": tp + fn,
        "missing_from_download": missing,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(p, 3), "recall": round(rc, 3), "f1": round(f1, 3),
        "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        "flagged_forged_correctly_localised": sum(x["localised"] for x in out if x["forged"]
                                                  and flagged(x)),
        "fp_rate_on_genuine_with_handwriting": round(
            sum(flagged(x) for x in benign_hw) / len(benign_hw), 3) if benign_hw else None,
        "fired_by_check": by_check,
        "mean_runtime_s": round(sum(x["runtime_s"] for x in out) / len(out), 3) if out else 0,
        "errors": sum(bool(x["errors"]) for x in out),
        "rows": out,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    metrics = evaluate(Path(a.root), a.split, a.limit)
    print(json.dumps({k: v for k, v in metrics.items() if k != "rows"}, indent=2))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(metrics, indent=2))
