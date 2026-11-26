"""MIRAGE top-level analysis (Day 10).

Combines ingestion, OCR and visual checks into the shared ToolResult
contract. Every finding carries evidence; a document with no flags is
reported as benign, one with low-confidence OCR or ambiguous flags is
reported as inconclusive rather than guessed at.
"""

from __future__ import annotations

from pathlib import Path

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
from phantom.mirage.consistency import ConsistencyResult, check_numeric_consistency
from phantom.mirage.copy_move import CLONE_THRESHOLD, find_clones
from phantom.mirage.fields import extract_amounts, looks_like_receipt
from phantom.mirage.ingest import IngestError, ingest
from phantom.mirage.ocr import ocr_page
from phantom.mirage.visual_checks import VisualFlag, run_visual_checks, table_checks_applicable

LOW_OCR_CONFIDENCE = 40.0  # below this, treat page text as unreliable
INPUT_CHECKS = ("input_quality", "ocr_confidence")


def overall_outcome(result: ToolResult) -> Outcome:
    """Document-level verdict used by evaluation and the demo UI: suspicious if
    any content check is suspicious, else inconclusive if any is inconclusive.
    Input-quality notes alone never make a document suspicious."""
    outcomes = {f.outcome for f in result.findings if f.check not in INPUT_CHECKS}
    if Outcome.SUSPICIOUS in outcomes:
        return Outcome.SUSPICIOUS
    if Outcome.INCONCLUSIVE in outcomes:
        return Outcome.INCONCLUSIVE
    return Outcome.BENIGN


def _consistency_to_finding(
    res: ConsistencyResult, page_evidence_id: str, page_number: int
) -> Finding | None:
    if res.status == "not_applicable":
        return None
    outcome = {
        "mismatch": Outcome.SUSPICIOUS,
        "unreadable": Outcome.INCONCLUSIVE,
        "consistent": Outcome.BENIGN,
    }[res.status]
    evidence = [
        EvidenceRef(
            evidence_id=page_evidence_id,
            kind="page_region",
            locator={"page": page_number, "bbox": list(w.bbox), "text": w.text},
        )
        for w in res.words
    ] or [EvidenceRef(evidence_id=page_evidence_id, kind="page_region",
                      locator={"page": page_number})]
    return Finding(
        check="numeric_consistency",
        description=res.description,
        outcome=outcome,
        evidence=evidence,
        uncertainty=Uncertainty(
            confidence=res.severity if outcome != Outcome.BENIGN else 0.8,
            notes="arithmetic over OCR'd amounts; only as reliable as the OCR read",
        ),
    )


def _flag_to_finding(flag: VisualFlag, page_evidence_id: str, page_number: int) -> Finding:
    outcome = Outcome.SUSPICIOUS if flag.severity >= 0.5 else Outcome.INCONCLUSIVE
    left, top, width, height = flag.bbox
    return Finding(
        check=flag.check,
        description=flag.description,
        outcome=outcome,
        evidence=[
            EvidenceRef(
                evidence_id=page_evidence_id,
                kind="page_region",
                locator={"page": page_number, "bbox": [left, top, width, height]},
            )
        ],
        uncertainty=Uncertainty(
            confidence=round(flag.severity, 2), notes="heuristic visual check, uncalibrated"
        ),
    )


def _boxes_touch(a, b, margin: int = 8) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + margin < bx or bx + bw + margin < ax
                or ay + ah + margin < by or by + bh + margin < ay)


def _corroborated(flag: VisualFlag, flags: list[VisualFlag]) -> bool:
    """A layout/visual flag only counts when a *different* check flags the same
    field. Each of these heuristics alone fired on genuine receipts about as
    often as on forged ones (Find it again! train split); agreement between
    independent checks on one field is far rarer by chance."""
    return any(o.check != flag.check and _boxes_touch(o.bbox, flag.bbox) for o in flags)


def _copy_move_finding(page, page_evidence_id: str) -> tuple[Finding | None, str | None]:
    """Returns (finding, limitation note)."""
    if page.original is None:
        return None, None
    res = find_clones(page.original)
    if res.status == "not_applicable":
        return None, f"Page {page.page_number}: copy-move check skipped ({res.note})."
    if res.status == "clean":
        return None, None
    evidence = []
    for k, pair in enumerate(res.pairs, start=1):
        for role, box in (("A", pair.a), ("B", pair.b)):
            evidence.append(EvidenceRef(
                evidence_id=page_evidence_id, kind="page_region",
                locator={"page": page.page_number, "bbox": list(page.map_bbox(box)),
                         "original_bbox": list(box), "clone_pair": k, "role": role,
                         "mean_abs_diff": round(pair.diff, 3)},
            ))
    best = res.pairs[0].diff
    n = len(res.pairs)
    return Finding(
        check="copy_move",
        description=(
            f"{n} pair{'s' if n > 1 else ''} of characters are pixel-for-pixel clones "
            f"(closest pair differs by {best:.2f} grey levels on average, scan noise included). "
            "Genuinely printed and scanned characters never match this closely; this is the "
            "trace left by copy-pasting a character from elsewhere on the page. Which copy is "
            "the original can't be told."
        ),
        outcome=Outcome.SUSPICIOUS,
        evidence=evidence,
        uncertainty=Uncertainty(
            confidence=0.9,
            notes=f"pixel comparison at original resolution; threshold {CLONE_THRESHOLD} "
                  "grey levels set on Find it again! train split",
        ),
    ), None


def analyze_document(
    path: str | Path, store: EvidenceStore, source: str | None = None
) -> ToolResult:
    path = Path(path)
    result = ToolResult(module=ModuleName.MIRAGE)
    data = path.read_bytes()
    doc_evidence_id = store.put(data, kind="document", source=source or path.name)
    result.evidence_ids.append(doc_evidence_id)

    try:
        pages = ingest(path)
    except IngestError as e:
        result.errors.append(ToolError(code="ingest_failed", message=str(e)))
        return result

    for page in pages:
        page_bytes = page.image.tobytes()
        page_evidence_id = store.put(
            page_bytes, kind="page_region", source=f"{source or path.name}#page{page.page_number}",
            meta={"page": page.page_number, "shape": page.image.shape},
        )
        result.evidence_ids.append(page_evidence_id)

        if page.quality.issues:
            result.findings.append(
                Finding(
                    check="input_quality",
                    description=f"Page {page.page_number}: {', '.join(page.quality.issues)}",
                    outcome=Outcome.INCONCLUSIVE,
                    evidence=[EvidenceRef(evidence_id=page_evidence_id, kind="page_region",
                                          locator={"page": page.page_number})],
                    uncertainty=Uncertainty(confidence=0.5, notes="heuristic quality check"),
                )
            )

        ocr = ocr_page(page.image, page.page_number)
        if ocr.mean_confidence and ocr.mean_confidence < LOW_OCR_CONFIDENCE:
            result.findings.append(
                Finding(
                    check="ocr_confidence",
                    description=(
                        f"Page {page.page_number}: mean OCR confidence "
                        f"{ocr.mean_confidence:.0f} is low"
                    ),
                    outcome=Outcome.INCONCLUSIVE,
                    evidence=[EvidenceRef(evidence_id=page_evidence_id, kind="page_region",
                                          locator={"page": page.page_number})],
                    uncertainty=Uncertainty(
                        confidence=0.4,
                        notes="low OCR confidence; findings below may be unreliable",
                    ),
                )
            )

        n = page.page_number
        receipt = looks_like_receipt(ocr)
        # Invoice-style table checks only on invoice-style pages. On real till
        # receipts (Find it again! train split) arithmetic and layout checks
        # fired on genuine receipts ~10x more often than on forged ones (OCR
        # misreads, mixed fonts, handwriting), so there they are notes only.
        table_layout = table_checks_applicable(extract_amounts(ocr)) and not receipt
        if receipt:
            result.limitations.append(
                f"Page {n}: recognised as a till receipt. Verdict uses the checks validated on "
                "real receipts (copy-paste clones); arithmetic and layout checks are reported as "
                "notes only."
            )
        arithmetic = check_numeric_consistency(ocr)
        if arithmetic.status == "unreadable":
            # Not being able to read a number is not evidence of tampering.
            result.limitations.append(f"Page {n}: {arithmetic.description}")
        elif arithmetic.status == "mismatch" and not table_layout:
            result.limitations.append(
                f"Page {n}: arithmetic note, not used for the verdict on this layout "
                f"(unreliable on receipt layouts because of OCR misreads): {arithmetic.description}"
            )
        else:
            consistency = _consistency_to_finding(arithmetic, page_evidence_id, n)
            if consistency is not None:
                result.findings.append(consistency)

        clone_finding, clone_note = _copy_move_finding(page, page_evidence_id)
        if clone_finding is not None:
            result.findings.append(clone_finding)
        if clone_note:
            result.limitations.append(clone_note)

        if not table_layout and not receipt:
            result.limitations.append(
                f"Page {n}: layout is not a one-amount-per-row table, so the table layout "
                "checks (alignment, baseline, font, background) were skipped."
            )
        flags = run_visual_checks(page.image, ocr) if table_layout else []
        for flag in flags:
            if _corroborated(flag, flags):
                result.findings.append(_flag_to_finding(flag, page_evidence_id, n))
            else:
                result.limitations.append(
                    f"Page {n}: weak indicator, not corroborated by a second check and not "
                    f"used for the verdict — [{flag.check}] {flag.description}"
                )

    if not any(f.outcome != Outcome.BENIGN for f in result.findings):
        result.findings.append(
            Finding(
                check="overall",
                description="No quality, OCR-confidence or visual-check issues found.",
                outcome=Outcome.BENIGN,
                evidence=[EvidenceRef(evidence_id=doc_evidence_id, kind="document", locator={})],
                uncertainty=Uncertainty(
                    confidence=0.6, notes="absence of flags is not proof of authenticity"
                ),
            )
        )
    result.limitations.append(
        "Visual checks are explainable heuristics, not a trained classifier; "
        "confidence scores are module-local and uncalibrated."
    )
    result.limitations.append(
        "The arithmetic check cannot catch an edit that changes an amount and the "
        "total consistently; visual checks are the only signal for that case."
    )
    result.needs_human_review = True  # always, per project scope guardrail
    return result
