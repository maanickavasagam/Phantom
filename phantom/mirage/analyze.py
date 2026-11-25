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
from phantom.mirage.ingest import IngestError, ingest
from phantom.mirage.ocr import ocr_page
from phantom.mirage.visual_checks import VisualFlag, run_visual_checks

LOW_OCR_CONFIDENCE = 40.0  # below this, treat page text as unreliable


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

        flags = run_visual_checks(page.image, ocr)
        for flag in flags:
            finding = _flag_to_finding(flag, page_evidence_id, page.page_number)
            result.findings.append(finding)

    if not result.findings:
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
    result.needs_human_review = True  # always, per project scope guardrail
    return result
