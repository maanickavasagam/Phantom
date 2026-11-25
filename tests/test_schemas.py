import pytest
from pydantic import ValidationError

from phantom.common.schemas import (
    EvidenceRef,
    Finding,
    ModuleName,
    Outcome,
    ToolInput,
    ToolResult,
    Uncertainty,
)


def test_finding_requires_evidence():
    with pytest.raises(ValidationError):
        Finding(check="x", description="x", outcome=Outcome.BENIGN, evidence=[],
                uncertainty=Uncertainty(confidence=0.5))


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        Uncertainty(confidence=1.5)


def test_timeout_bounds():
    with pytest.raises(ValidationError):
        ToolInput(module=ModuleName.MIRAGE, timeout_s=0)


def test_result_defaults_to_human_review():
    r = ToolResult(module=ModuleName.ECHO)
    assert r.needs_human_review and r.run_id.startswith("run_")


def test_valid_finding_roundtrip():
    f = Finding(check="alignment", description="ok", outcome=Outcome.INCONCLUSIVE,
                evidence=[EvidenceRef(evidence_id="ev_1", kind="page_region",
                                      locator={"page": 1})],
                uncertainty=Uncertainty(confidence=0.4))
    assert Finding.model_validate_json(f.model_dump_json()) == f
