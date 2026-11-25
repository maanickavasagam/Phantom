"""Shared tool contract (Day 3). Every module returns a ToolResult."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ModuleName(StrEnum):
    MIRAGE = "mirage"
    ECHO = "echo"
    BLACKBOX = "blackbox"


class Outcome(StrEnum):
    SUSPICIOUS = "suspicious"
    BENIGN = "benign"
    INCONCLUSIVE = "inconclusive"


class Uncertainty(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0, description="Module-local, uncalibrated score")
    calibrated: bool = False
    notes: str | None = None


class EvidenceRef(BaseModel):
    evidence_id: str
    kind: str = Field(description="e.g. page_region, transcript_segment, screenshot, action_trace")
    locator: dict[str, Any] = Field(default_factory=dict, description="page/bbox, start/end s, url")


class Finding(BaseModel):
    finding_id: str = Field(default_factory=lambda: f"fnd_{uuid.uuid4().hex[:10]}")
    check: str
    description: str
    outcome: Outcome
    evidence: list[EvidenceRef]
    uncertainty: Uncertainty

    @field_validator("evidence")
    @classmethod
    def must_have_evidence(cls, v: list[EvidenceRef]) -> list[EvidenceRef]:
        if not v:
            raise ValueError("every finding must reference at least one evidence item")
        return v


class ToolError(BaseModel):
    code: str
    message: str
    recoverable: bool = True


class ToolInput(BaseModel):
    module: ModuleName
    input_path: str | None = None
    target_url: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = Field(default=60.0, gt=0, le=600)


class ToolResult(BaseModel):
    module: ModuleName
    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    findings: list[Finding] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    errors: list[ToolError] = Field(default_factory=list)
    needs_human_review: bool = True
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Module-specific structured output that isn't a finding, e.g. ECHO's "
        "transcript segments and extracted claim records. Optional; findings remain the contract.",
    )
