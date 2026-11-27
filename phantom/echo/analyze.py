"""ECHO top-level analysis (Days 15-19).

Audio -> validated 16 kHz WAV -> Whisper transcript with word timings ->
claim records -> social-engineering indicators -> shared ToolResult.

Every finding cites transcript_segment evidence (start/end seconds, segment
ids, text) pointing into a stored transcript, which in turn is linked to the
stored original audio. The transcript and claims are returned in
ToolResult.artifacts for the report and the UI.
"""

from __future__ import annotations

import json
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
from phantom.echo import scam_classifier
from phantom.echo.claims import extract_claims
from phantom.echo.indicators import Indicator, decide, detect_indicators
from phantom.echo.ingest import IngestError, ingest
from phantom.echo.transcribe import TranscriptionError, transcribe

LOW_WORD_PROBABILITY = 0.6  # mean Whisper word probability below this: transcript unreliable

LIMITATIONS = [
    "Speech recognition (local Whisper base.en) degrades on noisy audio, overlapping speech and "
    "accents under-represented in its training data; English only in v1.",
    "Claim extraction is rule-based, not a trained classifier: indirect phrasing can be missed.",
    "No speaker diarization: the transcript does not separate caller from recipient.",
]


def _refs(transcript_id: str, ind: Indicator) -> list[EvidenceRef]:
    return [EvidenceRef(evidence_id=transcript_id, kind="transcript_segment",
                        locator={"start_s": e.start_s, "end_s": e.end_s,
                                 "segment_ids": e.segment_ids, "text": e.text})
            for e in ind.evidence] or [EvidenceRef(evidence_id=transcript_id, kind="transcript")]


def _phrase_refs(transcript_id: str, tr, phrases: list[str]) -> list[EvidenceRef]:
    """Transcript segments containing the classifier's top phrases."""
    refs = []
    for s in tr.segments:
        norm = scam_classifier.normalize(s.text)
        hit = [p for p in phrases if p in norm]
        if hit:
            refs.append(EvidenceRef(evidence_id=transcript_id, kind="transcript_segment",
                                    locator={"start_s": s.start, "end_s": s.end,
                                             "segment_ids": [s.id], "text": s.text,
                                             "phrases": hit}))
    return refs or [EvidenceRef(evidence_id=transcript_id, kind="transcript")]


def analyze_audio(path: str | Path, store: EvidenceStore, source: str | None = None) -> ToolResult:
    path = Path(path)
    result = ToolResult(module=ModuleName.ECHO)
    audio_id = store.put(path.read_bytes(), kind="audio", source=source or path.name)
    result.evidence_ids.append(audio_id)
    result.limitations.extend(LIMITATIONS)

    try:
        audio = ingest(path)
    except IngestError as e:
        result.errors.append(ToolError(code="ingest_failed", message=str(e)))
        return result
    if audio.quality.issues:
        result.limitations.append(f"Audio quality: {', '.join(audio.quality.issues)}")
    if "likely_silent" in audio.quality.issues or "empty_audio" in audio.quality.issues:
        result.errors.append(ToolError(code="no_speech", message="audio is silent or empty"))
        return result

    try:
        tr = transcribe(audio.path)
    except TranscriptionError as e:
        result.errors.append(ToolError(code="asr_failed", message=str(e), recoverable=False))
        return result

    claims = extract_claims(tr)
    transcript_id = store.put(
        json.dumps({"transcript": tr.to_dict(), "claims": [c.to_dict() for c in claims]},
                   indent=1).encode(),
        kind="transcript", source=f"{source or path.name}#transcript",
        meta={"audio_evidence_id": audio_id, "model": tr.model, "duration_s": tr.duration_s},
    )
    result.evidence_ids.append(transcript_id)
    result.artifacts = {
        "transcript": tr.to_dict(),
        "claims": [c.to_dict() for c in claims],
        "audio": {"duration_s": audio.quality.duration_s, "source_format": audio.source_format},
    }

    asr_conf = tr.mean_word_probability
    reliable = asr_conf >= LOW_WORD_PROBABILITY
    if not tr.segments:
        result.limitations.append("No speech was recognised in the recording.")
    elif not reliable:
        result.limitations.append(
            f"Transcript may be unreliable (mean word confidence {asr_conf:.2f}); "
            "indicator confidence is reduced accordingly."
        )
    scale = 1.0 if reliable else 0.6

    decision = decide(detect_indicators(tr, claims))
    strong, weak = decision.strong, decision.weak

    for ind in strong:
        result.findings.append(Finding(
            check=ind.name, description=ind.description, outcome=Outcome.SUSPICIOUS,
            evidence=_refs(transcript_id, ind),
            uncertainty=Uncertainty(confidence=round(0.85 * scale, 2),
                                    notes="rule-based indicator over the ASR transcript"),
        ))
    if decision.pattern:
        names = ", ".join(i.name for i in weak)
        result.findings.append(Finding(
            check="pressure_pattern",
            description=f"{len(weak)} different pressure tactics in one call ({names}): "
                        + " | ".join(i.description for i in weak),
            outcome=Outcome.SUSPICIOUS,
            evidence=[r for i in weak for r in _refs(transcript_id, i)],
            uncertainty=Uncertainty(confidence=round(min(0.9, 0.45 + 0.15 * len(weak)) * scale, 2),
                                    notes="combination of weak indicators; each alone is common "
                                          "in genuine calls"),
        ))
    else:
        for ind in weak:
            result.limitations.append(
                "Weak indicator on its own, not used for the verdict: "
                f"[{ind.name}] {ind.description}"
            )

    clf = scam_classifier.score(tr.text) if tr.segments else None
    if clf is not None:
        result.artifacts["classifier"] = {"probability": clf.probability,
                                          "threshold": clf.threshold,
                                          "top_phrases": clf.top_phrases}
    if clf is not None and clf.suspicious:
        phrases = ", ".join(f"'{p}'" for p, _ in clf.top_phrases[:5])
        refs = _phrase_refs(transcript_id, tr, [p for p, _ in clf.top_phrases])
        if decision.suspicious or decision.supports_classifier:
            # Classifier agrees with, or is corroborated by, rule evidence.
            outcome, note = Outcome.SUSPICIOUS, "corroborated by rule-based indicators"
        else:
            # Classifier alone: route to a human, don't accuse. On genuine Indian
            # money conversations it is not reliable on its own (docs/ECHO_EVAL.md).
            outcome, note = Outcome.INCONCLUSIVE, "classifier only, no supporting rule evidence"
        result.findings.append(Finding(
            check="ml_scam_classifier",
            description=f"Learned classifier scores this call {clf.probability:.0%} scam-like "
                        f"(threshold {clf.threshold:.0%}); phrases that raised it: {phrases}. "
                        + ("Rule evidence supports it." if outcome == Outcome.SUSPICIOUS else
                           "No rule found supporting evidence, so this is flagged for review "
                           "rather than treated as a scam."),
            outcome=outcome, evidence=refs,
            uncertainty=Uncertainty(confidence=round(clf.probability * scale, 2),
                                    notes=f"TF-IDF + logistic regression trained on public "
                                          f"scam dialogues; {note}"),
        ))
    elif clf is None:
        result.limitations.append("Learned scam classifier not available; rules only.")

    if not result.findings:
        result.findings.append(Finding(
            check="overall",
            description=f"No social-engineering pattern found in {len(tr.segments)} transcript "
                        f"segment(s); {len(claims)} claim(s) extracted for review.",
            outcome=Outcome.BENIGN,
            evidence=[EvidenceRef(evidence_id=transcript_id, kind="transcript")],
            uncertainty=Uncertainty(confidence=round(0.6 * scale, 2),
                                    notes="absence of indicators is not proof the call is genuine"),
        ))
    result.needs_human_review = True
    return result


def overall_outcome(result: ToolResult) -> Outcome:
    outcomes = {f.outcome for f in result.findings}
    if Outcome.SUSPICIOUS in outcomes:
        return Outcome.SUSPICIOUS
    if Outcome.INCONCLUSIVE in outcomes or result.errors:
        return Outcome.INCONCLUSIVE
    return Outcome.BENIGN
