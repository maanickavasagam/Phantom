import json
from pathlib import Path

import pytest

from phantom.common.evidence_store import EvidenceStore
from phantom.common.schemas import Outcome
from phantom.mirage.analyze import analyze_document

SAMPLES = Path("data/samples")


def _labels():
    labels_file = SAMPLES / "labels.json"
    if not labels_file.exists():
        pytest.skip("sample set not generated; run phantom.mirage.make_samples first")
    return json.loads(labels_file.read_text())


def test_every_finding_has_evidence(tmp_path, doc_png):
    store = EvidenceStore(tmp_path / "ev")
    result = analyze_document(doc_png, store)
    assert not result.errors
    for f in result.findings:
        assert f.evidence
        for ref in f.evidence:
            store.get(ref.evidence_id)  # raises if missing


def test_needs_human_review_always_true(tmp_path, doc_png):
    store = EvidenceStore(tmp_path / "ev")
    result = analyze_document(doc_png, store)
    assert result.needs_human_review is True


def test_clean_doc_without_flags_is_benign_overall(tmp_path, doc_png):
    store = EvidenceStore(tmp_path / "ev")
    result = analyze_document(doc_png, store)
    non_quality = [f for f in result.findings if f.check not in ("input_quality", "ocr_confidence")]
    if not non_quality:
        pytest.skip("no non-quality findings to check")
    assert any(f.outcome in (Outcome.BENIGN, Outcome.INCONCLUSIVE) for f in non_quality)


def test_suspicious_samples_flagged(tmp_path):
    if not SAMPLES.exists():
        pytest.skip("sample set not generated")
    store = EvidenceStore(tmp_path / "ev")
    suspicious = [e for e in _labels() if e["label"] == "suspicious"][:5]
    flagged = 0
    for entry in suspicious:
        result = analyze_document(SAMPLES / entry["file"], store)
        if any(f.outcome != Outcome.BENIGN for f in result.findings):
            flagged += 1
    msg = f"expected most suspicious samples flagged, got {flagged}/{len(suspicious)}"
    assert flagged >= 4, msg


def test_bad_file_returns_error_not_crash(tmp_path):
    store = EvidenceStore(tmp_path / "ev")
    bad = tmp_path / "bad.docx"
    bad.write_bytes(b"not a real document")
    result = analyze_document(bad, store)
    assert result.errors and result.errors[0].code == "ingest_failed"
