"""Coordinator failure and budget behaviour."""

from phantom.common.evidence_store import EvidenceStore
from phantom.coordinator import engine


def test_failed_call_consumes_budget_and_is_reported(tmp_path, monkeypatch):
    store = EvidenceStore(tmp_path / "evidence")
    document_id = store.put(b"fixture", kind="document", source="fixture.pdf")

    def broken_document(*args, **kwargs):
        raise RuntimeError("document tool unavailable")

    monkeypatch.setattr(engine, "analyze_document", broken_document)
    report = engine.coordinate(
        "Investigate document and website", store,
        document_id=document_id,
        target_url="http://127.0.0.1:8000/synthetic/fees/start",
        max_calls=1,
    )
    assert report["tool_calls"] == 1
    assert report["trace"][0]["status"] == "failed"
    assert report["trace"][1] == {
        "module": "blackbox", "status": "skipped", "reason": "call budget"
    }
    assert report["errors"][0]["module"] == "mirage"
    assert report["needs_human_review"]
