"""Coordinator interface support: live events, stop, verdict summary, keyword
routing, evidence upload / recent list and the NDJSON stream endpoint."""

import json

import pytest
from fastapi.testclient import TestClient

from phantom.common.evidence_store import EvidenceStore
from phantom.common.schemas import (
    EvidenceRef,
    Finding,
    ModuleName,
    Outcome,
    ToolResult,
    Uncertainty,
)
from phantom.coordinator import engine
from phantom.coordinator.engine import coordinate


def _result(module, outcome=Outcome.INCONCLUSIVE):
    return ToolResult(module=ModuleName(module), evidence_ids=["ev_0000000000000000"], findings=[
        Finding(check="fake", description="x", outcome=outcome,
                evidence=[EvidenceRef(evidence_id="ev_0000000000000000", kind="t")],
                uncertainty=Uncertainty(confidence=0.5))])


@pytest.fixture
def store(tmp_path):
    return EvidenceStore(tmp_path / "ev")


@pytest.fixture
def ids(store, monkeypatch):
    monkeypatch.setattr(engine, "analyze_document",
                        lambda *a, **k: _result("mirage", Outcome.SUSPICIOUS))
    monkeypatch.setattr(engine, "analyze_audio", lambda *a, **k: _result("echo", Outcome.BENIGN))
    return (store.put(b"doc", kind="document", source="a.pdf"),
            store.put(b"aud", kind="audio", source="a.wav"))


@pytest.mark.parametrize("goal,expected", [
    ("Check this invoice", ["mirage"]),  # "invoice" contains "voice": must not run ECHO
    ("Is this call a scam?", ["echo"]),
    ("check the calls and the bills", ["mirage", "echo"]),
    ("is this website real", ["blackbox"]),
    ("investigate", ["mirage", "echo", "blackbox"]),
])
def test_keyword_routing_uses_whole_words(store, ids, goal, expected):
    r = coordinate(goal, store, document_id=ids[0], audio_id=ids[1], website_url="flipkarrt.com")
    assert r["plan"] == expected


def test_progress_events_in_order(store, ids):
    events = []
    coordinate("investigate", store, document_id=ids[0], audio_id=ids[1],
               website_url="flipkarrt.com", on_event=events.append)
    kinds = [(e["type"], e.get("module") or e.get("step", {}).get("module")) for e in events]
    assert kinds[0] == ("plan", None)
    assert kinds[1:] == [("start", "mirage"), ("step", "mirage"), ("start", "echo"),
                         ("step", "echo"), ("start", "blackbox"), ("step", "blackbox")]


def test_stop_skips_remaining_tools(store, ids):
    calls = []
    r = coordinate("investigate", store, document_id=ids[0], audio_id=ids[1],
                   website_url="flipkarrt.com", on_event=calls.append,
                   should_stop=lambda: any(e["type"] == "step" for e in calls))
    assert [t["status"] for t in r["trace"]] == ["completed", "skipped", "skipped"]
    assert r["trace"][1]["reason"] == "stopped by user" and r["stop_reason"] == "stopped_by_user"
    assert r["status"] == "partial"


def test_broken_listener_does_not_break_the_run(store, ids):
    def boom(event):
        raise RuntimeError("ui gone")

    r = coordinate("check this invoice", store, document_id=ids[0], on_event=boom)
    assert r["status"] == "completed"


def test_report_carries_each_modules_own_verdict(store, ids):
    r = coordinate("investigate", store, document_id=ids[0], audio_id=ids[1],
                   website_url="flipkarrt.com")
    assert r["verdicts"] == [{"module": "mirage", "verdict": "suspicious"},
                             {"module": "echo", "verdict": "benign"},
                             {"module": "blackbox", "verdict": "likely_fake", "brand": "flipkart"}]


# ------------------------------------------------------------------ API
@pytest.fixture
def client(tmp_path, monkeypatch):
    from phantom.api import main

    monkeypatch.setattr(main, "store", EvidenceStore(tmp_path / "api_ev"))
    return TestClient(main.app), main


def test_upload_validates_kind_type_and_size(client):
    c, main = client
    ok = c.post("/evidence/upload?kind=document", files={"file": ("bill.pdf", b"%PDF-1.4")})
    assert ok.status_code == 200 and ok.json()["evidence_id"].startswith("ev_")
    assert main.store.get(ok.json()["evidence_id"])[1]["kind"] == "document"
    assert c.post("/evidence/upload?kind=document",
                  files={"file": ("x.exe", b"MZ")}).status_code == 400
    assert c.post("/evidence/upload?kind=audio",
                  files={"file": ("bill.pdf", b"%PDF")}).status_code == 400
    assert c.post("/evidence/upload?kind=secret",
                  files={"file": ("a.wav", b"RIFF")}).status_code == 400
    assert c.post("/evidence/upload?kind=audio",
                  files={"file": ("a.wav", b"")}).status_code == 400
    up = c.post("/evidence/upload?kind=audio", files={"file": ("../../evil.wav", b"RIFF")})
    assert up.json()["source"] == "evil.wav"  # path parts stripped from the name


def test_recent_lists_only_the_requested_kind(client):
    c, main = client
    d = main.store.put(b"d1", kind="document", source="d1.png")
    main.store.put(b"a1", kind="audio", source="a1.wav")
    items = c.get("/evidence/recent?kind=document").json()["items"]
    assert [i["evidence_id"] for i in items] == [d]
    assert c.get("/evidence/recent?kind=other").status_code == 400


def test_stream_endpoint_emits_ndjson_and_report(client):
    c, _ = client
    r = c.post("/coordinator/stream", json={"goal": "is this link real",
                                            "website_url": "https://amaz0n-offers.shop/"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert events[0]["type"] == "started" and events[0]["run"].startswith("crun_")
    assert [e["type"] for e in events[1:]] == ["plan", "start", "step", "report"]
    report = events[-1]["report"]
    assert report["verdicts"][0]["verdict"] == "likely_fake"
    assert c.post(f"/coordinator/stop/{events[0]['run']}").status_code == 404  # already over


def test_stream_rejects_bad_input_before_starting(client):
    c, _ = client
    assert c.post("/coordinator/stream", json={"goal": ""}).status_code == 422
    assert c.post("/coordinator/stream", json={"goal": "x", "max_seconds": 0}).status_code == 422


@pytest.mark.parametrize("goal", ["Was the person who rang me trying to trick me?",
                                  "Has anyone changed the amounts on this?",
                                  "Should I trust this?"])
def test_goal_naming_no_evidence_checks_everything_supplied(store, ids, goal):
    r = coordinate(goal, store, document_id=ids[0], audio_id=ids[1])
    assert r["routing"] == "no_evidence_named" and r["plan"] == ["mirage", "echo"]
    assert r["warnings"][0]["code"] == "goal_names_no_evidence"
    assert r["ignored_inputs"] == [] and r["missing_inputs"] == []


def test_keyword_goal_still_routes_narrowly(store, ids):
    r = coordinate("check this invoice", store, document_id=ids[0], audio_id=ids[1])
    assert r["routing"] == "keywords" and r["plan"] == ["mirage"]
    assert r["ignored_inputs"] == ["echo"]
    assert not any(w["code"] == "goal_names_no_evidence" for w in r["warnings"])


def test_goal_naming_no_evidence_with_nothing_supplied(store):
    r = coordinate("help me please", store)
    assert r["status"] == "nothing_to_do" and r["warnings"] == []
