from fastapi.testclient import TestClient

from phantom.api import main

client = TestClient(main.app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_mirage_analyze_and_evidence(doc_png, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", main.EvidenceStore(tmp_path / "ev"))
    r = client.post("/mirage/analyze", files={"file": ("doc.png", doc_png.read_bytes())})
    body = r.json()
    assert r.status_code == 200 and body["module"] == "mirage" and not body["errors"]
    assert body["findings"] and body["needs_human_review"] is True
    ev = client.get(f"/evidence/{body['evidence_ids'][0]}").json()
    assert ev["integrity_ok"]


def test_demo_ui_and_cases():
    assert "MIRAGE" in client.get("/").text
    cases = client.get("/demo/cases").json()["cases"]
    assert cases, "data/demo/manifest.json should list demo cases"
    assert client.get(f"/demo/cases/{cases[0]['file']}").status_code == 200
    assert client.get("/demo/cases/..%2Fmanifest.json").status_code == 404
    assert client.get("/demo/cases/does_not_exist.png").status_code == 404


def test_page_evidence_renders_and_verdict(doc_png, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", main.EvidenceStore(tmp_path / "ev"))
    body = client.post("/mirage/analyze", files={"file": ("doc.png", doc_png.read_bytes())}).json()
    doc_id, page_id = body["evidence_ids"][:2]
    img = client.get(f"/evidence/{page_id}/image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert client.get(f"/evidence/{doc_id}/image").status_code == 400  # not a rendered page
    verdict = client.post("/mirage/verdict", json=body).json()["verdict"]
    assert verdict in ("benign", "inconclusive", "suspicious")


def test_mirage_eval_endpoint():
    assert isinstance(client.get("/mirage/eval").json(), dict)


def test_mirage_bad_type(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", main.EvidenceStore(tmp_path / "ev"))
    body = client.post("/mirage/analyze", files={"file": ("x.docx", b"abc")}).json()
    assert body["errors"][0]["code"] == "ingest_failed"


def test_echo_analyze_valid_audio(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf

    from phantom.echo import analyze as echo_analyze
    from phantom.echo.transcribe import Segment, Transcript, Word

    monkeypatch.setattr(main, "store", main.EvidenceStore(tmp_path / "ev"))
    # Don't load Whisper in unit tests: fake the recogniser's output.
    words = [Word(w, i * 0.3, i * 0.3 + 0.25, 0.9) for i, w in enumerate("Read me the OTP".split())]
    monkeypatch.setattr(echo_analyze, "transcribe", lambda p: Transcript(
        "fake", "en", 1.0, [Segment(0, 0.0, 1.2, "Read me the OTP", -0.1, 0.0, words)]))
    wav_path = tmp_path / "clip.wav"
    t = np.linspace(0, 1.0, 44100, endpoint=False)
    sf.write(wav_path, (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 44100)
    r = client.post("/echo/analyze", files={"file": ("clip.wav", wav_path.read_bytes())})
    body = r.json()
    assert r.status_code == 200 and body["module"] == "echo" and not body["errors"]
    assert body["evidence_ids"] and body["artifacts"]["claims"][0]["type"] == "credential_request"
    assert client.post("/echo/verdict", json=body).json()["verdict"] == "suspicious"
    audio = client.get(f"/evidence/{body['evidence_ids'][0]}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"].startswith("audio/")


def test_echo_analyze_bad_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", main.EvidenceStore(tmp_path / "ev"))
    body = client.post("/echo/analyze", files={"file": ("x.ogg", b"not audio")}).json()
    assert body["errors"][0]["code"] == "ingest_failed"


def test_blackbox_mock():
    assert client.post("/blackbox/investigate").json()["module"] == "blackbox"
