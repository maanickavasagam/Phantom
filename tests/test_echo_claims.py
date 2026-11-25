import numpy as np
import pytest
import soundfile as sf

from phantom.common.evidence_store import EvidenceStore
from phantom.echo import analyze as echo_analyze
from phantom.echo.claims import extract_claims, parse_number
from phantom.echo.evaluate import match_claims, normalize_text
from phantom.echo.indicators import detect_indicators
from phantom.echo.transcribe import Segment, Transcript, Word


def make_transcript(*sentences, split_numbers=True):
    """Build a Transcript with word timings; optionally split '25,000' the way
    Whisper tokenises it ('25' + ',000')."""
    segs, t = [], 0.0
    for i, s in enumerate(sentences):
        words = []
        for w in s.split():
            parts = [w]
            if split_numbers and "," in w and w[0].isdigit():
                head, tail = w.split(",", 1)
                parts = [head, "," + tail]
            for p in parts:
                words.append(Word(p, t, t + 0.3, 0.95))
                t += 0.35
        segs.append(Segment(i, words[0].start, words[-1].end, s, -0.2, 0.01, words))
    return Transcript("test", "en", t, segs)


def keys(claims, type_=None):
    return sorted((c.type, c.key) for c in claims if type_ is None or c.type == type_)


@pytest.mark.parametrize("text,value", [
    ("50,000", 50000), ("fifty thousand", 50000), ("2 lakh", 200000), ("five lakh", 500000),
    ("1.5 crore", 15_000_000), ("twenty-five", 25), ("hello", None),
])
def test_parse_number(text, value):
    assert parse_number(text) == value


def test_scam_call_claims():
    tr = make_transcript(
        "Hello, this is Officer Sharma calling from the Cyber Crime Department.",
        "Your bank account will be blocked within the next two hours.",
        "To avoid arrest you must transfer 50,000 rupees by UPI immediately,",
        "and please read me the OTP you just received.")
    got = keys(extract_claims(tr))
    for expected in [("amount", "50000"), ("credential_request", "otp"), ("deadline", "2 hour"),
                     ("deadline", "immediately"), ("name", "sharma"),
                     ("name", "cyber crime department"), ("payment_request", "upi")]:
        assert expected in got


def test_whisper_split_numbers_are_rejoined():
    tr = make_transcript("You must pay a fee of 1,200 rupees today itself.")
    assert ("amount", "1200") in keys(extract_claims(tr))


def test_warning_about_otp_is_not_a_credential_request():
    tr = make_transcript("Please remember, we will never ask you to share your OTP or PIN.")
    assert keys(extract_claims(tr), "credential_request") == []


def test_name_needs_capitalised_word():
    tr = make_transcript("I am going to the market, this is urgent.")
    assert keys(extract_claims(tr), "name") == []


def test_unusual_channel_and_dedup():
    tr = make_transcript("Buy Google Play gift cards worth 5,000 rupees.",
                         "Send me the gift cards codes now.")
    pays = [c for c in extract_claims(tr) if c.type == "payment_request"]
    assert [c.key for c in pays] == ["gift card"] and pays[0].attrs["unusual_channel"]


def test_indicators_need_manipulation_for_pattern():
    bill = make_transcript("Your insurance premium of 20,000 rupees is due within thirty days.",
                           "You can pay through net banking.")
    names = {i.name for i in detect_indicators(bill, extract_claims(bill))}
    assert "authority" not in names and "threat" not in names


def test_identity_details_are_weak_and_own_card_is_not_a_request():
    tr = make_transcript("Can you give me your account number so I can look into it?",
                         "Sure, but what are the last four digits of my credit card number?")
    asks = [c for c in extract_claims(tr) if c.type == "credential_request"]
    assert [(c.key, c.attrs["level"]) for c in asks] == [("account number", "identity")]
    assert "credential_request" not in {i.name for i in detect_indicators(tr, asks)}


def test_im_fine_is_not_a_threat_and_wont_is_not_a_prize():
    tr = make_transcript("I'm fine, thanks.", "You won't be charged anything for the delivery.")
    names = {i.name for i in detect_indicators(tr, extract_claims(tr))}
    assert "threat" not in names and "lure" not in names


def test_us_tech_support_scam_is_suspicious():
    from phantom.echo.indicators import decide
    tr = make_transcript("This is Microsoft support, your computer has been hacked by a virus.",
                         "I need access to your computer right now to fix it.")
    assert decide(detect_indicators(tr, extract_claims(tr))).suspicious


def test_match_claims_lenient_names_only():
    tp, fp, fn = match_claims([("name", "varma"), ("amount", "500")],
                              [("name", "verma"), ("amount", "5000")])
    assert tp == [("name", "varma")] and fp == [("amount", "500")] and fn == [("amount", "5000")]


def test_normalize_text():
    assert normalize_text("Pay 50,000 rupees by five p.m.!") == "pay 50000 rupees by 5 pm"


def test_classifier_model_loads_and_explains():
    from phantom.echo import scam_classifier

    s = scam_classifier.score("I need to verify your social security number, there is a refund "
                              "of $500 and a processing fee, please buy gift cards.")
    assert s is not None and 0.0 <= s.probability <= 1.0 and s.suspicious
    assert s.top_phrases and all(w > 0 for _, w in s.top_phrases)
    assert scam_classifier.normalize("Suspect: pay $500 now") == "pay <amount> now"


@pytest.mark.parametrize("sentences,expected", [
    # classifier alone, no rule evidence -> review, not accusation
    (["Your premium is due this month, you can pay online."], "inconclusive"),
    # classifier + an authority claim from the rules -> suspicious
    (["This is Officer Rao from the Mumbai Police.", "We need to talk about your account."],
     "suspicious"),
])
def test_classifier_needs_rule_support_to_accuse(tmp_path, monkeypatch, sentences, expected):
    from phantom.echo import scam_classifier

    monkeypatch.setattr(echo_analyze, "ingest", lambda p: type("A", (), {
        "path": p, "source_format": ".wav",
        "quality": type("Q", (), {"issues": [], "duration_s": 1.0})()})())
    monkeypatch.setattr(echo_analyze, "transcribe", lambda p: make_transcript(*sentences))
    monkeypatch.setattr(scam_classifier, "score",
                        lambda text, top_k=6: scam_classifier.Score(0.9, 0.33, [("account", 0.4)]))
    res = echo_analyze.analyze_audio(_fake_audio(tmp_path), EvidenceStore(tmp_path / "ev"))
    assert echo_analyze.overall_outcome(res).value == expected
    assert any(f.check == "ml_scam_classifier" for f in res.findings)


def _fake_audio(tmp_path):
    p = tmp_path / "call.wav"
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sf.write(p, (0.2 * np.sin(2 * np.pi * 300 * t)).astype("float32"), 16000)
    return p


@pytest.mark.parametrize("sentences,verdict", [
    (["This is Inspector Verma from the Mumbai Police.", "There is an arrest warrant against you.",
      "Transfer 50,000 rupees within two hours."], "suspicious"),
    (["Hi, this is the City Clinic reception.", "Your appointment is on Monday at ten."], "benign"),
])
def test_analyze_audio_end_to_end(tmp_path, monkeypatch, sentences, verdict):
    monkeypatch.setattr(echo_analyze, "ingest", lambda p: type("A", (), {
        "path": p, "source_format": ".wav",
        "quality": type("Q", (), {"issues": [], "duration_s": 1.0})()})())
    monkeypatch.setattr(echo_analyze, "transcribe", lambda p: make_transcript(*sentences))
    res = echo_analyze.analyze_audio(_fake_audio(tmp_path), EvidenceStore(tmp_path / "ev"))
    assert echo_analyze.overall_outcome(res).value == verdict
    assert res.artifacts["transcript"]["segments"] and "claims" in res.artifacts
    for f in res.findings:
        assert f.evidence and all(e.evidence_id in res.evidence_ids for e in f.evidence)
