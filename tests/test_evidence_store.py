import pytest

from phantom.common.evidence_store import EvidenceStore


def test_put_get_verify(tmp_path):
    s = EvidenceStore(tmp_path)
    eid = s.put(b"hello", kind="document", source="t.pdf")
    data, rec = s.get(eid)
    assert data == b"hello" and rec["source"] == "t.pdf" and s.verify(eid)


def test_same_content_same_id(tmp_path):
    s = EvidenceStore(tmp_path)
    assert s.put(b"a", "x", "1") == s.put(b"a", "x", "2")


def test_tamper_detected(tmp_path):
    s = EvidenceStore(tmp_path)
    eid = s.put(b"orig", "document", "t")
    (tmp_path / eid / "blob").write_bytes(b"changed")
    assert not s.verify(eid)


def test_missing(tmp_path):
    with pytest.raises(KeyError):
        EvidenceStore(tmp_path).get("ev_nope")
