"""The end-to-end case set is reproducible, balanced and source-separated."""

from pathlib import Path

import pytest

from phantom.common.evidence_store import EvidenceStore
from phantom.coordinator.engine import _asks_for, coordinate
from phantom.e2e_eval import FINDIT, MODS, URLS, _prf, build_cases

needs_data = pytest.mark.skipif(
    not (FINDIT / "test.txt").exists() or not (URLS / "top-1m.csv").exists(),
    reason="evaluation datasets not downloaded (data/raw is gitignored)")


@needs_data
def test_case_set_is_reproducible_balanced_and_separated():
    a, b = build_cases(), build_cases()
    assert a == b  # fixed seed
    labels = [c["label"] for c in a if c["label"]]
    assert labels.count("fraud") == labels.count("genuine")
    refs = [c["inputs"][m]["ref"] for c in a for m in c["inputs"]]
    assert len(refs) == len(set(refs))  # every source item used once
    assert all("/test/" in r or "_v1_" in r or r.startswith("http") for r in refs)  # held-out


@needs_data
def test_goal_styles_route_as_designed():
    for c in build_cases():
        routed = [m for m in MODS if _asks_for(c["goal"].lower(), m)]
        if c["goal_style"] == "paraphrase":
            assert routed == []
        elif c["goal"] != "investigate":
            assert routed == [m for m in MODS if m in c["intended"]]


def test_report_lists_ignored_inputs(tmp_path):
    store = EvidenceStore(tmp_path / "ev")
    r = coordinate("is this link real", store, website_url="flipkarrt.com",
                   audio_id=store.put(b"RIFF", kind="audio", source="a.wav"))
    assert r["plan"] == ["blackbox"] and r["ignored_inputs"] == ["echo"]


def test_prf():
    m = _prf([(True, True), (True, False), (False, True), (False, False)])
    assert (m["precision"], m["recall"], m["false_positive_rate"], m["accuracy"]) == \
        (0.5, 0.5, 0.5, 0.5)
    assert Path(__file__).exists()
