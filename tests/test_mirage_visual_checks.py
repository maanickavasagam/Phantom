import json
from pathlib import Path

import pytest

from phantom.mirage.ingest import ingest
from phantom.mirage.ocr import ocr_page
from phantom.mirage.visual_checks import run_visual_checks

SAMPLES = Path("data/samples")


def _labels():
    labels_file = SAMPLES / "labels.json"
    if not labels_file.exists():
        pytest.skip("sample set not generated; run phantom.mirage.make_samples first")
    return json.loads(labels_file.read_text())


def test_most_suspicious_samples_raise_at_least_one_flag():
    """Heuristic checks aren't perfect (see docs/MIRAGE_EVAL.md); this
    tracks that they catch a meaningful majority, not every single case."""
    entries = [e for e in _labels() if e["label"] == "suspicious"][:10]
    if not entries:
        pytest.skip("sample set not generated")
    caught = 0
    for entry in entries:
        page = ingest(SAMPLES / entry["file"])[0]
        ocr = ocr_page(page.image, 1)
        if run_visual_checks(page.image, ocr):
            caught += 1
    assert caught >= 5, f"only {caught}/{len(entries)} suspicious samples raised a flag"


def test_benign_sample_runs_without_error():
    if not SAMPLES.exists():
        pytest.skip("sample set not generated")
    benign = next(e for e in _labels() if e["label"] == "benign")
    page = ingest(SAMPLES / benign["file"])[0]
    ocr = ocr_page(page.image, 1)
    # Should not raise; flags may or may not be empty (heuristics aren't perfect).
    run_visual_checks(page.image, ocr)


def test_no_words_no_flags(tmp_path):
    import cv2
    import numpy as np

    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), np.full((800, 600, 3), 255, np.uint8))
    page = ingest(p)[0]
    ocr = ocr_page(page.image, 1)
    assert run_visual_checks(page.image, ocr) == []
