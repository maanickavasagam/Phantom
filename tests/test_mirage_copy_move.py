import cv2
import numpy as np

from phantom.common.evidence_store import EvidenceStore
from phantom.mirage.analyze import analyze_document
from phantom.mirage.copy_move import find_clones


def _scanned_receipt(seed=0, noise=True):
    rng = np.random.default_rng(seed)
    img = np.full((900, 600), 255, np.uint8)
    for i in range(14):
        cv2.putText(img, f"ITEM {i:02d}   QTY 1   {13 * i + 7}.90", (30, 50 + 58 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    if noise:
        img = np.clip(img * 0.93 + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return img


def _paste_glyph(img):
    """Copy the first digit-sized blob found on row 3 onto row 9 (CPI forgery)."""
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if 150 <= y <= 230 and x > 200 and h >= 12:
            src = img[y - 3:y + h + 3, x - 3:x + w + 3].copy()
            dy = 58 * 6
            img[y - 3 + dy:y + h + 3 + dy, x - 3:x + w + 3] = src
            return img, (int(x), int(y + dy), int(w), int(h))
    raise AssertionError("no glyph to copy")


def test_genuine_scan_has_no_clones():
    assert find_clones(_scanned_receipt()).status == "clean"


def test_pasted_glyph_is_found():
    img, pasted = _paste_glyph(_scanned_receipt())
    res = find_clones(img)
    assert res.status == "clones_found"
    boxes = [b for p in res.pairs for b in (p.a, p.b)]
    assert any(abs(b[0] - pasted[0]) <= 2 and abs(b[1] - pasted[1]) <= 2 for b in boxes)
    assert res.pairs[0].diff < 0.3


def test_born_digital_image_is_skipped_not_flagged():
    """Without scan noise identical glyphs are normal; the check must not accuse."""
    assert find_clones(_scanned_receipt(noise=False)).status == "not_applicable"


def test_copy_move_finding_in_pipeline(tmp_path):
    img, _ = _paste_glyph(_scanned_receipt())
    p = tmp_path / "receipt.png"
    cv2.imwrite(str(p), img)
    result = analyze_document(p, EvidenceStore(tmp_path / "ev"))
    finding = next(f for f in result.findings if f.check == "copy_move")
    assert finding.outcome.value == "suspicious"
    assert all("bbox" in e.locator and e.locator["role"] in ("A", "B") for e in finding.evidence)
