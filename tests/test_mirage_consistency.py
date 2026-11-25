from decimal import Decimal

import cv2
import numpy as np

from phantom.mirage.consistency import check_numeric_consistency
from phantom.mirage.fields import RowKind, extract_amounts, parse_amount
from phantom.mirage.ingest import estimate_skew
from phantom.mirage.ocr import OcrResult, Word
from phantom.mirage.visual_checks import check_field_alignment, check_row_baseline


def _ocr(rows, conf=95.0, amount_right=1300, row_dy=None):
    """rows: [(label, amount_text)], laid out like an invoice table."""
    words, top = [], 300
    for i, (label, amount) in enumerate(rows):
        x = 100
        for part in label.split():
            words.append(Word(part, 1, x, top, 18 * len(part), 26, conf))
            x += 18 * len(part) + 15
        dy = row_dy[i] if row_dy else 0
        width = 18 * len(amount)
        words.append(Word(amount, 1, amount_right - width, top + dy, width, 26, conf))
        top += 80
    return OcrResult(1, words, conf, " ".join(w.text for w in words))


def test_parse_amount():
    assert parse_amount("1,234.50") == Decimal("1234.50")
    assert parse_amount("$99.00") == Decimal("99.00")
    assert parse_amount("12,000") == Decimal("12000")
    assert parse_amount("3") is None  # row number, not money
    assert parse_amount("INV-5561") is None
    assert parse_amount("2026") is None  # a year


def test_rows_classified():
    ocr = _ocr([("Item 1", "100.00"), ("SUBTOTAL", "100.00"), ("GST 18%", "18.00"),
                ("TOTAL", "118.00")])
    kinds = [a.kind for a in extract_amounts(ocr)]
    assert kinds == [RowKind.ITEM, RowKind.SUBTOTAL, RowKind.TAX, RowKind.TOTAL]


def test_consistent_invoice():
    ocr = _ocr([("Item 1", "1,200.00"), ("Item 2", "300.50"), ("TOTAL", "1,500.50")])
    assert check_numeric_consistency(ocr).status == "consistent"


def test_altered_item_detected():
    ocr = _ocr([("Item 1", "9,200.00"), ("Item 2", "300.50"), ("TOTAL", "1,500.50")])
    res = check_numeric_consistency(ocr)
    assert res.status == "mismatch"
    assert "off by 8,000.00" in res.description
    assert res.words  # evidence regions attached


def test_subtotal_and_tax():
    good = _ocr([("Chairs", "1,000.00"), ("Lamps", "500.00"), ("SUBTOTAL", "1,500.00"),
                 ("GST 18%", "270.00"), ("TOTAL", "1,770.00")])
    assert check_numeric_consistency(good).status == "consistent"
    bad = _ocr([("Chairs", "1,000.00"), ("Lamps", "500.00"), ("SUBTOTAL", "1,500.00"),
                ("GST 18%", "270.00"), ("TOTAL", "2,770.00")])
    assert check_numeric_consistency(bad).status == "mismatch"


def test_low_confidence_is_inconclusive_not_suspicious():
    ocr = _ocr([("Item 1", "9,200.00"), ("Item 2", "300.50"), ("TOTAL", "1,500.50")], conf=30)
    assert check_numeric_consistency(ocr).status == "unreadable"


def test_no_total_not_applicable():
    ocr = _ocr([("Item 1", "100.00"), ("Item 2", "200.00")])
    assert check_numeric_consistency(ocr).status == "not_applicable"


def test_row_baseline_flags_only_the_odd_row():
    rows = [("Item 1", "100.00"), ("Item 2", "200.00"), ("Item 3", "300.00"),
            ("TOTAL", "600.00")]
    # Every row shifted -8px (residual page tilt) except item 2, shifted +6px.
    ocr = _ocr(rows, row_dy=[-8, 6, -8, -8])
    flags = check_row_baseline(extract_amounts(ocr))
    assert [f.description.split("'")[1] for f in flags] == ["200.00"]


def test_field_alignment_flags_shifted_amount():
    rows = [(f"Item {i}", "1,000.00") for i in range(1, 6)]
    ocr = _ocr(rows)
    shifted = next(w for w in ocr.words if w.text == "1,000.00" and w.top == 300 + 2 * 80)
    shifted.left -= 40
    flags = check_field_alignment(extract_amounts(ocr))
    assert len(flags) == 1 and "40px left" in flags[0].description


def test_upright_page_is_not_rotated():
    """Regression: under OpenCV >= 4.5 the Day 7 minAreaRect estimate returned
    -90 deg for an upright page, turning every page sideways before OCR."""
    img = np.full((1000, 800), 255, np.uint8)
    for i in range(12):
        cv2.putText(img, f"LINE {i} AMOUNT 1,234.00", (60, 100 + 70 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    assert abs(estimate_skew(img)) < 0.3
