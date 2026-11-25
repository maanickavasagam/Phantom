import pymupdf as fitz
import pytest

from phantom.mirage.ingest import TARGET_WIDTH, IngestError, ingest


def test_png_normalized(doc_png):
    pages = ingest(doc_png)
    assert len(pages) == 1
    assert pages[0].image.shape[1] == TARGET_WIDTH
    assert pages[0].image.ndim == 2
    assert "possibly_blurry" not in pages[0].quality.issues


def test_skew_detected_and_corrected(skewed_png):
    page = ingest(skewed_png)[0]
    assert 2.0 < abs(page.rotation_corrected_deg) < 6.0


def test_blur_flagged(blurry_png):
    assert "possibly_blurry" in ingest(blurry_png)[0].quality.issues


def test_pdf_multi_page(tmp_path):
    p = tmp_path / "two.pdf"
    doc = fitz.open()
    for i in range(2):
        doc.new_page().insert_text((72, 72), f"Page {i + 1} invoice total 500.00", fontsize=14)
    doc.save(p)
    pages = ingest(p)
    assert [pg.page_number for pg in pages] == [1, 2]


def test_unsupported_type(tmp_path):
    p = tmp_path / "a.docx"
    p.write_bytes(b"x")
    with pytest.raises(IngestError):
        ingest(p)


def test_missing_file(tmp_path):
    with pytest.raises(IngestError):
        ingest(tmp_path / "nope.png")
