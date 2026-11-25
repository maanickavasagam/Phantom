from phantom.mirage.ingest import ingest
from phantom.mirage.ocr import ocr_page, ocr_pages


def test_ocr_extracts_words_with_bbox_and_confidence(doc_png):
    page = ingest(doc_png)[0]
    result = ocr_page(page.image, page.page_number)
    assert result.words, "expected at least one recognized word"
    assert any("INVOICE" in w.text.upper() for w in result.words)
    for w in result.words:
        assert w.page_number == 1
        assert w.width > 0 and w.height > 0
        assert 0 <= w.confidence <= 100
    assert 0 <= result.mean_confidence <= 100


def test_ocr_pages_multi_page(doc_png, blurry_png):
    pages = ingest(doc_png) + [p for p in ingest(blurry_png)]
    for i, p in enumerate(pages, start=1):
        p.page_number = i
    results = ocr_pages(pages)
    assert [r.page_number for r in results] == [1, 2]


def test_blank_page_yields_no_words(tmp_path):
    import cv2
    import numpy as np

    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), np.full((800, 600, 3), 255, np.uint8))
    page = ingest(p)[0]
    result = ocr_page(page.image, page.page_number)
    assert result.words == []
    assert result.mean_confidence == 0.0
