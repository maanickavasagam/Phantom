"""MIRAGE OCR integration (Day 8).

Wraps Tesseract via pytesseract, keeping per-word text, bounding box,
page number and confidence so every extracted word can be traced back
to an exact region of the source page.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytesseract

MIN_CONFIDENCE = 0  # Tesseract reports -1 for non-text blocks; we drop those


@dataclass
class Word:
    text: str
    page_number: int
    left: int
    top: int
    width: int
    height: int
    confidence: float  # 0-100, Tesseract's own estimate

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)


@dataclass
class OcrResult:
    page_number: int
    words: list[Word]
    mean_confidence: float
    text: str


def ocr_page(gray: np.ndarray, page_number: int) -> OcrResult:
    """Run OCR on a normalized grayscale page image (see mirage.ingest)."""
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    words: list[Word] = []
    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        conf = float(data["conf"][i])
        if not text or conf < MIN_CONFIDENCE:
            continue
        words.append(
            Word(
                text=text,
                page_number=page_number,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                confidence=conf,
            )
        )
    mean_conf = sum(w.confidence for w in words) / len(words) if words else 0.0
    full_text = " ".join(w.text for w in words)
    return OcrResult(
        page_number=page_number, words=words, mean_confidence=round(mean_conf, 2), text=full_text
    )


def ocr_pages(pages) -> list[OcrResult]:
    """pages: list[mirage.ingest.Page]"""
    return [ocr_page(p.image, p.page_number) for p in pages]
