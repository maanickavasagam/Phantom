"""MIRAGE ingestion and normalization (Days 6-7).

Loads PDF pages or images, runs quality checks, corrects rotation and resizes
to a standard working width. OCR (Day 8) builds on the pages returned here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_PDF_PAGES = 20
TARGET_WIDTH = 1700  # roughly A4 at 200 DPI
MIN_WIDTH = 600
BLUR_THRESHOLD = 60.0  # variance of Laplacian below this = likely blurry


class IngestError(ValueError):
    pass


@dataclass
class QualityReport:
    width: int
    height: int
    blur_score: float
    mean_brightness: float
    issues: list[str] = field(default_factory=list)


@dataclass
class Page:
    page_number: int
    image: np.ndarray  # grayscale, normalized
    rotation_corrected_deg: float
    quality: QualityReport


def load_images(path: str | Path) -> list[np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise IngestError(f"file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        doc = fitz.open(path)
        if doc.page_count > MAX_PDF_PAGES:
            raise IngestError(f"PDF has {doc.page_count} pages; limit is {MAX_PDF_PAGES}")
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            code = cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR
            images.append(cv2.cvtColor(arr, code))
        return images
    if ext in SUPPORTED_IMAGES:
        img = cv2.imread(str(path))
        if img is None:
            raise IngestError(f"could not decode image: {path}")
        return [img]
    raise IngestError(f"unsupported file type: {ext}")


def quality_check(gray: np.ndarray) -> QualityReport:
    h, w = gray.shape[:2]
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    issues = []
    if w < MIN_WIDTH:
        issues.append("low_resolution")
    if blur < BLUR_THRESHOLD:
        issues.append("possibly_blurry")
    if brightness < 60:
        issues.append("too_dark")
    elif brightness > 250:
        issues.append("nearly_blank")
    return QualityReport(w, h, round(blur, 2), round(brightness, 2), issues)


def estimate_skew(gray: np.ndarray) -> float:
    """Estimate small skew angle (degrees) from text pixels using minAreaRect."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle > 45:
        angle -= 90
    return float(angle)


def deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def normalize(img: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    angle = estimate_skew(gray)
    gray = deskew(gray, angle)
    h, w = gray.shape
    if w != TARGET_WIDTH:
        scale = TARGET_WIDTH / w
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, (TARGET_WIDTH, int(h * scale)), interpolation=interp)
    return gray, angle


def ingest(path: str | Path) -> list[Page]:
    pages = []
    for i, img in enumerate(load_images(path), start=1):
        raw_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        quality = quality_check(raw_gray)
        norm, angle = normalize(img)
        pages.append(Page(i, norm, round(angle, 2), quality))
    return pages
