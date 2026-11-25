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
    # Grayscale page at its original resolution, before deskew/resize. Pixel-exact
    # checks (copy-move) need it: resampling destroys exact duplicates.
    original: np.ndarray | None = None
    # 2x3 affine matrix mapping original-pixel coordinates to `image` coordinates.
    to_normalized: np.ndarray | None = None

    def map_bbox(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Map an (x, y, w, h) box from original pixels to normalized-page pixels."""
        if self.to_normalized is None:
            return bbox
        x, y, w, h = bbox
        corners = np.array([[x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]], float)
        pts = corners @ self.to_normalized.T
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))


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
    ink_fraction = float((gray < 128).mean())
    if brightness < 60:
        issues.append("too_dark")
    elif ink_fraction < 0.001:
        # A normal invoice is mostly white paper (mean brightness > 250 is
        # typical), so "blank" means almost no ink, not a bright page.
        issues.append("nearly_blank")
    return QualityReport(w, h, round(blur, 2), round(brightness, 2), issues)


MAX_SKEW_DEG = 8.0


def _profile_score(binary: np.ndarray, angle: float) -> float:
    """How 'line-like' the page is when rotated by angle: text rows give sharp
    peaks in the horizontal projection, so the sum of squared row-to-row
    differences is largest when the lines are exactly horizontal."""
    h, w = binary.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    profile = rotated.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(profile) ** 2))


def estimate_skew(gray: np.ndarray) -> float:
    """Estimate small skew angle (degrees) by projection-profile search.

    Returns the rotation to pass to deskew(). Replaces the Day 7 minAreaRect
    estimate, which (a) flipped upright pages by 90 deg under OpenCV >= 4.5's
    angle convention and (b) left ~0.5 deg of residual skew on noisy scans,
    enough to shift a right-hand amount column ~8 px off its row labels.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if int((binary > 0).sum()) < 50:
        return 0.0
    scale = 800.0 / max(binary.shape)
    small = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = (small > 64).astype(np.float32)
    coarse = np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + 1e-6, 0.25)
    best = max(coarse, key=lambda a: _profile_score(small, a))
    fine = np.arange(best - 0.3, best + 0.3 + 1e-6, 0.05)
    best = max(fine, key=lambda a: _profile_score(small, a))
    return round(float(best), 2)


def deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # Fill exposed corners with the page's own paper tone, not pure white, so the
    # background checks don't see bright triangles on a grey scan.
    paper = int(np.median(gray))
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=paper)


def normalize(img: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Deskew and resize to TARGET_WIDTH. Returns (page, angle, 2x3 transform
    from original-pixel to normalized-pixel coordinates)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    angle = estimate_skew(gray)
    gray = deskew(gray, angle)
    transform = (
        cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        if abs(angle) >= 0.1 else np.array([[1.0, 0, 0], [0, 1.0, 0]])
    )
    if w != TARGET_WIDTH:
        scale = TARGET_WIDTH / w
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, (TARGET_WIDTH, int(h * scale)), interpolation=interp)
        transform = transform * scale
    return gray, angle, transform


def ingest(path: str | Path) -> list[Page]:
    pages = []
    for i, img in enumerate(load_images(path), start=1):
        raw_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        quality = quality_check(raw_gray)
        norm, angle, transform = normalize(img)
        pages.append(Page(i, norm, round(angle, 2), quality, raw_gray, transform))
    return pages
