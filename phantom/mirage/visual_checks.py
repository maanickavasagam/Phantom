"""MIRAGE interpretable visual checks (Day 9).

These are simple, explainable heuristics (not a trained model): each
check inspects the page image and OCR words and raises a flag with a
plain-language reason and the exact region it concerns. They are meant
to catch the kinds of tampering in phantom/mirage/make_samples.py:
altered amounts, misaligned fields and font mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phantom.mirage.ocr import OcrResult, Word


@dataclass
class VisualFlag:
    check: str
    description: str
    bbox: tuple[int, int, int, int]  # left, top, width, height
    severity: float  # 0-1, heuristic


def _row_groups(words: list[Word], y_tolerance: int = 12) -> list[list[Word]]:
    """Group words into text rows by vertical position."""
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.top, w.left)):
        placed = False
        for row in rows:
            if abs(row[0].top - w.top) <= y_tolerance:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])
    return rows


def check_field_alignment(ocr: OcrResult, min_rows: int = 5) -> list[VisualFlag]:
    """Flag numeric/amount-like words whose LEFT edge (where a right-aligned
    amount column starts) is a statistical outlier versus the rest of the
    column. Uses a robust (MAD-based) z-score so natural digit-count
    variation in a left-anchored column doesn't trip the check; genuine
    columns are right-aligned, so comparing left edges of right-most
    numeric tokens is the stable signal."""
    rows = _row_groups(ocr.words)
    amount_lefts = []
    for row in rows:
        numeric = [w for w in row if any(c.isdigit() for c in w.text)]
        if numeric:
            rightmost = max(numeric, key=lambda w: w.left + w.width)
            amount_lefts.append(rightmost)
    if len(amount_lefts) < min_rows:
        return []
    lefts = np.array([w.left for w in amount_lefts], dtype=float)
    median = float(np.median(lefts))
    mad = float(np.median(np.abs(lefts - median))) or 1e-6
    flags = []
    for w in amount_lefts:
        z = abs(w.left - median) / (mad * 1.4826)
        if z > 4.0:
            flags.append(
                VisualFlag(
                    check="field_alignment",
                    description=(
                        f"Amount '{w.text}' starts at x={w.left}, "
                        f"a statistical outlier vs. the column's median x={median:.0f} (z={z:.1f})"
                    ),
                    bbox=w.bbox,
                    severity=min(1.0, z / 12.0),
                )
            )
    return flags


def check_region_inconsistency(gray: np.ndarray, ocr: OcrResult, pad: int = 4) -> list[VisualFlag]:
    """Flag words whose local background doesn't match the page's dominant
    background (evidence of a pasted-over or white-boxed region)."""
    page_bg = float(np.median(gray))
    flags = []
    h, w_img = gray.shape[:2]
    for w in ocr.words:
        x0, y0 = max(0, w.left - pad), max(0, w.top - pad)
        x1, y1 = min(w_img, w.left + w.width + pad), min(h, w.top + w.height + pad)
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        corner_pixels = np.concatenate(
            [patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]]
        )
        local_bg = float(np.median(corner_pixels))
        if abs(local_bg - page_bg) > 25:
            flags.append(
                VisualFlag(
                    check="region_inconsistency",
                    description=(
                        f"Background near '{w.text}' (median {local_bg:.0f}) differs from "
                        f"page background (median {page_bg:.0f})"
                    ),
                    bbox=w.bbox,
                    severity=min(1.0, abs(local_bg - page_bg) / 100.0),
                )
            )
    return flags


def check_font_consistency(gray: np.ndarray, ocr: OcrResult) -> list[VisualFlag]:
    """Flag words whose stroke weight (mean ink density inside the glyph
    box) is an outlier versus the page's other words — a cheap proxy for
    a different font or rendering being pasted in."""
    densities = []
    for w in ocr.words:
        x0, y0 = w.left, w.top
        x1, y1 = w.left + w.width, w.top + w.height
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        ink = float((patch < 128).mean())
        densities.append((w, ink))
    if len(densities) < 5:
        return []
    values = np.array([d for _, d in densities])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) or 1e-6
    flags = []
    for w, ink in densities:
        z = abs(ink - median) / (mad * 1.4826)
        if z > 3.5:
            flags.append(
                VisualFlag(
                    check="font_consistency",
                    description=(
                        f"Ink density for '{w.text}' ({ink:.2f}) is a statistical outlier "
                        f"vs. the page median ({median:.2f})"
                    ),
                    bbox=w.bbox,
                    severity=min(1.0, z / 10.0),
                )
            )
    return flags


def run_visual_checks(gray: np.ndarray, ocr: OcrResult) -> list[VisualFlag]:
    return [
        *check_field_alignment(ocr),
        *check_region_inconsistency(gray, ocr),
        *check_font_consistency(gray, ocr),
    ]
