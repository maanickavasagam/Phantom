"""MIRAGE interpretable visual checks (Day 9, reworked Week 3).

Simple, explainable heuristics (not a trained model). Each check inspects the
page image and OCR words and raises a flag with a plain-language reason and
the exact region it concerns.

Week 3 changes, after the Day 11 evaluation:
- Checks compare *amount fields* with each other (see mirage.fields) instead
  of every token containing a digit. Comparing "INV-5561" or a hyphen against
  prices was the main source of false positives.
- font_consistency uses stroke width (ink area / half the contour length),
  which is roughly independent of which characters are in the word, instead
  of raw ink density, which mostly measured the characters themselves.
- New row_baseline check: an amount that doesn't sit on its row's baseline.
- region_inconsistency compares both background level *and* background noise
  around each field to the rest of the page: on a scanned page a pasted patch
  tends to be the wrong shade or suspiciously clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from phantom.mirage.fields import AmountField, RowKind, extract_amounts
from phantom.mirage.ocr import OcrResult, Word


@dataclass
class VisualFlag:
    check: str
    description: str
    bbox: tuple[int, int, int, int]  # left, top, width, height
    severity: float  # 0-1, heuristic


def _robust_z(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = mad * 1.4826
    return (values - median) / (scale if scale > 1e-6 else 1e-6), median, scale


# ---------------------------------------------------------------- alignment


def check_field_alignment(
    amounts: list[AmountField], min_fields: int = 4, char_tolerance: float = 1.5
) -> list[VisualFlag]:
    """Amounts in a column are right-aligned; flag one whose right edge is far
    from where the rest of the column ends. The tolerance is the larger of a
    robust z-score threshold and 1.5 character widths, so a perfectly aligned
    column (MAD = 0) doesn't make OCR jitter look like tampering.

    Only line items are compared with each other: SUBTOTAL/TAX/TOTAL rows are
    often styled differently (bolder, larger), so their right edge legitimately
    sits elsewhere. Those rows are still covered by numeric_consistency and
    row_baseline."""
    column = [a for a in amounts if a.kind == RowKind.ITEM]
    if len(column) < min_fields:
        return []
    char_w = float(np.median([a.word.width / max(1, len(a.word.text)) for a in column]))
    rights = np.array([a.word.left + a.word.width for a in column], dtype=float)
    _, median, scale = _robust_z(rights)
    tolerance = max(4.0 * scale, char_tolerance * char_w)
    flags = []
    for a, r in zip(column, rights, strict=True):
        shift = r - median
        if abs(shift) > tolerance:
            flags.append(
                VisualFlag(
                    check="field_alignment",
                    description=(
                        f"Amount '{a.word.text}' ({a.label or 'unlabelled row'}) ends at "
                        f"x={r:.0f}, {abs(shift):.0f}px {'right' if shift > 0 else 'left'} "
                        f"of the amount column's edge (x={median:.0f})"
                    ),
                    bbox=a.word.bbox,
                    severity=min(1.0, 0.4 + abs(shift) / 80.0),
                )
            )
    return flags


def check_row_baseline(amounts: list[AmountField], min_offset_px: float = 6.0) -> list[VisualFlag]:
    """An amount and its row label are printed on the same line. Flag amounts
    whose vertical offset from their label differs from the page's typical
    offset. Using the typical offset (median over rows) as the reference
    cancels any residual page tilt, which shifts every row by the same amount."""
    labelled = [a for a in amounts if a.label_top is not None]
    if not labelled:
        return []
    offsets = [a.word.top - a.label_top for a in labelled]
    typical = float(np.median(offsets)) if len(labelled) >= 3 else 0.0
    flags = []
    for a, raw_dy in zip(labelled, offsets, strict=True):
        dy = raw_dy - typical
        tolerance = max(min_offset_px, 0.2 * a.word.height)
        if abs(dy) > tolerance:
            flags.append(
                VisualFlag(
                    check="row_baseline",
                    description=(
                        f"Amount '{a.word.text}' sits {abs(dy):.0f}px "
                        f"{'lower' if dy > 0 else 'higher'} relative to its row label "
                        f"'{a.label}' than the other rows do — the two should share a baseline"
                    ),
                    bbox=a.word.bbox,
                    severity=min(1.0, 0.4 + abs(dy) / 25.0),
                )
            )
    return flags


# --------------------------------------------------------------------- font


def _ink_mask(patch: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def stroke_width(gray: np.ndarray, w: Word) -> float | None:
    """Mean stroke thickness in px: 2 * ink area / total contour length."""
    patch = gray[w.top:w.top + w.height, w.left:w.left + w.width]
    if patch.size == 0 or patch.std() < 5:
        return None
    mask = _ink_mask(patch)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    if perimeter <= 0:
        return None
    return float(2.0 * (mask > 0).sum() / perimeter)


def check_font_consistency(
    gray: np.ndarray, amounts: list[AmountField], min_fields: int = 4, max_rel_dev: float = 0.22
) -> list[VisualFlag]:
    """Flag an amount whose stroke width differs from the other amounts by more
    than max_rel_dev (22%). Relative, not z-score: the amounts on a clean page
    are so uniform that a z-score would flag a slightly bolder TOTAL."""
    measured = [(a, stroke_width(gray, a.word)) for a in amounts]
    measured = [(a, s) for a, s in measured if s is not None]
    if len(measured) < min_fields:
        return []
    median = float(np.median([s for _, s in measured]))
    flags = []
    for a, s in measured:
        rel = (s - median) / median
        if abs(rel) > max_rel_dev:
            flags.append(
                VisualFlag(
                    check="font_consistency",
                    description=(
                        f"Amount '{a.word.text}' is printed with {abs(rel):.0%} "
                        f"{'thicker' if rel > 0 else 'thinner'} strokes ({s:.1f}px) than the "
                        f"other amounts ({median:.1f}px) — consistent with a different font or "
                        "renderer"
                    ),
                    bbox=a.word.bbox,
                    severity=min(1.0, 0.3 + abs(rel)),
                )
            )
    return flags


# --------------------------------------------------------------- background


def _ring_stats(gray: np.ndarray, w: Word, pad: int) -> tuple[float, float] | None:
    """Median level and noise (std) of background pixels in a ring around a word."""
    h, w_img = gray.shape[:2]
    x0, y0 = max(0, w.left - pad), max(0, w.top - pad)
    x1, y1 = min(w_img, w.left + w.width + pad), min(h, w.top + w.height + pad)
    outer = gray[y0:y1, x0:x1]
    if outer.size == 0:
        return None
    ring = np.ones(outer.shape, bool)
    ring[w.top - y0:w.top - y0 + w.height, w.left - x0:w.left - x0 + w.width] = False
    # Exclude ink *and* its anti-aliased fringe, otherwise glyph edges dominate
    # the "noise" estimate on a perfectly clean page.
    paper = float(np.median(outer))
    ink = cv2.dilate((outer < paper - 40).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    keep = ring & ~ink
    if int(keep.sum()) < 30:
        return None
    level = float(np.median(outer[keep]))
    # Texture = fine-grained high-pass residual (pixel minus its 5x5 blur). Unlike a
    # MAD of raw integer pixels this isn't quantised, and it ignores slow shading.
    f = outer.astype(np.float32)
    residual = f - cv2.blur(f, (5, 5))
    noise = float(np.std(residual[keep]))
    return level, noise


def check_region_inconsistency(
    gray: np.ndarray, ocr: OcrResult, amounts: list[AmountField], pad: int = 6
) -> list[VisualFlag]:
    """Compare the background right around each amount with the background
    around all other words on the page. Two tells of a pasted/whited-out patch:
    - level: noticeably lighter/darker than the paper elsewhere;
    - texture: on a page with visible scan noise, a patch that is much cleaner
      (lower noise) than everywhere else.
    On a perfectly clean digital page neither tell exists, so the check stays
    quiet there by design."""
    reference = [s for w in ocr.words if (s := _ring_stats(gray, w, pad)) is not None]
    if len(reference) < 8:
        return []
    ref_level = float(np.median([lv for lv, _ in reference]))
    ref_noise = float(np.median([n for _, n in reference]))
    flags = []
    for a in amounts:
        stats = _ring_stats(gray, a.word, pad)
        if stats is None:
            continue
        level, noise = stats
        reasons, severity = [], 0.0
        if abs(level - ref_level) > 10:
            reasons.append(
                f"background around it is {'lighter' if level > ref_level else 'darker'} "
                f"({level:.0f}) than the paper elsewhere ({ref_level:.0f})"
            )
            severity = max(severity, min(1.0, 0.4 + abs(level - ref_level) / 40.0))
        if ref_noise >= 0.8 and noise < 0.45 * ref_noise:
            reasons.append(
                f"background is unusually clean (noise {noise:.1f} vs {ref_noise:.1f} elsewhere)"
            )
            severity = max(severity, min(1.0, 0.4 + (1 - noise / ref_noise) / 2))
        if reasons:
            flags.append(
                VisualFlag(
                    check="region_inconsistency",
                    description=f"Amount '{a.word.text}': " + "; ".join(reasons)
                    + " — consistent with a pasted or whited-out patch",
                    bbox=a.word.bbox,
                    severity=severity,
                )
            )
    return flags


MIN_TABLE_ITEMS = 3


def table_checks_applicable(amounts: list[AmountField]) -> bool:
    """The layout checks below assume an invoice-style table: at least
    MIN_TABLE_ITEMS labelled line items, one amount per row. On the Find it
    again! train split (real receipts: QTY x PRICE = AMOUNT rows, mixed fonts,
    handwriting) they fired on genuine receipts about as often as on forged
    ones, i.e. they carried no signal there, so they are skipped on layouts
    they weren't designed for."""
    from phantom.mirage.consistency import single_column_items

    items = [a for a in amounts if a.kind == RowKind.ITEM]
    return len(items) >= MIN_TABLE_ITEMS and single_column_items(items)


def run_visual_checks(gray: np.ndarray, ocr: OcrResult) -> list[VisualFlag]:
    amounts = extract_amounts(ocr)
    if not table_checks_applicable(amounts):
        return []
    return [
        *check_field_alignment(amounts),
        *check_row_baseline(amounts),
        *check_region_inconsistency(gray, ocr, amounts),
        *check_font_consistency(gray, amounts),
    ]
