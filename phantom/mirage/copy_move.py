"""MIRAGE copy-move check (Week 3, from the "Find it again!" analysis).

Real forgers rarely retype a number in a new font. Most edits in the Find it
again! train split (CPI, "copy-paste inside") clone a digit from elsewhere on
the same receipt with Paint or GIMP. That leaves a tell that is easy to
explain: printing and scanning give every character its own pattern of ink
and scanner noise, so two genuinely printed "5"s never match pixel for pixel.
A pasted "5" is an exact clone of its source, noise included.

The check finds character-sized ink blobs on the ORIGINAL-resolution page and
flags pairs that are near-identical (mean absolute difference below
CLONE_THRESHOLD grey levels, patch margin included). The threshold sits in the
gap observed on the train split: forged clones at <= 0.15, the closest genuine
pairs at >= 0.56.

It can't tell which of the two is the source and which is the paste, and it
doesn't apply to born-digital images (no scan noise, so identical glyphs are
normal). Those are detected and skipped rather than flagged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

CLONE_THRESHOLD = 0.3  # mean abs grey-level difference; chosen on train split only
MARGIN = 2  # px of surrounding paper included, so the background noise must match too
MAX_GROUP = 600  # glyphs of one exact size compared pairwise; caps worst-case runtime
BORN_DIGITAL_CLONE_FRACTION = 0.25  # above this, identical glyphs are normal (no noise)
MIN_PATCH_STD = 20.0  # skip blobs with no edge detail (solid bars, rules)


@dataclass
class ClonePair:
    a: tuple[int, int, int, int]  # (x, y, w, h) in original pixels
    b: tuple[int, int, int, int]
    diff: float


@dataclass
class CopyMoveResult:
    status: str  # "clones_found" | "clean" | "not_applicable"
    pairs: list[ClonePair]
    n_glyphs: int
    note: str = ""


def _glyphs(gray: np.ndarray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    H, W = gray.shape
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (20 <= area <= 3000 and 8 <= h <= 90 and 3 <= w <= 90):
            continue
        x0, y0, x1, y1 = x - MARGIN, y - MARGIN, x + w + MARGIN, y + h + MARGIN
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        patch = gray[y0:y1, x0:x1]
        if patch.std() < MIN_PATCH_STD:
            continue
        yield (x, y, w, h), patch.astype(np.int16)


def find_clones(gray: np.ndarray, max_pairs: int = 12) -> CopyMoveResult:
    groups: dict[tuple[int, int], list] = defaultdict(list)
    n_glyphs = 0
    for box, patch in _glyphs(gray):
        groups[patch.shape].append((box, patch))
        n_glyphs += 1
    if n_glyphs < 10:
        return CopyMoveResult("not_applicable", [], n_glyphs, "too few characters found")

    pairs: list[ClonePair] = []
    cloned_glyphs: set[tuple[int, int, int, int]] = set()
    for items in groups.values():
        if len(items) < 2:
            continue
        items = items[:MAX_GROUP]
        flat = np.stack([p for _, p in items]).reshape(len(items), -1)
        for i in range(len(items) - 1):
            d = np.abs(flat[i + 1:] - flat[i]).mean(axis=1)
            for j in np.nonzero(d < CLONE_THRESHOLD)[0]:
                a, b = items[i][0], items[i + 1 + int(j)][0]
                cloned_glyphs.update((a, b))
                pairs.append(ClonePair(a, b, float(d[j])))

    if len(cloned_glyphs) > BORN_DIGITAL_CLONE_FRACTION * n_glyphs:
        return CopyMoveResult(
            "not_applicable", [], n_glyphs,
            f"{len(cloned_glyphs)} of {n_glyphs} characters have exact twins: the image looks "
            "born-digital (no scan noise), where identical characters are normal",
        )
    if not pairs:
        return CopyMoveResult("clean", [], n_glyphs)
    pairs.sort(key=lambda p: p.diff)
    return CopyMoveResult("clones_found", pairs[:max_pairs], n_glyphs)
