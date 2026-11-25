"""MIRAGE numeric consistency check (Week 3).

A purely visual check can't see an amount that was retyped in the same font
at the same position. But the numbers on an invoice or receipt have to agree
with each other. This check tests every relationship it can read:

  items:    line items (+ tax - discount + rounding) add up to a stated total
  totals:   subtotal - discount + tax + rounding = total
  payment:  cash tendered - change = total            (receipts)

Real receipts often print several "totals" (excl. GST, incl. GST, GST summary,
rounded). The check accepts a relationship if ANY plausible reading of those
totals makes it add up, and only reports a mismatch when none does. This
trades some recall for not accusing documents whose layout it only half
understands.

It is also conservative about OCR: if an amount it needs was read with low
confidence or looks garbled, it reports *inconclusive*, not a mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from itertools import product

from phantom.mirage.fields import (
    AmountField,
    RowKind,
    extract_amounts,
    fmt,
    garbled_amount_words,
)
from phantom.mirage.ocr import OcrResult, Word

TOLERANCE = Decimal("0.01")
CASH_ROUNDING = Decimal("0.05")  # Malaysian/Indian retail rounding of cash totals
MIN_WORD_CONFIDENCE = 60.0  # Tesseract's 0-100 scale
MIN_ITEMS = 2


@dataclass
class ConsistencyResult:
    status: str  # "consistent" | "mismatch" | "unreadable" | "not_applicable"
    description: str
    words: list[Word] = field(default_factory=list)  # every region the conclusion rests on
    severity: float = 0.0
    expected: Decimal | None = None
    stated: Decimal | None = None


@dataclass
class _Relation:
    name: str
    ok: bool
    description: str
    words: list[Word]
    expected: Decimal
    stated: Decimal


def _sum(fields: list[AmountField]) -> Decimal:
    return sum((f.value for f in fields), Decimal("0"))


def _close(a: Decimal, b: Decimal, cash: bool = False) -> bool:
    if abs(a - b) <= TOLERANCE:
        return True
    # A cash total may be rounded to the nearest 0.05.
    return cash and abs(a - b) <= CASH_ROUNDING / 2 + TOLERANCE


def single_column_items(amounts: list[AmountField]) -> bool:
    """True when line items look like one amount per row (a plain table). Receipts
    that print QTY x PRICE = AMOUNT put several amounts on a row; summing those
    would be meaningless, so the items relation is skipped for them."""
    items = [a for a in amounts if a.kind == RowKind.ITEM]
    rows: dict[int, int] = {}
    for a in items:
        key = next((k for k in rows if abs(k - a.word.top) <= 0.6 * a.word.height), a.word.top)
        rows[key] = rows.get(key, 0) + 1
    return len(items) >= MIN_ITEMS and all(n == 1 for n in rows.values())


def _items_relation(amounts, totals, subtotals, taxes, discounts, roundings) -> _Relation | None:
    if not totals and not subtotals:
        return None
    first_summary = min(f.word.top for f in [*totals, *subtotals])
    items = [f for f in amounts if f.kind == RowKind.ITEM and f.word.top < first_summary]
    if len(items) < MIN_ITEMS or not single_column_items(items):
        return None
    item_sum = _sum(items)
    adjust_options = {Decimal("0")}
    for t, d, r in product([Decimal("0"), _sum(taxes)], [Decimal("0"), _sum(discounts)],
                           [Decimal("0"), _sum(roundings)]):
        adjust_options.add(t - d + r)
    targets = [*subtotals, *totals]
    for target, adj in product(targets, adjust_options):
        if _close(item_sum + adj, target.value):
            return _Relation(
                "items", True,
                f"{len(items)} line items add up to {fmt(item_sum)}, consistent with the stated "
                f"{target.label or 'total'} {fmt(target.value)}.",
                [f.word for f in items] + [target.word], item_sum + adj, target.value,
            )
    closest = min(targets, key=lambda t: abs(t.value - item_sum))
    return _Relation(
        "items", False,
        f"{len(items)} line items add up to {fmt(item_sum)}, but no stated total matches "
        f"(closest: {closest.label or 'TOTAL'} {fmt(closest.value)}, off by "
        f"{fmt(abs(closest.value - item_sum))}). A line item or a total has likely been changed.",
        [f.word for f in items] + [t.word for t in targets], item_sum, closest.value,
    )


def _totals_relation(totals, subtotals, taxes, discounts, roundings) -> _Relation | None:
    if not subtotals or not taxes or not totals:
        return None
    sub, tax = subtotals[0], _sum(taxes)
    options = {sub.value + tax - d + r
               for d in (Decimal("0"), _sum(discounts)) for r in (Decimal("0"), _sum(roundings))}
    words = [sub.word, *[t.word for t in taxes], *[t.word for t in totals]]
    for total, expected in product(totals, options):
        if _close(expected, total.value):
            return _Relation(
                "totals", True,
                f"subtotal {fmt(sub.value)} + tax {fmt(tax)} matches the stated "
                f"{total.label or 'TOTAL'} {fmt(total.value)}.",
                words, expected, total.value,
            )
    expected = sub.value + tax
    closest = min(totals, key=lambda t: abs(t.value - expected))
    return _Relation(
        "totals", False,
        f"subtotal {fmt(sub.value)} + tax {fmt(tax)} = {fmt(expected)}, but the stated "
        f"{closest.label or 'TOTAL'} is {fmt(closest.value)} (off by "
        f"{fmt(abs(closest.value - expected))}). An amount or the total has likely been changed.",
        words, expected, closest.value,
    )


def _payment_relation(totals, cash, change) -> _Relation | None:
    if len(cash) != 1 or len(change) != 1 or not totals:
        return None
    c, g = cash[0], change[0]
    paid = c.value - g.value
    words = [c.word, g.word, *[t.word for t in totals]]
    for total in totals:
        if _close(paid, total.value, cash=True):
            return _Relation(
                "payment", True,
                f"cash {fmt(c.value)} - change {fmt(g.value)} = {fmt(paid)}, matching the "
                f"stated {total.label or 'TOTAL'} {fmt(total.value)}.",
                words, paid, total.value,
            )
    closest = min(totals, key=lambda t: abs(t.value - paid))
    return _Relation(
        "payment", False,
        f"cash {fmt(c.value)} - change {fmt(g.value)} = {fmt(paid)}, but the stated "
        f"{closest.label or 'TOTAL'} is {fmt(closest.value)} (off by "
        f"{fmt(abs(closest.value - paid))}). The total, cash or change has likely been changed.",
        words, paid, closest.value,
    )


def check_numeric_consistency(ocr: OcrResult) -> ConsistencyResult:
    amounts = extract_amounts(ocr)

    def kind(k: RowKind) -> list[AmountField]:
        return [f for f in amounts if f.kind == k]

    totals, subtotals, taxes = kind(RowKind.TOTAL), kind(RowKind.SUBTOTAL), kind(RowKind.TAX)
    discounts, roundings = kind(RowKind.DISCOUNT), kind(RowKind.ROUNDING)
    cash, change = kind(RowKind.CASH), kind(RowKind.CHANGE)

    relations = [r for r in (
        _items_relation(amounts, totals, subtotals, taxes, discounts, roundings),
        _totals_relation(totals, subtotals, taxes, discounts, roundings),
        _payment_relation(totals, cash, change),
    ) if r is not None]
    if not relations:
        return ConsistencyResult(
            "not_applicable", "No checkable total/payment structure found; arithmetic skipped."
        )

    # OCR reliability gate: low-confidence reads of the numbers involved, or
    # garbled amount-like tokens between the first and last amount involved.
    involved = {id(w): w for r in relations for w in r.words}.values()
    top = min(w.top for w in involved)
    bottom = max(w.top + w.height for w in involved)
    garbled = [w for w in garbled_amount_words(ocr) if top - 5 <= w.top <= bottom]
    low_conf = [w for w in involved if w.confidence < MIN_WORD_CONFIDENCE]
    failed = [r for r in relations if not r.ok]
    if failed and (garbled or low_conf):
        bad = [*garbled, *low_conf]
        names = ", ".join(f"'{w.text}'" for w in bad[:4])
        return ConsistencyResult(
            "unreadable",
            f"The amounts don't seem to add up, but {names} could not be read reliably, so "
            "this may be an OCR misread. Arithmetic check is inconclusive.",
            words=bad, severity=0.3,
        )
    if failed:
        r = failed[0]
        return ConsistencyResult("mismatch", r.description, words=r.words, severity=0.9,
                                 expected=r.expected, stated=r.stated)
    r = relations[0]
    extra = f" ({len(relations)} relationships checked)" if len(relations) > 1 else ""
    return ConsistencyResult("consistent", r.description + extra, words=r.words,
                             expected=r.expected, stated=r.stated)
