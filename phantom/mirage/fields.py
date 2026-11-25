"""MIRAGE field extraction helpers (Week 3).

Turns raw OCR words into the structure the content checks need: which words
are monetary amounts, and which text label sits on the same row as each
amount ("Item 3", "SUBTOTAL", "TOTAL", ...). Kept deliberately simple and
explainable: regexes plus row geometry, no trained layout model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from phantom.mirage.ocr import OcrResult, Word

# 1,234.00 | 1234.00 | $1,234.00 | Rs.1,234.00 | 1,234 (no decimals is allowed but rarer)
_MONEY = re.compile(r"^(?:[$€£₹]|rs\.?|inr)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d{2})?$", re.IGNORECASE)
# Something that looks like it *wants* to be an amount but didn't parse cleanly
# (e.g. OCR read "1,2O5.00"); used to downgrade the numeric check to inconclusive.
_MONEYISH = re.compile(r"^[\d,.$€£₹OoIlSB]{4,}$")


class RowKind(StrEnum):
    ITEM = "item"
    SUBTOTAL = "subtotal"
    TAX = "tax"
    TOTAL = "total"
    DISCOUNT = "discount"
    ROUNDING = "rounding"
    CASH = "cash"  # amount tendered
    CHANGE = "change"
    OTHER = "other"


# Summary-row vocabulary seen on invoices and (Malaysian) retail receipts.
# Matched as whole words, so "WIRELESS mouse" is not a discount ("less").
_CASH = {"cash", "tender", "tendered", "paid", "payment"}
_CHANGE = {"change", "baki"}
_ROUNDING = {"rounding", "rnd", "adj", "adjustment", "adjustments"}
_DISCOUNT = {"discount", "disc", "less"}
_TAX = {"tax", "gst", "vat", "sst"}
_NOT_ITEM = {"qty", "quantity", "items", "tel", "id", "no", "date", "time"}


@dataclass
class AmountField:
    word: Word
    value: Decimal
    label_words: list[Word]
    kind: RowKind

    @property
    def label(self) -> str:
        return " ".join(w.text for w in self.label_words)

    @property
    def label_top(self) -> int | None:
        return self.label_words[0].top if self.label_words else None


def parse_amount(text: str) -> Decimal | None:
    m = _MONEY.match(text.strip())
    if not m:
        return None
    # Bare integers like "3" or "2026" are row numbers/years, not amounts.
    if m.group(2) is None and "," not in m.group(1):
        return None
    try:
        return Decimal(m.group(1).replace(",", "") + (m.group(2) or ""))
    except InvalidOperation:
        return None


def looks_like_garbled_amount(text: str) -> bool:
    return parse_amount(text) is None and bool(_MONEYISH.match(text)) and any(
        c.isdigit() for c in text
    )


def _classify(label: str) -> RowKind:
    s = label.lower().replace(" ", "")
    tokens = set(re.findall(r"[a-z]+", label.lower()))
    if not s:
        return RowKind.OTHER
    if tokens & _CHANGE or ("balance" in tokens and "due" not in tokens):
        return RowKind.CHANGE
    if tokens & _CASH and "total" not in s:
        return RowKind.CASH
    if tokens & _ROUNDING or "rounding" in s:
        return RowKind.ROUNDING
    if tokens & _DISCOUNT or "discount" in s:
        return RowKind.DISCOUNT
    has_tax_word = bool(tokens & _TAX) or "gst" in s
    if "subtotal" in s or "excl" in s or ("total" in s and "before" in s):
        return RowKind.SUBTOTAL
    if "total" in s or "amountdue" in s or "balancedue" in s or "nett" in tokens:
        # "TOTAL GST" is a tax line; "TOTAL ... (INCLUSIVE OF GST)" is the total.
        if has_tax_word and "incl" not in s:
            return RowKind.TAX
        return RowKind.TOTAL
    if has_tax_word:
        return RowKind.TAX
    if tokens & _NOT_ITEM:
        return RowKind.OTHER
    # Any other labelled amount row is treated as a line item (descriptions vary).
    return RowKind.ITEM


def _same_row(a: Word, b: Word) -> bool:
    """Rows are matched on vertical centre, tolerant to a few px of drift."""
    ca, cb = a.top + a.height / 2, b.top + b.height / 2
    return abs(ca - cb) <= 0.8 * max(a.height, b.height)


def extract_amounts(ocr: OcrResult) -> list[AmountField]:
    """All monetary amounts on the page, each paired with its row label."""
    fields: list[AmountField] = []
    for w in ocr.words:
        value = parse_amount(w.text)
        if value is None:
            continue
        label_words = sorted(
            (
                o
                for o in ocr.words
                if o is not w
                and o.left + o.width <= w.left
                and _same_row(o, w)
                and parse_amount(o.text) is None
            ),
            key=lambda o: o.left,
        )
        kind = _classify(" ".join(o.text for o in label_words)) if label_words else RowKind.OTHER
        fields.append(AmountField(word=w, value=value, label_words=label_words, kind=kind))
    return sorted(fields, key=lambda f: f.word.top)


def garbled_amount_words(ocr: OcrResult) -> list[Word]:
    return [w for w in ocr.words if looks_like_garbled_amount(w.text)]


_RECEIPT_WORDS = {"cash", "change", "rounding", "receipt", "cashier", "thank", "tender",
                  "tendered", "counter", "member", "items", "qty"}


def looks_like_receipt(ocr: OcrResult, min_hits: int = 2) -> bool:
    """Retail/till receipt vocabulary (two or more distinct words). Used to route
    a page to the checks that were validated on real receipts (Find it again!).
    Synthetic invoices (Description / Amount / SUBTOTAL / GST / TOTAL) don't match."""
    words = {re.sub(r"[^a-z]", "", w.text.lower()) for w in ocr.words}
    return len(words & _RECEIPT_WORDS) >= min_hits


def fmt(value: Decimal) -> str:
    return f"{value:,.2f}"
