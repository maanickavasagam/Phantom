"""Generate a harder, more realistic synthetic invoice set (Week 3).

The Day 6 set (make_samples.py) turned out to be too easy once the numeric
consistency check existed: every manipulation changed one value without
touching the total. This generator is meant to measure MIRAGE more honestly:

- realistic layout: right-aligned amounts, 3-8 described line items, paise,
  optional SUBTOTAL + GST rows;
- every page (benign too) goes through a scan simulation: paper tone, sensor
  noise, slight skew, blur and JPEG compression;
- manipulations include ``consistent_edit``, which changes a line item *and*
  the totals so the arithmetic still adds up (the numeric check can't catch
  this by design; only visual traces can);
- edits happen either *after* scanning (a patch may be visible) or *before*
  scanning (the scan hides the patch, as when a digital PDF is edited then
  printed and scanned).

Use one seed for tuning (dev) and a different seed for the reported numbers
(holdout), and never tune thresholds on the holdout set:

    python -m phantom.mirage.make_hard_samples --out data/dev_hard --n 20 --seed 11
    python -m phantom.mirage.make_hard_samples --out data/holdout_hard --n 30 --seed 2026
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

W, H = 1240, 1754
SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
OTHER_FONTS = [cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX]

VENDORS = ["ACME SUPPLIES", "NORTHWIND TRADERS", "GLOBEX OFFICE", "INITECH SERVICES"]
ITEMS = [
    "Office chairs", "Printer toner", "A4 paper reams", "Desk lamps", "USB keyboards",
    "Network cables", "Whiteboard markers", "Filing cabinets", "Laptop stands",
    "Monitor arms", "Cleaning service", "Courier charges", "Software licence",
    "Annual maintenance", "Projector rental", "Catering",
]


@dataclass
class Field:
    name: str
    value: float
    right: int  # x of right edge
    baseline: int  # y of text baseline
    scale: float
    thickness: int


def money(v: float) -> str:
    return f"{v:,.2f}"


def _put_right(img, text, right, baseline, font, scale, thickness, color=(0, 0, 0)):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (right - tw, baseline), font, scale, color, thickness, cv2.LINE_AA)


def _field_box(f: Field, font=SIMPLEX, pad: int = 6) -> tuple[int, int, int, int]:
    (tw, th), base = cv2.getTextSize(money(f.value), font, f.scale, f.thickness)
    return (f.right - tw - pad, f.baseline - th - pad, tw + 2 * pad, th + base + 2 * pad)


def _invoice(rng: random.Random, template: int) -> tuple[np.ndarray, dict[str, Field], dict]:
    img = np.full((H, W, 3), 255, np.uint8)
    x0 = 90 + 15 * template
    right = 1100 + 20 * (template % 3)
    scale = [0.85, 0.9, 0.8, 0.88][template % 4]
    vendor = VENDORS[template % len(VENDORS)]
    cv2.putText(img, vendor, (x0, 120), SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, f"Invoice No: INV-{rng.randint(1000, 9999)}", (x0, 190), SIMPLEX, 0.8,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Description", (x0, 280), SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    _put_right(img, "Amount", right, 280, SIMPLEX, 0.8, 2)
    cv2.line(img, (x0, 300), (right, 300), (0, 0, 0), 2)

    fields: dict[str, Field] = {}
    y = 360
    n_items = rng.randint(3, 8)
    names = rng.sample(ITEMS, n_items)
    subtotal = 0.0
    for i, name in enumerate(names):
        v = round(rng.randint(150, 48000) + rng.choice([0, 0, 0.5, 0.25, rng.randint(1, 99) / 100]),
                  2)
        subtotal += v
        cv2.putText(img, name, (x0, y), SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
        f = Field(f"item_{i + 1}", v, right, y, scale, 2)
        _put_right(img, money(v), right, y, SIMPLEX, scale, 2)
        fields[f.name] = f
        y += 62
    subtotal = round(subtotal, 2)
    with_tax = template % 2 == 1
    cv2.line(img, (right - 380, y - 30), (right, y - 30), (0, 0, 0), 1)
    y += 20
    if with_tax:
        tax = round(subtotal * 0.18, 2)
        for label, name, v in (("SUBTOTAL", "subtotal", subtotal), ("GST 18%", "tax", tax)):
            cv2.putText(img, label, (x0, y), SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
            _put_right(img, money(v), right, y, SIMPLEX, scale, 2)
            fields[name] = Field(name, v, right, y, scale, 2)
            y += 62
        total = round(subtotal + tax, 2)
    else:
        total = subtotal
    cv2.putText(img, "TOTAL", (x0, y + 10), SIMPLEX, scale + 0.1, (0, 0, 0), 3, cv2.LINE_AA)
    _put_right(img, money(total), right, y + 10, SIMPLEX, scale + 0.1, 3)
    fields["total"] = Field("total", total, right, y + 10, scale + 0.1, 3)
    return img, fields, {"with_tax": with_tax, "n_items": n_items}


def _scan(img: np.ndarray, rng: random.Random, skew: bool = True) -> np.ndarray:
    """Simulate print-and-scan: paper tone, skew, blur, noise, JPEG."""
    tone = rng.uniform(0.89, 0.97)
    out = img.astype(np.float32) * tone
    angle = rng.uniform(-1.2, 1.2) if skew else 0.0
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    paper = float(255 * tone)
    out = cv2.warpAffine(out, m, (W, H), borderValue=(paper, paper, paper))
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 0.9))
    out += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2.5, 5.0), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 85)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else out


def _redraw(img, f: Field, new_value: float, fill, *, font=SIMPLEX, dx=0, dy=0, thickness=None,
            color=(0, 0, 0), scale=None) -> tuple[int, int, int, int]:
    x, y, w, h = _field_box(f)
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    _put_right(img, money(new_value), f.right + dx, f.baseline + dy, font,
               scale or f.scale, thickness or f.thickness, color)
    return x + min(dx, 0), y + min(dy, 0), w + abs(dx), h + abs(dy)


def _manipulate(img, fields: dict[str, Field], meta: dict, rng: random.Random, stage: str,
                donor: np.ndarray) -> tuple[np.ndarray, dict]:
    img = img.copy()
    kind = rng.choice(
        ["altered_amount", "consistent_edit", "misaligned_field", "font_mismatch", "pasted_patch"]
    )
    items = [k for k in fields if k.startswith("item_")]
    target = fields[rng.choice(items)]
    # Fill colour for the white-out: pure white (careless) or the paper colour (careful).
    if stage == "after_scan" and rng.random() < 0.5:
        paper = int(np.median(img))
        fill = (paper, paper, paper)
    else:
        fill = (255, 255, 255)
    consistent = kind == "consistent_edit" or (
        kind in ("misaligned_field", "font_mismatch") and rng.random() < 0.5
    )
    new_value = round(target.value * rng.uniform(1.3, 4.0) + rng.randint(1, 99) / 100, 2)
    regions = []

    if kind == "pasted_patch":
        # Paste an amount-sized patch from a *different* scanned invoice.
        x, y, w, h = _field_box(target)
        dx, dy = rng.randint(-200, 200), rng.randint(0, 600)
        sx, sy = max(0, min(W - w, x + dx)), max(0, min(H - h, y + dy))
        img[y:y + h, x:x + w] = donor[sy:sy + h, sx:sx + w]
        regions.append({"field": target.name, "bbox": [x, y, w, h]})
    else:
        kw = {}
        if kind == "misaligned_field":
            kw = {"dx": rng.choice([-1, 1]) * rng.randint(18, 40),
                  "dy": rng.choice([0, rng.choice([-1, 1]) * rng.randint(6, 12)])}
        elif kind == "font_mismatch":
            kw = {"font": rng.choice(OTHER_FONTS), "thickness": 1, "color": (40, 40, 40),
                  "scale": target.scale * rng.uniform(0.85, 1.0)}
        box = _redraw(img, target, new_value, fill, **kw)
        regions.append({"field": target.name, "bbox": list(box)})
        if consistent:
            delta = new_value - target.value
            if meta["with_tax"]:
                sub = fields["subtotal"]
                new_sub = round(sub.value + delta, 2)
                new_tax = round(new_sub * 0.18, 2)
                for f, v in ((sub, new_sub), (fields["tax"], new_tax)):
                    regions.append({"field": f.name, "bbox": list(_redraw(img, f, v, fill))})
                new_total = round(new_sub + new_tax, 2)
            else:
                new_total = round(fields["total"].value + delta, 2)
            t = fields["total"]
            regions.append({"field": "total", "bbox": list(_redraw(img, t, new_total, fill))})
    return img, {"type": kind, "stage": stage, "arithmetic_consistent": consistent,
                 "regions": regions}


def main(out: str, n: int, seed: int) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for i in range(n):
        template = i % 4
        clean, fields, meta = _invoice(rng, template)
        stage = rng.choice(["after_scan", "before_scan"])
        # Pages that get edited after scanning are scanned straight (sheet-fed), so
        # the edit lands where the field actually is.
        benign_scan = _scan(clean, rng, skew=stage == "before_scan")
        name = f"hard_{i:03d}_benign.png"
        cv2.imwrite(str(out_dir / name), benign_scan)
        labels.append({"file": name, "label": "benign", "template": template, **meta,
                       "manipulation": None})

        donor_clean, _, _ = _invoice(rng, (template + 1) % 4)
        donor = _scan(donor_clean, rng, skew=False)
        if stage == "after_scan":
            # Edit the scanned page directly. Re-use the benign twin's scan so the
            # only difference between the pair is the edit itself.
            bad, info = _manipulate(benign_scan, fields, meta, rng, stage, donor)
            # Edits made after scanning are normally re-saved, often as JPEG.
            ok, enc = cv2.imencode(".jpg", bad, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 92)])
            bad = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        else:
            bad, info = _manipulate(clean, fields, meta, rng, stage, donor_clean)
            bad = _scan(bad, rng)
        name = f"hard_{i:03d}_suspicious.png"
        cv2.imwrite(str(out_dir / name), bad)
        labels.append({"file": name, "label": "suspicious", "template": template, **meta,
                       "manipulation": info})
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"wrote {len(labels)} images + labels.json to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/holdout_hard")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    main(a.out, a.n, a.seed)
