"""Generate a small labelled synthetic invoice set (Day 6 starter set).

Each benign invoice gets a suspicious twin with one planted manipulation.
Ground truth (type + bounding box) is written to labels.json.

    python -m phantom.mirage.make_samples --out data/samples --n 10
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

W, H = 1240, 1754
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _invoice(rng: random.Random, template: int) -> tuple[np.ndarray, dict]:
    img = np.full((H, W, 3), 255, np.uint8)
    x0 = 90 + template * 20
    cv2.putText(img, f"ACME SUPPLIES  - TEMPLATE {template}", (x0, 120), FONT, 1.3, (0, 0, 0), 3)
    inv_no = f"Invoice No: INV-{rng.randint(1000, 9999)}"
    cv2.putText(img, inv_no, (x0, 200), FONT, 0.9, (0, 0, 0), 2)
    fields = {}
    y = 320
    total = 0
    for i in range(6):
        amt = rng.randint(100, 5000)
        total += amt
        cv2.putText(img, f"Item {i + 1}", (x0, y), FONT, 0.9, (0, 0, 0), 2)
        cv2.putText(img, f"{amt:>8,}.00", (850, y), FONT, 0.9, (0, 0, 0), 2)
        fields[f"item_{i + 1}"] = (850, y - 30, 260, 40)
        y += 70
    cv2.putText(img, "TOTAL", (x0, y + 40), FONT, 1.0, (0, 0, 0), 2)
    cv2.putText(img, f"{total:>8,}.00", (850, y + 40), FONT, 1.0, (0, 0, 0), 2)
    fields["total"] = (850, y + 5, 280, 45)
    return img, fields


def _manipulate(img: np.ndarray, fields: dict, rng: random.Random) -> tuple[np.ndarray, dict]:
    img = img.copy()
    kind = rng.choice(["altered_amount", "misaligned_field", "font_mismatch"])
    name = rng.choice(list(fields))
    x, y, w, h = fields[name]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    new_val = f"{rng.randint(10000, 99999):,}.00"
    if kind == "altered_amount":
        cv2.putText(img, new_val, (x, y + 30), FONT, 0.9, (0, 0, 0), 2)
    elif kind == "misaligned_field":
        x += rng.choice([-25, 25])
        y += rng.choice([-8, 8])
        cv2.putText(img, new_val, (x, y + 30), FONT, 0.9, (0, 0, 0), 2)
    else:
        cv2.putText(img, new_val, (x, y + 30), cv2.FONT_HERSHEY_COMPLEX, 0.85, (30, 30, 30), 1)
    return img, {"type": kind, "field": name, "bbox": [x, y, w, h]}


def main(out: str, n: int, seed: int = 7) -> None:
    rng = random.Random(seed)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for i in range(n):
        template = i % 3
        img, fields = _invoice(rng, template)
        benign = f"inv_{i:03d}_benign.png"
        cv2.imwrite(str(out_dir / benign), img)
        labels.append({"file": benign, "label": "benign", "template": template, "regions": []})
        bad, region = _manipulate(img, fields, rng)
        sus = f"inv_{i:03d}_suspicious.png"
        cv2.imwrite(str(out_dir / sus), bad)
        labels.append(
            {"file": sus, "label": "suspicious", "template": template, "regions": [region]}
        )
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"wrote {len(labels)} images + labels.json to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    main(a.out, a.n)
