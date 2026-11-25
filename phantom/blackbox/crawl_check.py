"""Exhaustive crawl to validate the detectors against ground truth (no budget).

If a full crawl doesn't find exactly the planted behaviours (and no decoys),
the detectors are wrong and no explorer comparison would mean anything.

    python -m phantom.blackbox.crawl_check --split train --limit 10
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

from phantom.blackbox.browser import SafeBrowser
from phantom.blackbox.detectors import detect
from phantom.blackbox.sites import load_truth, serve


def crawl(root: Path, site_id: str, browser: SafeBrowser, base: str) -> dict:
    start = f"{base}/{site_id}/index.html"
    seen, queue, found = {start}, deque([start]), set()
    while queue:
        st = browser.load(queue.popleft())
        found |= {(d.type, d.page.rsplit("/", 1)[-1]) for d in detect(st, base)}
        for a in browser.actions(st):
            if a.target not in seen:
                seen.add(a.target)
                queue.append(a.target)
    truth = load_truth(root, site_id)
    planted = {(b["type"], b["page"]) for b in truth["planted"]}
    return {"site": site_id, "missed": sorted(planted - found), "false": sorted(found - planted),
            "pages_loaded": len(seen)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/blackbox_sites")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args()
    root = Path(a.root)
    ids = [s["id"] for s in json.loads((root / "index.json").read_text())
           if s["split"] == a.split][: a.limit]
    srv, base = serve(root)
    tot_m = tot_f = tot_p = 0
    with SafeBrowser(base) as b:
        for sid in ids:
            r = crawl(root, sid, b, base)
            tot_m += len(r["missed"])
            tot_f += len(r["false"])
            tot_p += len(load_truth(root, sid)["planted"])
            if r["missed"] or r["false"]:
                print(r)
        print(f"{len(ids)} sites, {tot_p} planted: missed {tot_m}, false positives {tot_f}; "
              f"blocked external requests: {len(b.blocked)}")
    srv.shutdown()
