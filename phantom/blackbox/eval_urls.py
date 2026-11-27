"""Evaluate BLACKBOX's URL lookalike checker on independent public data.

Offline and reproducible (no lookups; nothing is fetched from the URLs):

1. dnstwist lookalikes: an independent open-source generator (dnstwist) makes
   typo / homoglyph / hyphenation / ... variants of each brand's main domain.
   Recall per variant type. These are *possible* fakes, not confirmed phishing.
2. Tranco top sites (tranco-list.eu, research ranking of popular domains):
   false-alarm rate on the top N legitimate domains.
3. PhiUSIIL (UCI ML repository, 2024, CC BY 4.0): 235k real phishing and
   legitimate URLs with the page TITLE. Brand targets are labelled from the
   title, independently of the URL: phishing pages whose title names one of our
   brands (recall), and legitimate pages whose title names one (false alarms).

    python -m phantom.blackbox.eval_urls --data data/raw/urls --json-out docs/blackbox_url_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from phantom.blackbox.brands import BRANDS
from phantom.blackbox.urlcheck import check_url

PRIMARY = {"amazon": ["amazon.in", "amazon.com"], "flipkart": ["flipkart.com"],
           "myntra": ["myntra.com"], "meesho": ["meesho.com"], "paytm": ["paytm.com"],
           "phonepe": ["phonepe.com"], "gpay": [], "sbi": ["onlinesbi.com", "sbi.co.in"],
           "hdfc": ["hdfcbank.com"], "icici": ["icicibank.com"], "irctc": ["irctc.co.in"],
           "indiapost": ["indiapost.gov.in"]}
TITLE_NAMES = {"amazon": r"amazon", "flipkart": r"flipkart", "myntra": r"myntra",
               "meesho": r"meesho", "paytm": r"paytm", "phonepe": r"phonepe",
               "gpay": r"google pay|gpay", "sbi": r"\bsbi\b|state bank of india|onlinesbi",
               "hdfc": r"hdfc", "icici": r"icici", "irctc": r"irctc",
               "indiapost": r"india post|indiapost"}
FLAGGED = ("likely_fake", "suspicious")


def eval_dnstwist() -> dict:
    import dnstwist

    per_type: dict[str, Counter] = defaultdict(Counter)
    misses: dict[str, list[str]] = defaultdict(list)
    for key, domains in PRIMARY.items():
        official = BRANDS[key].official
        for d in domains:
            f = dnstwist.Fuzzer(d)
            f.generate()
            for v in f.domains:
                name, kind = v["domain"], v["fuzzer"]
                if kind == "*original" or name in official:
                    continue
                r = check_url(name)
                c = per_type[kind]
                c["n"] += 1
                c["likely_fake_right_brand"] += r.verdict == "likely_fake" and r.brand == key
                c["flagged"] += r.verdict in FLAGGED
                if r.verdict not in FLAGGED and len(misses[kind]) < 6:
                    misses[kind].append(name)
    out = {}
    tot = Counter()
    for kind, c in sorted(per_type.items()):
        tot.update(c)
        out[kind] = {"n": c["n"], "flagged": round(c["flagged"] / c["n"], 3),
                     "likely_fake_right_brand": round(c["likely_fake_right_brand"] / c["n"], 3),
                     "example_misses": misses[kind]}
    out["ALL"] = {"n": tot["n"], "flagged": round(tot["flagged"] / tot["n"], 3),
                  "likely_fake_right_brand": round(tot["likely_fake_right_brand"] / tot["n"], 3)}
    return out


def eval_tranco(path: Path, top: int, start: int = 0) -> dict:
    verdicts, fakes, susp, officials = Counter(), [], [], Counter()
    with open(path, encoding="utf-8") as fh:
        for i, (rank, domain) in enumerate(csv.reader(fh)):
            if i < start:
                continue
            if i >= top:
                break
            r = check_url(domain)
            verdicts[r.verdict] += 1
            if r.verdict == "official":
                officials[r.brand] += 1
            elif r.verdict == "likely_fake" and len(fakes) < 40:
                fakes.append(f"#{rank} {domain} -> {r.brand}: "
                             + "; ".join(s.name for s in r.signals))
            elif r.verdict == "suspicious" and len(susp) < 15:
                susp.append(f"#{rank} {domain}: " + "; ".join(s.name for s in r.signals))
    n = sum(verdicts.values())
    return {"n": n, "verdicts": dict(verdicts),
            "likely_fake_rate": round(verdicts["likely_fake"] / n, 4),
            "flagged_rate": round((verdicts["likely_fake"] + verdicts["suspicious"]) / n, 4),
            "official_by_brand": dict(officials), "likely_fake_examples": fakes,
            "suspicious_examples": susp}


def eval_phiusiil(path: Path) -> dict:
    title_rx = {k: re.compile(v, re.I) for k, v in TITLE_NAMES.items()}
    stats = {"phish_brand": Counter(), "legit_brand": Counter(), "phish_all": Counter(),
             "legit_all": Counter()}
    by_brand: dict[str, Counter] = defaultdict(Counter)
    examples = {"missed_phish": [], "false_alarm_legit": []}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            url, title = row["URL"], row.get("Title") or ""
            phishing = row["label"].strip() == "0"  # PhiUSIIL: 1 = legitimate, 0 = phishing
            r = check_url(url)
            group = "phish_all" if phishing else "legit_all"
            stats[group][r.verdict] += 1
            target = next((k for k, rx in title_rx.items() if rx.search(title)), None)
            if not target:
                continue
            g = "phish_brand" if phishing else "legit_brand"
            stats[g][r.verdict] += 1
            bb = by_brand[f"{target}:{'phish' if phishing else 'legit'}"]
            bb["n"] += 1
            bb["flagged"] += r.verdict in FLAGGED
            bb["official"] += r.verdict == "official"
            if phishing and r.verdict not in FLAGGED and len(examples["missed_phish"]) < 15:
                examples["missed_phish"].append(f"{url}  [title: {title[:60]}] -> {r.verdict}")
            if not phishing and r.verdict == "likely_fake" and \
                    len(examples["false_alarm_legit"]) < 15:
                examples["false_alarm_legit"].append(f"{url}  [title: {title[:60]}]")

    def rate(c, keys):
        n = sum(c.values())
        return {"n": n, **{k: round(c[k] / n, 3) if n else None for k in keys},
                "flagged": round((c["likely_fake"] + c["suspicious"]) / n, 3) if n else None}

    keys = ("official", "likely_fake", "suspicious", "no_brand_link")
    return {"brand_titled_phishing": rate(stats["phish_brand"], keys),
            "brand_titled_legitimate": rate(stats["legit_brand"], keys),
            "all_phishing": rate(stats["phish_all"], keys),
            "all_legitimate": rate(stats["legit_all"], keys),
            "by_brand": {k: dict(v) for k, v in sorted(by_brand.items())},
            "examples": examples}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/urls")
    ap.add_argument("--tranco-top", type=int, default=100000)
    ap.add_argument("--tranco-start", type=int, default=0,
                    help="skip the first N ranks (held-out split: ranks 1-100k were used "
                         "to find false alarms, report 100,001-200,000)")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    data, res = Path(a.data), {}
    for name, fn in (("dnstwist", eval_dnstwist),
                     ("tranco", lambda: eval_tranco(data / "top-1m.csv", a.tranco_top,
                                                     a.tranco_start)),
                     ("phiusiil",
                      lambda: eval_phiusiil(data / "PhiUSIIL_Phishing_URL_Dataset.csv"))):
        t0 = time.perf_counter()
        res[name] = fn()
        res[name]["seconds"] = round(time.perf_counter() - t0, 1)
        print(name, json.dumps({k: v for k, v in res[name].items()
                                if k not in ("examples", "likely_fake_examples",
                                             "suspicious_examples", "by_brand")}, indent=1))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=2))
