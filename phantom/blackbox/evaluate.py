"""BLACKBOX evaluation (Day 27): learned ranker vs baselines on held-out sites.

Held-out = TEST-theme sites (govt_benefits, crypto, tech_support): themes the
ranker never saw. Every strategy gets the same action budget on the same
sites, repeated over several trials with shuffled link order.

Metrics per episode: known-behaviour discovery (found / planted), coverage
(pages visited / site pages), false positives (detections that aren't planted
behaviours), actions to first discovery, runtime. Learned vs each baseline is
compared with a paired Wilcoxon signed-rank test over (site, trial) pairs.

An "oracle" that knows the site map is reported as an upper bound only.

    python -m phantom.blackbox.evaluate --trials 5 --json-out docs/blackbox_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import zlib
from pathlib import Path

from phantom.blackbox.browser import SafeBrowser
from phantom.blackbox.explore import FixedRuleStrategy, RandomStrategy, Strategy, run_episode
from phantom.blackbox.ranker import BUDGET, LearnedStrategy, _distances, _page
from phantom.blackbox.sites import load_truth, serve


class OracleStrategy(Strategy):
    """Upper bound: knows the site graph and planted pages (NOT a real strategy)."""
    name = "oracle (upper bound)"

    def __init__(self, truth: dict):
        self.edges = truth["edges"]
        self.planted = {b["page"] for b in truth["planted"]}

    def choose(self, state, actions, ctx, rng):
        seen = {_page(u) for u in ctx.history}
        todo = self.planted - seen
        if not todo:
            return actions[0]
        dist = _distances(self.edges, todo)
        return min(actions, key=lambda a: (dist.get(_page(a.target), 99),
                                           ctx.visited.get(a.target.split("?")[0], 0)))


def evaluate(root: Path, trials: int, budget: int, split: str = "test") -> dict:
    idx = json.loads((root / "index.json").read_text())
    ids = [s["id"] for s in idx if s["split"] == split]
    learned = LearnedStrategy()
    strategies = {"random": lambda t: RandomStrategy(), "fixed_rule": lambda t: FixedRuleStrategy(),
                  "learned": lambda t: learned, "oracle": OracleStrategy}
    rows = []
    srv, base = serve(root)
    try:
        with SafeBrowser(base) as b:
            for sid in ids:
                truth = load_truth(root, sid)
                planted = {(x["type"], x["page"]) for x in truth["planted"]}
                for trial in range(trials):
                    for name, make in strategies.items():
                        # same seed per (site, trial) for every strategy: identical link orders
                        seed = zlib.crc32(f"{sid}:{trial}".encode())  # reproducible
                        ep = run_episode(b, base, sid, make(truth), budget, random.Random(seed))
                        hits = {k: s for k, s in ep.found.items() if k in planted}
                        rows.append({
                            "site": sid, "theme": truth["theme"], "trial": trial, "strategy": name,
                            "planted": len(planted), "found": len(hits),
                            "found_types": sorted(t for t, _ in hits),
                            "missed_types": sorted(t for t, _ in planted - set(hits)),
                            "false_positives": len([k for k in ep.found if k not in planted]),
                            "coverage": len(ep.pages_visited) / truth["n_pages"],
                            "first_discovery": min(hits.values()) if hits else None,
                            "runtime_s": round(ep.runtime_s, 3),
                            "blocked_requests": ep.blocked_requests,
                        })
            blocked_total = len(b.blocked)
    finally:
        srv.shutdown()
    return {"split": split, "trials": trials, "budget": budget, "n_sites": len(ids),
            "summary": summarize(rows), "blocked_external_requests": blocked_total, "rows": rows}


def summarize(rows: list[dict]) -> dict:
    from scipy.stats import wilcoxon

    out = {}
    for name in dict.fromkeys(r["strategy"] for r in rows):
        rs = [r for r in rows if r["strategy"] == name]
        disc = [r["found"] / r["planted"] for r in rs]
        firsts = [r["first_discovery"] for r in rs if r["first_discovery"] is not None]
        per_type: dict[str, list[int]] = {}
        for r in rs:
            for t in r["found_types"]:
                per_type.setdefault(t, [0, 0])[0] += 1
            for t in r["found_types"] + r["missed_types"]:
                per_type.setdefault(t, [0, 0])[1] += 1
        out[name] = {
            "episodes": len(rs),
            "discovery_mean": round(statistics.mean(disc), 3),
            "discovery_sd": round(statistics.pstdev(disc), 3),
            "behaviours_found": f"{sum(r['found'] for r in rs)}/{sum(r['planted'] for r in rs)}",
            "coverage_mean": round(statistics.mean(r["coverage"] for r in rs), 3),
            "false_positives_total": sum(r["false_positives"] for r in rs),
            "actions_to_first_discovery_mean":
                round(statistics.mean(firsts), 2) if firsts else None,
            "episodes_with_no_discovery": sum(1 for r in rs if r["found"] == 0),
            "runtime_s_mean": round(statistics.mean(r["runtime_s"] for r in rs), 3),
            "discovery_by_type": {t: round(v[0] / v[1], 2) for t, v in sorted(per_type.items())},
        }
    key = {(r["site"], r["trial"], r["strategy"]): r["found"] / r["planted"] for r in rows}
    pairs = sorted({(r["site"], r["trial"]) for r in rows})
    for other in ("fixed_rule", "random"):
        a = [key[(s, t, "learned")] for s, t in pairs]
        o = [key[(s, t, other)] for s, t in pairs]
        diff = [x - y for x, y in zip(a, o, strict=True)]
        try:
            p = float(wilcoxon(a, o).pvalue) if any(diff) else 1.0
        except ValueError:
            p = 1.0
        out[f"learned_vs_{other}"] = {
            "mean_difference": round(statistics.mean(diff), 3),
            "learned_better": sum(d > 0 for d in diff), "ties": sum(d == 0 for d in diff),
            "learned_worse": sum(d < 0 for d in diff), "wilcoxon_p": round(p, 4)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/blackbox_sites")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--split", default="test")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    res = evaluate(Path(a.root), a.trials, a.budget, a.split)
    print(json.dumps(res["summary"], indent=2))
    print("blocked external requests:", res["blocked_external_requests"])
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=2))
