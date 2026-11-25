"""BLACKBOX learned action ranker (Days 25-26).

A small neural network (MLP, 2 hidden layers) that scores each permitted next
action from OBSERVABLE features only: the link/button text, where it sits on
the page, form fields, what was already visited, depth and remaining budget.
It never sees the ground truth or URL names (pages are named p<N>.html).

Training labels come from an oracle that knows the site graph: an action is
"good" if it moves strictly closer (in clicks) to a planted behaviour the
episode hasn't found yet. The network imitates that oracle (supervised
learning; reinforcement learning is out of scope per docs/SCOPE.md), with one
DAgger round so it also learns from the states its own choices lead to.

Data: TRAIN-theme sites only. 8 of them are held back as validation for model
selection; the test-theme sites are never touched here.

    python -m phantom.blackbox.ranker train --root data/blackbox_sites
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from collections import deque
from pathlib import Path

import numpy as np

from phantom.blackbox.browser import Action, PageState, SafeBrowser
from phantom.blackbox.explore import (
    CTA_WORDS,
    Context,
    FixedRuleStrategy,
    RandomStrategy,
    Strategy,
    _is_plain_form,
    run_episode,
)
from phantom.blackbox.sites import load_truth, serve

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "ranker.joblib"
META_PATH = MODEL_DIR / "ranker.json"
N_HASH = 64
BUDGET = 12
REGIONS = ("nav", "body", "button", "footer", "form")


def _h(token: str) -> int:
    return zlib.crc32(token.encode()) % N_HASH


def featurize(state: PageState, a: Action, ctx: Context) -> np.ndarray:
    text = (a.text or "").strip()
    words = re.findall(r"[a-z]+", text.lower())
    target = a.target.split("?")[0]
    fields = (a.form or {}).get("fields", [])
    f = [
        a.kind == "link", a.kind == "form", a.kind == "back",
        *[a.region == r for r in REGIONS],
        ctx.visited.get(target, 0) == 0, min(ctx.visited.get(target, 0), 3) / 3,
        min(len(words), 8) / 8, text.lower() in ("home", "back"),
        bool(CTA_WORDS.search(text)), text.lower().startswith("continue"),
        text[:1].isupper() and len(words) >= 2,
        len(fields) / 4, any(fl["type"] == "password" for fl in fields),
        any(fl["type"] in ("tel", "text") for fl in fields),
        a.kind == "form" and _is_plain_form(a),
        bool(state.forms), len(state.links) / 20,
        ctx.depth / 4, ctx.step / max(ctx.budget, 1), min(ctx.found, 4) / 4,
    ]
    bag = np.zeros(N_HASH)
    for w in words:
        bag[_h(w)] = 1.0
    return np.concatenate([np.array(f, dtype=float), bag])


# ------------------------------------------------------------------ oracle labels


def _distances(edges: dict[str, list[str]], targets: set[str]) -> dict[str, int]:
    """Clicks from each page to the nearest target (reverse BFS)."""
    rev: dict[str, list[str]] = {}
    for src, dsts in edges.items():
        for d in dsts:
            rev.setdefault(d, []).append(src)
    dist = {t: 0 for t in targets}
    q = deque(targets)
    while q:
        u = q.popleft()
        for v in rev.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _page(url: str) -> str:
    return url.split("?")[0].rsplit("/", 1)[-1]


class _Collector:
    """Records (features, oracle label) for every permitted action in every visited state."""

    def __init__(self, truth: dict):
        self.edges = truth["edges"]
        self.planted = {(b["type"], b["page"]) for b in truth["planted"]}
        self.X, self.y = [], []

    def __call__(self, state: PageState, actions: list[Action], ctx: Context,
                 found: set) -> None:
        todo = {p for t, p in self.planted if (t, p) not in found}
        if not todo:
            return
        dist = _distances(self.edges, todo)
        here = dist.get(_page(state.url), 99)
        for a in actions:
            there = dist.get(_page(a.target), 99)
            self.X.append(featurize(state, a, ctx))
            self.y.append(int(there < here))


class LearnedStrategy(Strategy):
    name = "learned"

    def __init__(self, model=None):
        if model is None:
            import joblib

            model = joblib.load(MODEL_PATH)
        self.model = model

    def choose(self, state, actions, ctx, rng):
        # Same exploration constraint the fixed rule has (unvisited targets first), so
        # the comparison is about ranking, not about one strategy being allowed to
        # waste actions on revisits. The network ranks within that set.
        fresh = [a for a in actions
                 if a.kind != "back" and ctx.visited.get(a.target.split("?")[0], 0) == 0]
        cands = fresh or actions
        X = np.stack([featurize(state, a, ctx) for a in cands])
        p = self.model.predict_proba(X)[:, 1]
        return cands[int(np.argmax(p))]


class _Explore(Strategy):
    """Behaviour policy for data collection: mixes the two baselines."""
    name = "collect"

    def __init__(self, eps: float):
        self.eps, self.r, self.f = eps, RandomStrategy(), FixedRuleStrategy()

    def choose(self, state, actions, ctx, rng):
        return (self.r if rng.random() < self.eps else self.f).choose(state, actions, ctx, rng)


def _collect(browser, base, root, ids, strategy_fn, episodes, seed):
    X, y = [], []
    rng = random.Random(seed)
    for sid in ids:
        truth = load_truth(root, sid)
        for _ in range(episodes):
            col = _Collector(truth)  # labels use what THIS episode has found so far
            run_episode(browser, base, sid, strategy_fn(), BUDGET, rng, on_state=col)
            X += col.X
            y += col.y
    return np.array(X), np.array(y)


def _policy_score(browser, base, root, ids, strategy, trials=3, seed=7) -> float:
    rng = random.Random(seed)
    found = total = 0
    for sid in ids:
        planted = {(b["type"], b["page"]) for b in load_truth(root, sid)["planted"]}
        for _ in range(trials):
            ep = run_episode(browser, base, sid, strategy, BUDGET, rng)
            found += len(set(ep.found) & planted)
            total += len(planted)
    return found / total


def _fit(X, y, hidden, alpha):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=hidden, alpha=alpha, max_iter=300, early_stopping=True,
        random_state=0)).fit(X, y)


def train(root: Path) -> dict:
    import joblib

    idx = json.loads((root / "index.json").read_text())
    train_ids = [s["id"] for s in idx if s["split"] == "train"]
    rng = random.Random(0)
    rng.shuffle(train_ids)
    val_ids, fit_ids = train_ids[:8], train_ids[8:]
    srv, base = serve(root)
    try:
        with SafeBrowser(base) as b:
            X, y = _collect(b, base, root, fit_ids, lambda: _Explore(0.5), 4, seed=1)
            best = None
            for hidden in ((32,), (64, 32)):
                for alpha in (1e-4, 1e-2):
                    m = _fit(X, y, hidden, alpha)
                    # one DAgger round: add states the learned policy itself visits
                    X2, y2 = _collect(b, base, root, fit_ids, lambda m=m: LearnedStrategy(m), 2,
                                      seed=2)
                    m2 = _fit(np.vstack([X, X2]), np.concatenate([y, y2]), hidden, alpha)
                    val = _policy_score(b, base, root, val_ids, LearnedStrategy(m2))
                    print(f"  hidden={hidden} alpha={alpha}: validation discovery {val:.3f}")
                    if best is None or val > best["val_discovery"]:
                        best = {"hidden": list(hidden), "alpha": alpha,
                                "val_discovery": round(val, 3),
                                "model": m2, "n_examples": int(len(y) + len(y2)),
                                "pos_rate": round(float(np.mean(np.concatenate([y, y2]))), 3)}
            val_fixed = _policy_score(b, base, root, val_ids, FixedRuleStrategy())
            val_random = _policy_score(b, base, root, val_ids, RandomStrategy())
    finally:
        srv.shutdown()
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best.pop("model"), MODEL_PATH)
    meta = {**best, "val_fixed_rule": round(val_fixed, 3), "val_random": round(val_random, 3),
            "fit_sites": len(fit_ids), "val_sites": len(val_ids), "budget": BUDGET,
            "trained_on": "TRAIN-theme sites only (shop, bank, lottery, courier, jobs)",
            "features": "observable only: link text (hashed words + simple cues), page region, "
                        "form fields, visit counts, depth, budget used, behaviours found"}
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--root", default="data/blackbox_sites")
    a = ap.parse_args()
    print(json.dumps(train(Path(a.root)), indent=2))
