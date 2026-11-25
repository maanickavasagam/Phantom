"""BLACKBOX exploration loop and strategies (Days 23-26).

An episode starts on a site's home page and may take at most `budget` actions
(open a link, submit a form with dummy data, or go back). After every action
the detectors run on the new page. Strategies only see what a real investigator
would see (the page state and their own history), never the ground truth.

Strategies:
  random      uniformly random permitted action (baseline 1)
  fixed_rule  hand-written priorities: unvisited first, forms, call-to-action
              keywords, buttons > body > nav > footer (baseline 2; keywords
              chosen from the TRAIN themes only)
  learned     small neural network that scores each permitted action from
              observable features (phantom.blackbox.ranker)
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from phantom.blackbox.browser import Action, PageState, SafeBrowser
from phantom.blackbox.detectors import Detection, detect

# Train-theme call-to-action words + generic scam lexicon. Deliberately NOT tuned on
# the test themes (govt_benefits, crypto, tech_support).
CTA_WORDS = re.compile(r"claim|verify|update|reward|prize|voucher|gift|free|unlock|confirm|pay|"
                       r"release|reactivate|unblock|accept|activate|collect|win|continue|"
                       r"reschedule|book|complete|check", re.I)
_PLAIN_FORM = re.compile(r"^(?:username|password|email|user|pass)$", re.I)


@dataclass
class Context:
    """What a strategy may use: its own history, never the ground truth."""
    visited: dict[str, int] = field(default_factory=dict)  # url -> visits
    history: list[str] = field(default_factory=list)  # urls in order
    found: int = 0
    step: int = 0
    budget: int = 12
    depth: int = 0  # page depth reached by the current path


class Strategy:
    name = "base"

    def choose(self, state: PageState, actions: list[Action], ctx: Context,
               rng: random.Random) -> Action:
        raise NotImplementedError


class RandomStrategy(Strategy):
    name = "random"

    def choose(self, state, actions, ctx, rng):
        return rng.choice(actions)


def _is_plain_form(a: Action) -> bool:
    names = [f["name"] for f in (a.form or {}).get("fields", []) if f["name"]]
    return bool(names) and all(_PLAIN_FORM.match(n) for n in names)


class FixedRuleStrategy(Strategy):
    name = "fixed_rule"

    def score(self, a: Action, ctx: Context) -> float:
        if a.kind == "back":
            return 0.1
        s = 3.0 if ctx.visited.get(a.target.split("?")[0], 0) == 0 else -5.0
        if a.kind == "form":
            s += 0.5 if _is_plain_form(a) else 3.0
        if CTA_WORDS.search(a.text):
            s += 2.0
        s += {"button": 1.5, "body": 1.0, "nav": 0.5, "footer": 0.3}.get(a.region, 0.0)
        return s

    def choose(self, state, actions, ctx, rng):
        # ties keep the (per-trial shuffled) order, so repeated trials differ
        return max(actions, key=lambda a: self.score(a, ctx))


@dataclass
class StepRecord:
    step: int
    action: str
    target: str
    text: str
    url: str
    detections: list[dict]


@dataclass
class Episode:
    site: str
    strategy: str
    budget: int
    steps: list[StepRecord]
    found: dict[tuple[str, str], int]  # (type, page) -> step first found
    pages_visited: set[str]
    runtime_s: float
    blocked_requests: int = 0


def run_episode(browser: SafeBrowser, base: str, site_id: str, strategy: Strategy,
                budget: int, rng: random.Random, on_state=None) -> Episode:
    t0 = time.perf_counter()
    blocked0 = len(browser.blocked)
    ctx = Context(budget=budget)
    start = f"{base}/{site_id}/index.html"
    state = browser.load(start)
    stack: list[str] = []
    found: dict[tuple[str, str], int] = {}
    steps: list[StepRecord] = []

    def observe(st: PageState, step: int) -> list[Detection]:
        ds = detect(st, base)
        for d in ds:
            found.setdefault((d.type, d.page.rsplit("/", 1)[-1]), step)
        ctx.visited[st.url.split("?")[0]] = ctx.visited.get(st.url.split("?")[0], 0) + 1
        ctx.history.append(st.url)
        ctx.found = len(found)
        return ds

    observe(state, 0)
    for step in range(1, budget + 1):
        ctx.step = step - 1
        actions = browser.actions(state)
        rng.shuffle(actions)  # per-trial variation; strategies must not rely on order
        if stack:
            actions.append(Action("back", stack[-1], "Back", "nav"))
        if not actions:
            break
        if on_state:  # used only for training-data collection; strategies never see `found`
            on_state(state, actions, ctx, set(found))
        a = strategy.choose(state, actions, ctx, rng)
        if a.kind == "back":
            stack.pop()
            ctx.depth = max(0, ctx.depth - 1)
        else:
            stack.append(state.url)
            ctx.depth += 1
        try:
            state = browser.load(a.target)
        except Exception as e:  # noqa: BLE001 - a broken page costs the action, not the run
            steps.append(StepRecord(step, a.kind, a.target, a.text, "", [{"error": str(e)[:100]}]))
            continue
        ds = observe(state, step)
        steps.append(StepRecord(step, a.kind, urlparse(a.target).path, a.text,
                                urlparse(state.url).path, [d.__dict__ for d in ds]))
    pages = {urlparse(u).path.rsplit("/", 1)[-1] for u in ctx.history}
    return Episode(site_id, strategy.name, budget, steps, found, pages,
                   time.perf_counter() - t0, len(browser.blocked) - blocked0)
