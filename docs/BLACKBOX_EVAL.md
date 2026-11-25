# BLACKBOX Evaluation (Days 20–27, built Week 4)

**Research question (`SCOPE.md`):** under a fixed investigation budget, does adaptive browser
exploration discover more known planted behaviours than fixed-rule and random exploration,
while keeping false positives and runtime acceptable?

**Short answer:**
- The learned ranker is far better than random exploration (significant).
- It is slightly better than a carefully hand-written rule strategy (+4 points of discovery),
  but that difference is **not statistically significant** (p = 0.14).
- All strategies had zero false positives, at the same runtime.

## Setup

```
python -m phantom.blackbox.sites --out data/blackbox_sites --n-train 40 --n-test 24
python -m phantom.blackbox.crawl_check --split test --limit 24        # detector validation
python -m phantom.blackbox.ranker train --root data/blackbox_sites     # train + model selection
python -m phantom.blackbox.evaluate --trials 5 --json-out docs/blackbox_eval.json
```

**Synthetic sites.** 64 static websites of 20–29 pages, served only on 127.0.0.1.
- **Planted behaviours** (3–4 per site, 1–3 clicks deep, harmless): credential-harvesting form
  (OTP / CVV / ATM PIN), fee demand (UPI / gift card), fake countdown, forced `.apk` download,
  silent redirect to a login page, a form that sends personal data to another host, and a
  scareware pop-up with a phone number. Some appear only after submitting a harmless-looking
  form.
- **Decoys on every site** (benign look-alikes, used to measure false positives): a normal
  username/password login, a PDF brochure, a contact phone number, a newsletter form, "sale ends
  Sunday" text and an external partner link.
- **No answer leaks.** Pages are named `pN.html`, so URLs don't reveal behaviours, and the ground
  truth is stored outside the served folder.
- **Train/test split by theme.**
  - Train: shop, bank, lottery, courier, jobs (40 sites; 32 used to fit, 8 for validation).
  - **Test: government benefits, crypto, tech support** (24 sites, 85 planted behaviours).
    Their vocabulary and button texts never appear in training.

**Safety.**
- Playwright headless Chromium with an allowlist (only the local origin); every other request is
  aborted and logged.
- Downloads are never accepted, and file links are observed but never opened.
- Forms get dummy data only, and only same-origin forms are submitted.
- 5 s timeout per action and a hard action budget (12).

**Detectors.** Explainable rules over the observed page state. They were validated with an
exhaustive crawl, which found **141/141 planted behaviours on the training sites and 85/85 on
the test sites, with 0 false positives on the decoys**. So the strategy comparison below
measures exploration only. Caveat: we wrote both the sites and the detectors, so near-perfect
detection is expected here. Real websites would need broader detectors.

**Strategies (same budget, same sites, same per-trial link order).**
- **Random:** a uniformly random permitted action.
- **Fixed rules:**
  - unvisited pages first;
  - forms (except plain login/newsletter forms);
  - call-to-action keywords, chosen from the *training* themes only;
  - buttons > body > menu > footer.
- **Neural ranker:** an MLP over observable features only (link text as hashed words plus simple
  cues, page region, form fields, visit counts, depth, budget used). Its training labels come
  from an oracle that knows the site map: is this action a step closer to an unfound
  behaviour? It learns by imitation, plus one DAgger round (no reinforcement learning, per
  scope). It shares the fixed rule's "unvisited first" constraint. Model selection was on the 8
  validation sites:
  - first version: 48.1% vs fixed 50.6%;
  - with the unvisited-first constraint: **55.6% vs 50.6%**;
  - chosen model: MLP (32), alpha 0.01.
- **Oracle:** knows the site map. Reported as an **upper bound only**, not a competitor.

## Results: held-out test themes (24 sites × 5 trials = 120 episodes per strategy)

| Strategy | Discovery (mean ± sd) | Behaviours found | Coverage | False positives | Actions to 1st find | Episodes with no find | Time / episode |
|---|---|---|---|---|---|---|---|
| Random | 0.12 ± 0.16 | 52 / 425 | 0.26 | 0 | 5.7 | 73 / 120 | 0.57 s |
| Fixed rules | 0.41 ± 0.19 | 173 / 425 | 0.29 | 0 | 3.6 | 4 / 120 | 0.57 s |
| **Neural ranker** | **0.45 ± 0.19** | **190 / 425** | **0.34** | **0** | **3.4** | 5 / 120 | 0.58 s |
| Oracle (upper bound) | 0.71 ± 0.22 | 297 / 425 | 0.29 | 0 | 1.4 | 0 / 120 | 0.58 s |

**Paired comparison** (same site and trial; Wilcoxon signed-rank test):

| | Mean difference | Learned better / tied / worse | p-value |
|---|---|---|---|
| Neural vs **random** | **+0.33** | 91 / 29 / 0 | **< 0.001** (significant) |
| Neural vs **fixed rules** | +0.04 | 38 / 57 / 25 | **0.14** (not significant) |

**Discovery rate by behaviour type:**

| Behaviour | Random | Fixed | Neural | Oracle |
|---|---|---|---|---|
| Fake urgency | 0.11 | 0.38 | **0.78** | 0.80 |
| Data exfiltration form | 0.12 | 0.45 | **0.58** | 0.63 |
| Fee demand | 0.16 | 0.31 | **0.44** | 0.84 |
| Credential form | 0.06 | 0.38 | 0.41 | 0.53 |
| Forced download | 0.17 | 0.29 | 0.31 | 0.80 |
| Scareware pop-up | 0.15 | **0.55** | 0.42 | 0.69 |
| Deceptive redirect | 0.10 | **0.50** | 0.30 | 0.70 |

No outside requests were made in any episode: external links and forms are filtered out before
the browser could load them.

## Interpretation (honest)

- **Adaptive exploration clearly helps.** Both informed strategies find 3–4× more than random
  under the same budget, and almost never come back empty-handed (4–5 of 120 episodes vs 73).
- **Learned vs hand-written: a small, uncertain gain.**
  - The ranker finds more overall (190 vs 173) and covers more of the site.
  - It reaches its first find slightly sooner.
  - It is not reliably better: tied on 57 episodes, worse on 25. With p = 0.14 we **cannot
    claim** it beats a well-designed rule strategy.
  - It is better on some behaviour types (fake urgency, exfiltration, fee demands) and worse on
    others (redirects, pop-ups).
- **There is headroom.** The oracle reaches 0.71, so better exploration could still find much
  more within 12 actions.
- **Why the gap is small.**
  - The fixed rule's keyword prior is already strong, and the synthetic sites are regular.
  - Test themes use button texts the network never saw, so its hashed-word features help less
    than on training themes (validation 55.6% vs test 44.9%).

## Limitations

- Everything is synthetic: we built the sites, planted the behaviours and wrote the detectors.
  Real websites are messier and adversarial.
- One site generator family: train and test differ in theme, not in how pages are built.
- 24 test sites × 5 trials. The per-type rates rest on 9–17 instances each.
- The ranker imitates an oracle on our sites. It is a small action-ranking component, not
  autonomous web defence (scope guardrail).

## Next steps

- More diverse site templates (different layouts and navigation patterns) for training.
- Semantic text features (e.g. small sentence embeddings) instead of hashed words, to handle
  unseen button texts.
- More trials or sites to tighten the learned-vs-fixed comparison.
