# BLACKBOX synthetic evaluation

BLACKBOX is limited to the six local scenarios served at
`/synthetic/{scenario}/start`. Each scenario has one harmless planted
behaviour on a clue page, one safe page, and a documented ground-truth marker.
No live website, real form data, payment, or credential is used.

The action ranker is a small scikit-learn MLP trained on a labelled action
phrase catalogue using fee, privacy, deadline, and generic details vocabulary.
Subscription and identity are held out by scenario family. The ranker uses
observable link text, and the fixed baseline takes the first permitted link.
The random baseline uses a seeded PRNG. Each policy gets one action and the
same two held-out scenarios, with ten seeds per scenario.

Run `GET /blackbox/eval` while the API is serving locally to reproduce the
comparison. Playwright captures each page and the evidence store hashes the
screenshots and action traces.

| Strategy | Discovery / coverage | False positives | Mean actions to discovery | Mean runtime / trial |
|---|---:|---:|---:|---:|
| Fixed | 0/20 (0.00) | 0 | — | 1.453 s |
| Random | 8/20 (0.40) | 0 | 1.0 | 1.329 s |
| Learned | 20/20 (1.00) | 0 | 1.0 | 1.318 s |

Measured locally on Windows on 25 November 2026, with Playwright Chromium
headless shell and the action budget fixed at one. This is a small, simple
synthetic benchmark. The difference demonstrates that the ranker can select
the planted clue in these cases; it does not establish generalisation to
unseen website structures or live sites. The fixed baseline is deliberately
simple, and all scenarios present the clue as one of two links. Wider
scenario families, richer state/history features, and independent reviewers
are needed before making a stronger research claim.
