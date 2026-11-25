# PHANTOM AI — Scope Agreement (Day 1)

## Objective
A controlled investigation prototype that examines suspicious documents (MIRAGE), permitted
audio recordings (ECHO) and local synthetic websites (BLACKBOX). It preserves traceable
evidence, reports uncertainty and produces reports for human review.

## Ownership
| Module | Primary owner | Shared responsibilities |
|---|---|---|
| MIRAGE — visual document forensics | Maanickavasagam | Shared schemas, evidence store |
| ECHO — voice incident intelligence | Aadhityan | Dataset audit, API skeleton |
| BLACKBOX — controlled browser investigation + action ranker | Aswin Karthik | Repo tooling, CI, safety rules |
| Agentic coordinator | All (built during integration) | — |

## Supported inputs (v1)
- **MIRAGE:** PDF (≤ 20 pages), PNG, JPG/JPEG, TIFF. English text only.
- **ECHO:** WAV, MP3, FLAC, M4A; mono or stereo; ≤ 10 minutes; English (Tamil as a stretch goal).
- **BLACKBOX:** Local synthetic sites served from `localhost` only, from an allowlist.

## Success criteria
| Module | Metric | v1 target |
|---|---|---|
| MIRAGE | F1 on held-out document sources | ≥ 0.70 |
| MIRAGE | Runtime per page | ≤ 5 s on CPU |
| ECHO | WER on clean English audio | ≤ 15% |
| ECHO | Claim extraction F1 | ≥ 0.70 |
| BLACKBOX | Known-behaviour discovery vs. random baseline (same budget) | Higher, over ≥ 10 repeated trials |
| All | Every finding links to an evidence ID | 100% |

## Explicit exclusions
- Training large OCR, vision or speech models from scratch.
- Browsing live/public websites, or real credentials or personal data.
- Reinforcement learning (not required for the action ranker).
- Autonomous accusations: outputs are investigation support for a human reviewer.
- Paid LLM APIs in the core workflow.

## Ways of working
- One feature branch per task; PR with at least one reviewer before merging to `main`.
- Each working day ends with a visible artifact (code, test, audit, result or reviewed PR).
