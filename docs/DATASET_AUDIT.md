# Dataset & Licence Audit (Day 4)

Rule: no dataset is used for training or evaluation until its licence row is marked **Confirmed**
by the module owner after reading the licence page. Splits are by **source**, never by random
page/clip, so near-duplicate documents or speakers never cross train/test.

| Module | Candidate | What it gives us | Licence (as understood) | Status | Split plan | Privacy / limitations |
|---|---|---|---|---|---|---|
| MIRAGE | FUNSD | 199 scanned forms with word boxes + labels | Non-commercial research use | To confirm | Use as OCR/layout benchmark only | Old, noisy scans; no tampering labels |
| MIRAGE | SROIE (ICDAR 2019) | Scanned receipts with text + key fields | Research use (competition data) | To confirm | Split by receipt source/store | Receipts may contain store names; no tampering labels |
| MIRAGE | Find it again! (ICDAR 2023, L3i La Rochelle) — https://l3i-share.univ-lr.fr/2023Finditagain/ | 988 real scanned SROIE receipts, 163 manually forged by people, with forgery bboxes + entity types; official train/val/test split | No licence stated on the page; academic dataset, cite Martínez Tornés et al., ICDAR 2023. Research use only until confirmed | To confirm (downloaded 25 Nov 2026; test split evaluated once, see MIRAGE_EVAL.md). Images kept in gitignored data/raw/, never committed | Used official split: tuned on train/val only, test run once | Real store receipts (store names/addresses, no personal card data seen yet); forgeries are "pseudo-realistic", made by participants; receipts not invoices |
| MIRAGE | Synthetic invoice set (our generator) | Benign + manipulated pairs with exact ground truth | Our own work | Confirmed | Split by template family | Synthetic layouts may be easier than real ones |
| ECHO | LibriSpeech (test-clean / test-other) | Read English speech with reference transcripts | CC BY 4.0 (as listed on openslr.org/12) | Downloaded 25 Nov 2026 (test-clean and test-other, gitignored data/raw/); owner to confirm | Use official test splits | Read speech, not phone-call style |
| ECHO | Mozilla Common Voice (English) | Many speakers and accents, incl. Indian English | CC0 | To confirm | Split by speaker ID | Short clips; self-reported metadata |
| ECHO | Scripted scam-call recordings (team-recorded) | Target fields: amounts, deadlines, OTP/credential asks | Our own work, consent from speakers | Planned | Split by script | Small set; actors, not real victims |
| ECHO | Synthetic scripted calls (`phantom.echo.make_calls`, offline Windows TTS voices) | 84 two-speaker calls, 8 scam + 6 genuine families, exact claim labels; noisy and phone-band copies | Our own work; generated locally with built-in Windows voices | Confirmed (used Week 4) | Split by wording: wording 0 = dev, wording 1 = test | TTS is cleaner than real calls, so results are an upper bound; stand-in until team recordings exist |
| ECHO | BothBosu scam-dialogue datasets (single-agent, multi-agent, Scammer-Conversation, youtube-scam-conversations), huggingface.co/BothBosu | 4,220 labelled scam / non-scam phone dialogue transcripts; 20 are real scam calls from scam-baiter videos | Apache-2.0 (dataset cards) | Confirmed licence; downloaded 25 Nov 2026 (gitignored data/raw/bothbosu) | Official train/test splits for single- and multi-agent: rules tuned on train only; the other two never used for tuning | Mostly LLM-generated US-style scams (SSN, refund, tech support, reward); text only, no audio; some "genuine" wrong-number calls ask for SSN/card numbers |
| ECHO | MUSAN noise | Background noise to create noisy test copies | CC BY 4.0 | To confirm | Noise added only to held-out copies | — |
| BLACKBOX | Local synthetic websites (our build) | Planted harmless behaviours with documented ground truth | Our own work | Planned (Day 20) | Split by scenario family | Dummy data only; localhost allowlist |

## Labels
- **MIRAGE:** document-level `benign / suspicious`, plus region-level boxes for each planted manipulation
  (`font_mismatch`, `misaligned_field`, `pasted_region`, `altered_amount`).
- **ECHO:** reference transcript; claim records `{type, value, start_s, end_s}` for
  `amount, deadline, payment_request, name, credential_request`.
- **BLACKBOX:** per scenario, list of planted behaviours with the page and action that reveal each one.
