# Dataset & Licence Audit (Day 4)

Rule: no dataset is used for training or evaluation until its licence row is marked **Confirmed**
by the module owner after reading the licence page. Splits are by **source**, never by random
page/clip, so near-duplicate documents or speakers never cross train/test.

| Module | Candidate | What it gives us | Licence (as understood) | Status | Split plan | Privacy / limitations |
|---|---|---|---|---|---|---|
| MIRAGE | FUNSD | 199 scanned forms with word boxes + labels | Non-commercial research use | To confirm | Use as OCR/layout benchmark only | Old, noisy scans; no tampering labels |
| MIRAGE | SROIE (ICDAR 2019) | Scanned receipts with text + key fields | Research use (competition data) | To confirm | Split by receipt source/store | Receipts may contain store names; no tampering labels |
| MIRAGE | Synthetic invoice set (our generator) | Benign + manipulated pairs with exact ground truth | Our own work | Confirmed | Split by template family | Synthetic layouts may be easier than real ones |
| ECHO | LibriSpeech (test-clean / test-other) | Read English speech with reference transcripts | CC BY 4.0 | To confirm | Use official test splits | Read speech, not phone-call style |
| ECHO | Mozilla Common Voice (English) | Many speakers and accents, incl. Indian English | CC0 | To confirm | Split by speaker ID | Short clips; self-reported metadata |
| ECHO | Scripted scam-call recordings (team-recorded) | Target fields: amounts, deadlines, OTP/credential asks | Our own work, consent from speakers | Planned | Split by script | Small set; actors, not real victims |
| ECHO | MUSAN noise | Background noise to create noisy test copies | CC BY 4.0 | To confirm | Noise added only to held-out copies | — |
| BLACKBOX | Local synthetic websites (our build) | Planted harmless behaviours with documented ground truth | Our own work | Planned (Day 20) | Split by scenario family | Dummy data only; localhost allowlist |

## Labels
- **MIRAGE:** document-level `benign / suspicious`, plus region-level boxes for each planted manipulation
  (`font_mismatch`, `misaligned_field`, `pasted_region`, `altered_amount`).
- **ECHO:** reference transcript; claim records `{type, value, start_s, end_s}` for
  `amount, deadline, payment_request, name, credential_request`.
- **BLACKBOX:** per scenario, list of planted behaviours with the page and action that reveal each one.
