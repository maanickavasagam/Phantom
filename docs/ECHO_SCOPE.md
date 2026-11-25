# ECHO — Scope (Day 13)

Voice incident intelligence: turn permitted audio recordings into a transcript
plus structured claims, each traceable to the exact transcript passage it came
from.

## Supported audio
- **Formats:** WAV, MP3, FLAC, M4A (converted internally to 16kHz mono WAV for
  the speech engine).
- **Length:** up to 10 minutes per recording for v1.
- **Channels:** mono or stereo (stereo is downmixed).
- **Language:** English for v1. Tamil is a stretch goal, dependent on which
  speech-recognition engine we land on (see "Open decisions" below).
- **Source:** permitted recordings only — consented, team-recorded scripted
  calls, and public research datasets with a usable licence (see
  `docs/DATASET_AUDIT.md`). No scraped or real victim recordings.

## Target fields (claims to extract)
Each claim record: `{type, value, start_s, end_s, supporting_text}`.

| Type | Example | Notes |
|---|---|---|
| `amount` | "transfer ₹50,000" | Numeric value + currency if stated |
| `deadline` | "within the next hour" | Relative or absolute time reference |
| `payment_request` | "pay via gift card" | Flag unusual payment channels |
| `name` | "this is Officer Sharma" | Self-identification claims |
| `credential_request` | "read me the OTP" | OTP, password, PIN, card number asks |

## Known limitations (to document in every ECHO result)
- Speech recognition accuracy drops on noisy audio, overlapping speech, and
  non-native accents not well represented in training data.
- Claim extraction is pattern/heuristic-based in v1, not a trained classifier —
  expect misses on indirect phrasing.
- No speaker diarization in v1 (can't yet distinguish caller from recipient in
  a two-person recording).

## Success criteria (from SCOPE.md)
- WER ≤ 15% on clean English audio (LibriSpeech test-clean as reference).
- Claim-extraction precision/recall/F1 ≥ 0.70 on the team-recorded scripted set.
- Degrades gracefully (lower confidence, not silent failure) on noisy audio.

## Decisions (resolved Week 4)
- **Speech-recognition engine: local Whisper** (faster-whisper, `base.en`, int8 on CPU).
  It runs offline, so no audio leaves the machine and no licensing/privacy review of a
  hosted API is needed. The model is configurable via `ECHO_WHISPER_MODEL`.
- **Tamil: deferred to future work.** v1 is English only.

Results: `docs/ECHO_EVAL.md`.

## Open decisions for Week 3 (original text, kept for the record)
- Speech-recognition engine: local Whisper (offline, no data leaves the
  machine, heavier compute) vs. a hosted API (faster to integrate, sends audio
  off-device — needs a licensing/privacy check before use on any recording
  that isn't purely synthetic/team-recorded).
- Whether Tamil support is in scope for the v1 demo or deferred entirely.
