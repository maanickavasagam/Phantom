# ECHO Evaluation (Week 4)

Pipeline: audio → ffmpeg 16 kHz mono → local Whisper `base.en` (faster-whisper, int8, CPU)
with word timestamps → rule-based claim extraction → social-engineering indicators →
`ToolResult` (findings + transcript and claims in `artifacts`).

Decisions resolved from `ECHO_SCOPE.md`:
- **Engine: local Whisper.** Audio never leaves the machine, there's no API key or cost,
  and it's reproducible offline.
- **Tamil: deferred.** English only for v1; Tamil is future work.

## 1. Speech recognition: LibriSpeech test-clean

```
python -m phantom.echo.evaluate librispeech --root data/raw/LibriSpeech/test-clean --n 200
```

| Utterances | Audio | **WER** | Speed (CPU, int8) | v1 target |
|---|---|---|---|---|
| 200 (random, fixed seed) | 27.0 min | **5.3%** | 0.15 × real time (≈ 6.5× faster than real time) | ≤ 15% ✓ |

The text is normalised (lowercase, no punctuation, digits for one–ten) before scoring. Raw
results: `docs/echo_eval_librispeech.json`.

## 2. Claims and scam detection: scripted calls

```
python -m phantom.echo.make_calls --out data/echo_calls --fills 3
python -m phantom.echo.evaluate calls --split dev                  # tuning
python -m phantom.echo.evaluate calls --split test --json-out docs/echo_eval_calls_test.json   # run once
```

**The dataset.** 84 two-speaker calls generated with the offline Windows voices: 8 scam
families (digital arrest, bank KYC/OTP, gift-card "boss", tech support, lottery,
courier/customs, electricity disconnection, crypto) and 6 genuine families. The genuine
calls include hard negatives:
- a bank reminder that *mentions* the OTP/CVV only to warn;
- a friend asking for UPI money;
- an insurance premium with a deadline.

Each family has two wordings. Wording 0 is the dev split, wording 1 is the test split, so
test calls never reuse dev phrasing. Test calls are also scored with white noise at 10 and
5 dB SNR, and through a simulated 8 kHz phone line.

### Held-out test results (current pipeline: second test run, after a dev-only fix)

| Audio | Calls | Claim P | Claim R | **Claim F1** | Scam P / R | FP rate (genuine) | WER |
|---|---|---|---|---|---|---|---|
| **Clean** | 42 | 0.78 | 0.92 | **0.85** | **1.00 / 1.00** | **0.00** | 4.6% |
| Noise 10 dB | 42 | 0.75 | 0.86 | 0.80 | 1.00 / 1.00 | 0.00 | 3.7% |
| Noise 5 dB | 42 | 0.75 | 0.86 | 0.80 | 1.00 / 1.00 | 0.00 | 4.7% |
| **Phone line (8 kHz)** | 42 | 0.75 | 0.86 | **0.80** | **1.00 / 1.00** | **0.00** | **3.3%** |

Targets (`SCOPE.md`): claim F1 ≥ 0.70, met in **every** condition; degrade gracefully on noisy
audio.

**Phone-line fix (VAD threshold).** The first test run (`docs/echo_eval_calls_test_v1.json`)
scored claim F1 0.59, scam recall 11/24 and WER 35.9% on phone-band audio. The cause was not
recognition: Whisper's voice-activity filter (Silero VAD, default threshold 0.5) dropped
whole sentences of band-limited speech as "silence". We diagnosed and tuned this **only on
phone-band copies of the dev calls** (`make_calls --dev-variants`):

| Dev phone copies | WER | Claim F1 | Scams caught |
|---|---|---|---|
| VAD threshold 0.5 (default) | 46.9% | 0.51 | 10 / 24 |
| VAD off | 5.9%* | – | – |
| **VAD threshold 0.2, min silence 1 s** | **3.2%** | **0.94** | **24 / 24** |

\*On 10 calls. We kept a low-threshold VAD rather than turning it off, because skipping true
silence prevents Whisper from inventing text in gaps.

We then re-ran the test split once. This is a second look at the test set, so both runs are
reported:

| Test split | Run 1 (original) | Run 2 (VAD fix) |
|---|---|---|
| Phone: claim F1 / scams caught / WER | 0.59 / 11 of 24 / 35.9% | **0.80 / 24 of 24 / 3.3%** |
| Clean: claim F1 / WER | 0.83 / 2.2% | 0.85 / 4.6% |
| Noise 10 dB: claim F1 | 0.80 | 0.80 |
| Noise 5 dB: claim F1 | 0.80 | 0.80 |
| LibriSpeech WER | 5.3% | 5.3% (unchanged) |

Clean-audio WER rose slightly (2.2% → 4.6%). The more sensitive filter now passes a little
non-speech audio, which Whisper occasionally transcribes. Claim accuracy still improved.

Dev split, for reference (tuning set): clean claim F1 0.97, phone claim F1 0.94, scam
detection 24/24 in both, 0 false alarms.

### Label audit (stated separately, not the headline)

Several "false positive" names on the test split are organisations the test script literally
says ("this message is from Star Health", "calling from Sunrise Hospital", "the DHL parcel
department"), but which the generator forgot to include in the expected claims. That was an
annotation error in *our* ground truth. With only those labels added (listed in
`docs/echo_eval_label_audit.json`), claim F1 is:

| Audio (run 2) | Claim F1 (as recorded) | Claim F1 (labels corrected) |
|---|---|---|
| Clean | 0.85 | 0.93 |
| Noise 10 dB | 0.80 | 0.90 |
| Noise 5 dB | 0.80 | 0.89 |
| Phone line | 0.80 | 0.89 |

The as-recorded numbers above remain the official result. The generator is fixed for future
runs (see next steps).

### How the verdict works

- **Strong indicators**, enough on their own: a credential request (OTP, PIN, CVV, password,
  card or account number), or an unusual payment channel (gift cards, crypto, wire transfer,
  QR code, remote-access apps such as AnyDesk or TeamViewer).
- **Weak indicators**, which only count together: urgency, authority claims, threats,
  secrecy, too-good-to-be-true lures, and payment requests. Two or more different weak
  indicators, at least one of which is a manipulation tactic (authority, threat, secrecy or
  lure), make a "pressure pattern".
- **Warnings aren't requests.** "We will never ask for your OTP" is not counted as a
  credential request.

### Known limitations

- **Synthetic voices.** TTS speech is cleaner and more regular than real calls. Treat these
  numbers as an upper bound until the team records real scripted calls (as planned in the
  scope doc). The generator and evaluator work unchanged on recorded WAVs with a
  `labels.json`.
- **Phone-quality audio** is now handled with the low-threshold VAD (claim F1 0.80 on test).
  Our phone simulation is a clean band-limit, though. Real phone lines add codec artefacts and
  packet loss, which a larger model (`ECHO_WHISPER_MODEL=small.en`) would handle better.
- **Name spelling.** Whisper misspells proper nouns ("ICICI" → "ISISI", "Meera" → "Mira"). The
  evaluation scores names leniently (≥ 0.8 similarity, or the transcribed name containing the
  expected one); all other claim types must match exactly.
- **No speaker diarization.** ECHO can't tell caller from recipient, so a recipient saying
  "I'll do it tonight" can produce a deadline claim.
- **Rule-based extraction.** Indirect phrasing ("you know what to do with the code") is missed.

## Next steps

- Record 20–30 real scripted calls (team members, consented) and run the same evaluator.
- Fix the generator labels for organisation mentions, then regenerate.
- Try `small.en` on real phone recordings; add a 3-tier verdict with "inconclusive" for single
  weak indicators if reviewers want them surfaced.
