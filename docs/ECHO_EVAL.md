# ECHO Evaluation (Week 4)

Pipeline: audio → ffmpeg 16 kHz mono → local Whisper `base.en` (faster-whisper, int8, CPU)
with word timestamps → rule-based claim extraction → social-engineering indicators →
`ToolResult` (findings + transcript and claims in `artifacts`).

Decisions resolved from `ECHO_SCOPE.md`:
- **Engine: local Whisper.** Audio never leaves the machine, there's no API key or cost,
  and it's reproducible offline.
- **Tamil: deferred.** English only for v1; Tamil is future work.

## 1. Speech recognition: LibriSpeech test-clean and test-other

```
python -m phantom.echo.evaluate librispeech --root data/raw/LibriSpeech/test-clean --n 200
```

| Utterances | Audio | **WER** | Speed (CPU, int8) | v1 target |
|---|---|---|---|---|
| test-clean: 200 (random, fixed seed) | 27.0 min | **5.3%** | 0.13–0.15 × real time (≈ 7× faster than real time) | ≤ 15% ✓ |
| test-other (harder speakers): 200 | 23.0 min | **10.7%** | 0.14 × real time | ≤ 15% ✓ |

The text is normalised (lowercase, no punctuation, digits for one–ten) before scoring. Raw
results: `docs/echo_eval_librispeech.json`.

## 2. Scam detection on public datasets (independent of our own data)

Our scripted calls (section 3) were written by us, and the rules were tuned on their dev half.
That can't show whether ECHO generalises, so we also evaluate on public scam-dialogue datasets
written by other researchers (huggingface.co/BothBosu, Apache-2.0):

| Dataset | Size | What it is |
|---|---|---|
| single-agent-scam-conversations | 1,600 (official 1,280 train / 320 test) | Llama-3-70B dialogues: SSN, refund, tech-support and reward scams vs delivery, insurance, appointment and wrong-number calls |
| multi-agent-scam-conversation | 1,600 (1,280 / 320) | Two AI agents talking, 7 victim personalities |
| Scammer-Conversation | 1,000 | A different generator (Gretel); scam, scam-baiting and normal calls |
| youtube-scam-conversations | 20 | **Real** scam calls transcribed from scam-baiter videos (all scams) |

These are text transcripts, so this measures claim extraction, indicators and the verdict
rule, not speech recognition (section 1 covers that). Both speakers' turns are kept, as in
a real transcript.

```
python -m phantom.echo.eval_public --root data/raw/bothbosu --split test
```

**Protocol.**
- **Stage A:** ran the frozen pipeline on everything.
- **Stage B:** fixed rules using only the **train** halves of the two split datasets. The
  dev scripted calls were re-checked for regressions (unchanged: claim F1 0.97, 24/24).
- **Final:** ran once on the **test** halves and on the two sets never used for tuning.

| Dataset | Stage A (frozen, on the full dataset) | **Final (on the test half; rules fixed on train only)** | False alarms (final) |
|---|---|---|---|
| Single-agent | P 0.89 · R 0.57 · F1 0.69 (1,600) | **P 0.93 · R 0.85 · F1 0.89** (320) | 6.2% |
| Multi-agent | P 0.89 · R 0.77 · F1 0.82 (1,600) | **P 0.97 · R 0.90 · F1 0.94** (320) | 2.5% |
| Scammer-Conversation (never tuned on) | R 0.03 | R 0.03 | 0% |
| **Real YouTube scams** (never tuned on) | 6 / 20 | 6 / 20 | n/a |

**What stage A revealed: the rules had overfit to our own Indian-scam style.** They knew OTP,
Aadhaar and UPI, but not SSN, date-of-birth, "verify your account", virus/remote-access
tech-support pretexts or unsolicited refunds. Stage A also found real bugs: "I'm fine"
matched the threat rule, "you won't" matched the prize lure, and a careful victim asking "what
are the last four digits of *my* card?" counted as a credential request. Stage B fixed these as
general rules:
- **Secrets vs identity data.** Secrets (OTP, PIN, password, CVV, full card number, SSN, Aadhaar)
  are a strong indicator on their own. Identity data (account number, date of birth) is weak,
  because genuine services ask for it too.
- **Remote access** now includes "access to your computer" and screen sharing.
- **Tech-support threats** (virus, "your computer is hacked") and **refund lures** are recognised.
- **US authorities** (IRS, SSA, FTC) count as authority claims.

**Remaining false alarms** on the single-agent test half are almost all "wrong number" calls
labelled genuine, in which the LLM-written caller asks for the person's **social security or
credit card number**. ECHO flags them on purpose: a stranger asking for those is the red flag it
exists to catch.

**Where it still fails, and why.**
- **Scammer-Conversation:** mostly short scam-*baiting* calls where the target refuses after the
  opening line. The scammer never reaches a credential or payment request, so there is only one
  weak hint, and ECHO's verdict needs two tactics by design.
- **Real YouTube calls (6/20):** real scammers are slower and subtler (median 442 words), e.g. a
  fake "$400 Geek Squad renewal" pretext leading to "are you in front of your computer?". A
  keyword-rule system has a ceiling here.

Both point to the same next step: add a learned classifier trained on the public train splits
alongside the explainable rules, and surface single weak indicators as "inconclusive".

## 2b. Learned classifier alongside the rules

`phantom/echo/scam_classifier.py` is a TF-IDF (word 1–2 grams) + logistic regression model,
trained **only on the public train halves** (2,560 dialogues). Speaker labels are removed, and
money and digits are masked, so "$" can't become a scam feature. The threshold comes from 5-fold
cross-validation on the train data, capped at 2% false positives. The model file is
`phantom/echo/models/scam_clf.joblib`, and every score is explained by the transcript phrases
that raised it most.

```
python -m phantom.echo.scam_classifier train --root data/raw/bothbosu --dev-calls ""
python -m phantom.echo.eval_hybrid --json-out docs/echo_eval_hybrid.json
```

**Domain shift found before shipping.** On its own, the classifier flagged **half of our genuine
Indian calls** (bank reminders, a friend asking for UPI money, insurance). It learned "money +
bank talk = scam" from US data with no ordinary Indian money conversations. Adding our 42 dev
calls to training (weighted ×20) only "memorised" them: the test-split genuine calls, with new
wording, were still 50% false alarms. So we didn't train on them. Instead, the **dev calls, unseen
by the classifier, were used to choose how to combine it with the rules**:

- **Suspicious:** the rules fire, **or** the classifier fires *and* the rules found supporting
  evidence (a strong indicator, or authority / threat / secrecy / lure / identity request).
  Dev result: 24/24 scams, 0 false alarms; the classifier alone gave 9/18 false alarms.
- **Inconclusive (review):** the classifier fires with no supporting rule evidence. A human
  listens; nobody is accused.

**Final held-out results (single run, nothing here used for training or choosing the rule):**

| Held-out set | Rules only | **Rules + classifier: "suspicious"** | Flagged at least for review | False alarms ("suspicious") |
|---|---|---|---|---|
| Single-agent test half (320) | F1 0.89 | **F1 0.95** (R 0.96) | recall 1.00 | 6.9% |
| Multi-agent test half (320) | F1 0.94 | **F1 0.99** (R 1.00) | recall 1.00 | 2.5% |
| Scammer-Conversation (1,000) | recall 0.03 | **recall 0.59** | **recall 1.00** | 0% |
| **Real YouTube scams (20)** | 6 / 20 | **12 / 20** | **20 / 20** | n/a |
| Our Indian scripted test calls (42) | 24/24, 0 FP | **24/24, 0 FP** | 24/24 | 0% (but 9/18 genuine go to "review") |

What this means:
- On real scam calls, detection doubled, and every real scam is at least routed to a reviewer.
- Nothing got worse for our own calls' "suspicious" verdict.
- Half of genuine Indian money conversations now get a "review" flag. That's the cost of a
  classifier trained on US data, and labelled Indian call data is the fix.
- Rules + classifier on the Scammer-Conversation set at 0% false alarms: that set's genuine
  calls are very different in style, so it says little about false alarms elsewhere.

## 3. Claims and scam detection: scripted calls

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
