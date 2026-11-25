"""Generate a scripted call set for ECHO evaluation (Week 4).

Stand-in for the team-recorded scripted calls planned in docs/ECHO_SCOPE.md:
two-speaker calls synthesized with the offline Windows voices (System.Speech),
with exact ground truth (scam/benign label, reference transcript, claims).

- 8 scam families and 6 genuine-call families. Genuine calls deliberately
  contain hard negatives: a bank reminder that *mentions* the OTP ("we will
  never ask for it"), a friend asking for UPI money, an insurance renewal with
  a deadline.
- Every family has two phrasings. Phrasing 0 -> dev split, phrasing 1 -> test
  split, so test calls never reuse dev wording ("split by script").
- Test calls also get noisy copies (white noise at 10 and 5 dB SNR) and a
  telephone-band copy (300-3400 Hz, 8 kHz).

TTS speech is cleaner and more regular than real phone calls; treat results as
an upper bound until real recorded calls are available.

    python -m phantom.echo.make_calls --out data/echo_calls --fills 3
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

CALLER, RECIPIENT = "Microsoft David Desktop", "Microsoft Zira Desktop"

# Slot pools. Spoken forms are what the voice reads; keys are the expected claim keys.
OFFICERS = [("Officer", "Sharma"), ("Inspector", "Verma"), ("Officer", "Khan"), ("Inspector", "Reddy")]
AUTH_ORGS = ["Cyber Crime Department", "Central Bureau of Investigation", "Customs Department",
             "Mumbai Police"]
BANKS = ["State Bank", "HDFC Bank", "ICICI Bank", "Axis Bank"]
AMOUNTS = [(25000, "25,000"), (50000, "50,000"), (75000, "75,000"), (12500, "12,500"),
           (199000, "1,99,000"), (8000, "8,000")]
SMALL_AMOUNTS = [(300, "300"), (450, "450"), (1200, "1,200"), (2500, "2,500")]
HOURS = [(2, "two"), (3, "three"), (1, "one")]
MINUTES = [(30, "thirty"), (15, "fifteen"), (20, "twenty")]
FIRST_NAMES = ["Priya", "Rahul", "Anita", "Vikram", "Meera", "Arjun"]


def _scam_families():
    """Each family returns (turns, claims) for phrasing v in {0, 1}."""

    def digital_arrest(r, v):
        t, n = r.choice(OFFICERS)
        org = r.choice(AUTH_ORGS)
        amt, amt_s = r.choice(AMOUNTS)
        h, h_s = r.choice(HOURS)
        if v == 0:
            turns = [("C", f"Hello, this is {t} {n} calling from the {org}."),
                     ("R", "Yes, what is this about?"),
                     ("C", "A parcel in your name contains illegal drugs. There is an arrest warrant against you."),
                     ("C", f"To clear your name you must transfer {amt_s} rupees to our verification account within {h_s} hours."),
                     ("C", "Do not tell anyone about this call and do not disconnect.")]
        else:
            turns = [("C", f"Good afternoon. My name is {t} {n}, I am speaking on behalf of the {org}."),
                     ("R", "Okay, how can I help?"),
                     ("C", "Your Aadhaar number has been linked to money laundering. You will be arrested today."),
                     ("C", f"You need to send {amt_s} rupees as a security deposit in the next {h_s} hours."),
                     ("C", "Keep this confidential, it is a secret investigation.")]
        claims = [("name", n.lower()), ("name", org.lower()), ("amount", str(amt)),
                  ("deadline", f"{h} hour"), ("payment_request", "payment")]
        return turns, claims

    def bank_kyc(r, v):
        bank = r.choice(BANKS)
        m, m_s = r.choice(MINUTES)
        if v == 0:
            turns = [("C", f"Hello madam, I am calling from {bank} customer care."),
                     ("C", f"Your account will be blocked within {m_s} minutes because your KYC is not updated."),
                     ("R", "Oh no, what should I do?"),
                     ("C", "I have sent a code to your phone. Please read me the OTP to verify your identity.")]
            claims = [("name", bank.lower()), ("deadline", f"{m} minute"), ("credential_request", "otp")]
        else:
            turns = [("C", f"Sir, this call is from {bank} security team."),
                     ("C", "Your debit card has been blocked due to suspicious activity."),
                     ("R", "What do I need to do?"),
                     ("C", f"Tell me your card number and the CVV on the back, and we will unblock it in {m_s} minutes.")]
            claims = [("credential_request", "card number"), ("credential_request", "cvv"),
                      ("deadline", f"{m} minute")]
        return turns, claims

    def gift_card_boss(r, v):
        name = r.choice(FIRST_NAMES)
        amt, amt_s = r.choice(AMOUNTS[:4])
        if v == 0:
            turns = [("C", f"Hi, this is {name}, your manager. I am in a meeting and need a favour."),
                     ("C", f"Buy Google Play gift cards worth {amt_s} rupees and send me the codes immediately."),
                     ("R", "Is everything alright?"),
                     ("C", "It is urgent. Don't tell anyone in the office.")]
        else:
            turns = [("C", f"Hello, I'm {name} from the director's office."),
                     ("C", f"The director needs Amazon gift cards for {amt_s} rupees right now for a client."),
                     ("C", "Please send the card numbers on this call, it is very urgent.")]
        claims = [("name", name.lower()), ("amount", str(amt)), ("payment_request", "gift card"),
                  ("deadline", "immediately" if v == 0 else "right now")]
        return turns, claims

    def tech_support(r, v):
        amt, amt_s = r.choice(SMALL_AMOUNTS)
        if v == 0:
            turns = [("C", "Hello, I am calling from Microsoft technical support. Your computer has a virus."),
                     ("R", "Really? I didn't notice anything."),
                     ("C", "Please download AnyDesk so I can fix it remotely."),
                     ("C", f"After that you must pay {amt_s} rupees for the security licence.")]
            claims = [("name", "microsoft technical support"), ("payment_request", "remote access app"),
                      ("amount", str(amt)), ("payment_request", "payment")]
        else:
            turns = [("C", "This is the Windows help desk. We detected hackers on your laptop."),
                     ("C", "Install TeamViewer now and give me the access code."),
                     ("R", "Okay, one moment."),
                     ("C", f"The repair fee is {amt_s} rupees, pay it by UPI after we finish.")]
            claims = [("payment_request", "remote access app"), ("amount", str(amt)),
                      ("payment_request", "upi"), ("deadline", "now")]
        return turns, claims

    def lottery(r, v):
        amt, amt_s = r.choice(AMOUNTS)
        fee, fee_s = r.choice(SMALL_AMOUNTS)
        if v == 0:
            turns = [("C", f"Congratulations! You have won {amt_s} rupees in the lucky draw."),
                     ("R", "Wow, really?"),
                     ("C", f"To release the prize you must pay a processing fee of {fee_s} rupees by Google Pay today itself.")]
            claims = [("amount", str(amt)), ("amount", str(fee)), ("payment_request", "upi"),
                      ("deadline", "today itself")]
        else:
            turns = [("C", f"Hello, your mobile number has won a prize of {amt_s} rupees."),
                     ("C", f"Send {fee_s} rupees tax through PhonePe before tonight to claim it."),
                     ("R", "Who is this?"),
                     ("C", "Do not tell your family, they will ask for a share.")]
            claims = [("amount", str(amt)), ("amount", str(fee)), ("payment_request", "upi"),
                      ("deadline", "tonight")]
        return turns, claims

    def courier_customs(r, v):
        t, n = r.choice(OFFICERS)
        amt, amt_s = r.choice(AMOUNTS)
        if v == 0:
            turns = [("C", "Hello, I am calling from FedEx courier. A package in your name was stopped by customs."),
                     ("C", f"I am connecting you to {t} {n}."),
                     ("C", f"This is {t} {n}. The package has illegal items. You will face legal action."),
                     ("C", f"Pay a fine of {amt_s} rupees by bank transfer immediately to close the case.")]
            claims = [("name", "fedex courier"), ("name", n.lower()), ("amount", str(amt)),
                      ("payment_request", "bank transfer"), ("deadline", "immediately")]
        else:
            turns = [("C", "Good morning, this call is from the DHL parcel department."),
                     ("C", "Your shipment was seized at the airport and a police case has been filed."),
                     ("R", "But I didn't send any parcel."),
                     ("C", f"You can settle it now. Transfer {amt_s} rupees through NEFT within one hour.")]
            claims = [("amount", str(amt)), ("payment_request", "bank transfer"), ("deadline", "1 hour")]
        return turns, claims

    def electricity(r, v):
        amt, amt_s = r.choice(SMALL_AMOUNTS)
        if v == 0:
            turns = [("C", "Dear customer, I am calling from the electricity board."),
                     ("C", "Your power connection will be disconnected tonight because your last bill is not updated."),
                     ("C", f"Pay {amt_s} rupees by scanning the QR code I am sending you on WhatsApp.")]
            claims = [("name", "electricity board"), ("deadline", "tonight"), ("amount", str(amt)),
                      ("payment_request", "qr code")]
        else:
            turns = [("C", "This is a final notice from the power department."),
                     ("C", f"Your connection is being disconnected in thirty minutes. Pay {amt_s} rupees now."),
                     ("R", "Can I pay at the office?"),
                     ("C", "No, install the Quick Support app and I will help you pay.")]
            claims = [("deadline", "30 minute"), ("amount", str(amt)), ("payment_request", "payment"),
                      ("payment_request", "remote access app")]
        return turns, claims

    def crypto(r, v):
        amt, amt_s = r.choice(AMOUNTS)
        name = r.choice(FIRST_NAMES)
        if v == 0:
            turns = [("C", f"Hi, my name is {name}, I am an investment advisor."),
                     ("C", f"If you invest {amt_s} rupees in Bitcoin today, it will double in one week."),
                     ("R", "How do I invest?"),
                     ("C", "Send the money to my crypto wallet right now, the offer closes soon.")]
            claims = [("name", name.lower()), ("amount", str(amt)), ("payment_request", "crypto"),
                      ("deadline", "right now")]
        else:
            turns = [("C", "Hello, I'm calling about a special trading scheme."),
                     ("C", f"Transfer {amt_s} rupees in USDT and you will earn ten percent every day."),
                     ("C", "Don't tell your bank about it, they will try to stop you.")]
            claims = [("amount", str(amt)), ("payment_request", "crypto")]
        return turns, claims

    return {"digital_arrest": digital_arrest, "bank_kyc": bank_kyc, "gift_card_boss": gift_card_boss,
            "tech_support": tech_support, "lottery": lottery, "courier_customs": courier_customs,
            "electricity": electricity, "crypto": crypto}


def _benign_families():
    def bank_reminder(r, v):
        bank = r.choice(BANKS)
        amt, amt_s = r.choice(SMALL_AMOUNTS)
        if v == 0:
            turns = [("C", f"Hello, this is an automated reminder from {bank}."),
                     ("C", f"Your credit card bill of {amt_s} rupees is due by tomorrow."),
                     ("C", "Please remember, we will never ask you to share your OTP or PIN over the phone.")]
            claims = [("name", bank.lower()), ("amount", str(amt)), ("deadline", "tomorrow")]
        else:
            turns = [("C", f"Good evening, this message is from {bank}."),
                     ("C", f"A payment of {amt_s} rupees is due on your loan account this month."),
                     ("C", "For your safety, do not share your password or CVV with anyone.")]
            claims = [("amount", str(amt))]
        return turns, claims

    def doctor(r, v):
        name = r.choice(FIRST_NAMES)
        if v == 0:
            turns = [("C", "Hello, this is the City Clinic reception."),
                     ("C", f"I am calling to confirm the appointment for {name} on Monday at ten in the morning."),
                     ("R", "Yes, that's fine, thank you.")]
        else:
            turns = [("C", "Hi, calling from Sunrise Hospital."),
                     ("C", f"Doctor Mehta has asked {name} to come for a follow-up check next week."),
                     ("R", "Okay, I will book a slot.")]
        return turns, []

    def family(r, v):
        name = r.choice(FIRST_NAMES)
        amt, amt_s = r.choice(SMALL_AMOUNTS)
        if v == 0:
            turns = [("C", f"Hi, it's {name}. Are you coming for dinner on Sunday?"),
                     ("R", f"Yes! I already bought the movie tickets for {amt_s} rupees."),
                     ("C", "Great, see you then.")]
            claims = [("name", name.lower()), ("amount", str(amt))]
        else:
            turns = [("C", "Hey, how was the trip?"),
                     ("R", f"It was lovely. The hotel cost {amt_s} rupees a night, which was fine."),
                     ("C", "Send me the photos when you can.")]
            claims = [("amount", str(amt))]
        return turns, claims

    def delivery(r, v):
        if v == 0:
            turns = [("C", "Hello, I'm calling from Blue Dart. I have a delivery for you."),
                     ("C", "I will reach your building by five p.m. Will someone be at home?"),
                     ("R", "Yes, my mother is at home.")]
            claims = [("name", "blue dart"), ("deadline", "5 pm")]
        else:
            turns = [("C", "Hi, this is the delivery partner for your grocery order."),
                     ("C", "I am outside the gate, should I leave it with the security guard?"),
                     ("R", "Yes please, thank you.")]
            claims = []
        return turns, claims

    def friend_upi(r, v):
        name = r.choice(FIRST_NAMES)
        amt, amt_s = r.choice(SMALL_AMOUNTS)
        if v == 0:
            turns = [("C", f"Hey, it's {name}. You owe me for the lunch yesterday."),
                     ("C", f"Can you send {amt_s} rupees on Google Pay when you get time?"),
                     ("R", "Sure, I'll do it tonight.")]
            claims = [("name", name.lower()), ("amount", str(amt)), ("payment_request", "upi")]
        else:
            turns = [("C", "Hi, about the birthday gift we planned."),
                     ("C", f"Your share is {amt_s} rupees, just transfer it by UPI whenever you are free."),
                     ("R", "No problem.")]
            claims = [("amount", str(amt)), ("payment_request", "upi")]
        return turns, claims

    def insurance(r, v):
        amt, amt_s = r.choice(AMOUNTS[:4])
        if v == 0:
            turns = [("C", "Hello sir, I am calling from LIC regarding your policy renewal."),
                     ("C", f"Your premium of {amt_s} rupees is due within thirty days."),
                     ("C", "You can pay through net banking or at any of our branches.")]
            claims = [("name", "lic"), ("amount", str(amt)), ("deadline", "30 day")]
        else:
            turns = [("C", "Good morning, this is a courtesy call from Star Health."),
                     ("C", f"Your renewal premium is {amt_s} rupees and the policy expires next month."),
                     ("R", "Thanks, I'll pay on the website.")]
            claims = [("amount", str(amt))]
        return turns, claims

    return {"bank_reminder": bank_reminder, "doctor": doctor, "family": family,
            "delivery": delivery, "friend_upi": friend_upi, "insurance": insurance}


# ------------------------------------------------------------------- audio


def _synthesize(jobs: list[tuple[str, str, Path]]) -> None:
    """Render (voice, text, wav_path) jobs in one PowerShell process."""
    lines = ["Add-Type -AssemblyName System.Speech",
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer", "$s.Rate = 0"]
    for voice, text, path in jobs:
        safe = text.replace("'", "''")
        lines += [f"$s.SelectVoice('{voice}')", f"$s.SetOutputToWaveFile('{path}')",
                  f"$s.Speak('{safe}')"]
    lines.append("$s.Dispose()")
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8-sig") as f:
        f.write("\n".join(lines))
    try:
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", f.name],
                       check=True, capture_output=True)
    finally:
        Path(f.name).unlink(missing_ok=True)


def _load_16k(path: Path) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        x = np.arange(0, len(data), sr / 16000)
        data = np.interp(x, np.arange(len(data)), data).astype(np.float32)
    return data


def _add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = float(np.mean(x ** 2)) or 1e-9
    noise = rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), x.shape)
    return np.clip(x + noise, -1, 1).astype(np.float32)


def _phone_band(x: np.ndarray) -> np.ndarray:
    """300-3400 Hz band, as if through an 8 kHz phone line (FFT mask + 8 kHz round trip)."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / 16000)
    spec[(freqs < 300) | (freqs > 3400)] = 0
    y = np.fft.irfft(spec, len(x))
    y8 = y[::2]
    return np.clip(np.interp(np.arange(len(x)) / 2, np.arange(len(y8)), y8), -1, 1).astype(np.float32)


def main(out: str, fills: int, seed: int, dev_variants: bool = False) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    calls = []
    for label, fams in (("scam", _scam_families()), ("benign", _benign_families())):
        for fam, fn in fams.items():
            for v, split in ((0, "dev"), (1, "test")):
                for k in range(fills):
                    turns, claims = fn(rng, v)
                    calls.append({"id": f"{split}_{fam}_v{v}_{k}", "split": split, "family": fam,
                                  "label": label, "turns": turns,
                                  "claims": [{"type": t, "key": key} for t, key in claims]})
    work = Path(tempfile.mkdtemp(prefix="echo_tts_"))
    jobs = []
    for c in calls:
        for i, (who, text) in enumerate(c["turns"]):
            jobs.append((CALLER if who == "C" else RECIPIENT, text, work / f"{c['id']}_{i}.wav"))
    _synthesize(jobs)

    labels = []
    for c in calls:
        pause = np.zeros(int(16000 * 0.45), np.float32)
        parts = []
        for i in range(len(c["turns"])):
            parts += [_load_16k(work / f"{c['id']}_{i}.wav"), pause]
        audio = np.concatenate(parts)
        reference = " ".join(t for _, t in c["turns"])
        variants = [("clean", audio)]
        if c["split"] == "test":
            variants += [("snr10", _add_noise(audio, 10, nrng)), ("snr5", _add_noise(audio, 5, nrng)),
                         ("phone", _phone_band(audio))]
        elif dev_variants:
            # Dev-only degraded copies for tuning (e.g. the VAD threshold); added after
            # the first test run, so they are opt-in to keep the default set unchanged.
            variants.append(("phone", _phone_band(audio)))
        for cond, a in variants:
            name = f"{c['id']}_{cond}.wav"
            sf.write(out_dir / name, a, 16000)
            labels.append({"file": name, "id": c["id"], "split": c["split"], "family": c["family"],
                           "label": c["label"], "condition": cond, "reference": reference,
                           "claims": c["claims"]})
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"wrote {len(labels)} recordings ({len(calls)} scripted calls) to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/echo_calls")
    ap.add_argument("--fills", type=int, default=3, help="slot fillings per family and phrasing")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--dev-variants", action="store_true",
                    help="also write phone-band copies of dev calls (for tuning)")
    a = ap.parse_args()
    main(a.out, a.fills, a.seed, a.dev_variants)
