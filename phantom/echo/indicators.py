"""ECHO social-engineering indicators (Day 17).

Explainable indicators, each tied to the transcript passages that triggered it.

Strong indicators (each is enough to mark a call suspicious):
  credential_request       caller asks for an OTP, PIN, CVV, password, card/account number
  unusual_payment_channel  gift cards, crypto, wire/Western Union, QR code, remote-access apps

Weak indicators (common in genuine calls too, so they only count in combination):
  urgency      deadlines / "immediately"
  authority    police, CBI, customs, RBI, court... or a title like "Officer"
  threat       arrest, warrant, legal action, account/SIM blocked, fines
  secrecy      "don't tell anyone", "don't disconnect", "stay on the line"
  lure         prizes, lotteries, "your money will double", refunds you didn't ask for
  identity     asks for account number, date of birth, personal details
  payment      a request to pay or transfer money

Rule: two or more *different* weak indicators, at least one of them a
manipulation tactic (authority, threat, secrecy, lure) -> suspicious "pressure pattern".
A single weak indicator is reported as a note, not a finding: a genuine bill
reminder has a deadline, a genuine bank call names the bank.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from phantom.echo.claims import Claim
from phantom.echo.transcribe import Transcript

_AUTHORITY_ORG = re.compile(
    r"police|cyber ?crime|\bcbi\b|crime branch|customs|income ?tax|\brbi\b|reserve bank|"
    r"\bcourt\b|\btrai\b|telecom (?:regulatory|department)|enforcement directorate|\bed\b|"
    r"narcotics|\bncb\b|government|ministry|\bfbi\b|\birs\b|social security|internal revenue|"
    r"federal trade commission|\bftc\b|medicare|department of (?:justice|treasury|homeland)|"
    r"sheriff", re.I)
_AUTHORITY_TITLE = re.compile(r"officer|inspector|constable|agent|advocate", re.I)
_THREAT = re.compile(
    r"\barrest(?:ed)?\b|\bwarrant\b|legal action|\bjail\b|\bfir\b|police case|court case|"
    r"(?:account|card|sim|number|connection)s?\s+(?:will be|is being|has been|is)\s+"
    r"(?:blocked|suspended|frozen|closed|deactivated|disconnected)|"
    # "fine" only as a penalty: "pay a fine", "fine of", "be fined", not "I'm fine"
    r"(?:pay|paying|a|heavy|huge)\s+(?:the\s+)?fine\b|\bfine of\b|\bfined\b|penalty of|"
    r"money laundering|drug trafficking|illegal (?:parcel|package)|"
    r"(?:computer|device|laptop|system|account)\s+(?:is|has been|was)\s+(?:infected|hacked|"
    r"compromised)|\b(?:virus|malware|spyware)\b", re.I)
_SECRECY = re.compile(
    r"(?:don'?t|do not|never)\s+(?:tell|inform|share this with|discuss)|keep (?:this|it) "
    r"(?:secret|confidential|between us)|(?:don'?t|do not)\s+(?:disconnect|hang up|cut the call)|"
    r"stay on the line|(?:don'?t|do not)\s+(?:go to|visit|call)\s+(?:the )?(?:bank|police)", re.I)
_LURE = re.compile(
    r"\byou(?:'ve| have)? won\b(?!')|\bwinner\b|\blottery\b|lucky draw|\bjackpot\b|\bprize\b|"
    r"sweepstakes|cash ?back offer|(?:will |it will )?double in|guaranteed (?:returns?|profit)|"
    r"\bpercent every day\b|risk[- ]free (?:returns?|investment)|"
    r"eligible for (?:a |an )?refund|refund of|\boverc?harged\b|you(?:'ve| have) been selected",
    re.I)
_URGENCY_WORDS = re.compile(r"\burgent(?:ly)?\b|last chance|final (?:notice|warning)|"
                            r"(?:don'?t|do not) delay", re.I)


@dataclass
class Evidence:
    start_s: float
    end_s: float
    segment_ids: list[int]
    text: str


@dataclass
class Indicator:
    name: str
    strong: bool
    description: str
    evidence: list[Evidence] = field(default_factory=list)


def _ev_from_claim(c: Claim) -> Evidence:
    return Evidence(c.start_s, c.end_s, c.segment_ids, c.supporting_text)


def _scan(tr: Transcript, rx: re.Pattern) -> list[tuple[Evidence, str]]:
    out = []
    for s in tr.segments:
        m = rx.search(s.text)
        if m:
            out.append((Evidence(s.start, s.end, [s.id], s.text), m.group(0)))
    return out


MIN_WEAK_FOR_PATTERN = 2
MANIPULATION = {"authority", "threat", "secrecy", "lure"}
# Evidence that lets the learned classifier's score count towards "suspicious"
# (chosen on our dev calls, see docs/ECHO_EVAL.md section 2b).
CLASSIFIER_SUPPORT = MANIPULATION | {"identity_request"}


@dataclass
class Decision:
    strong: list[Indicator]
    weak: list[Indicator]
    pattern: bool  # weak indicators combine into a suspicious pressure pattern

    @property
    def suspicious(self) -> bool:
        return bool(self.strong) or self.pattern

    @property
    def supports_classifier(self) -> bool:
        """Rule evidence that corroborates a learned-classifier hit."""
        return bool(self.strong) or any(i.name in CLASSIFIER_SUPPORT for i in self.weak)


def decide(indicators: list[Indicator]) -> Decision:
    """The single verdict rule, shared by the live pipeline and the evaluators.
    Deadline + payment alone describes ordinary commerce (a bill reminder), so a
    pattern also needs one manipulation tactic."""
    strong = [i for i in indicators if i.strong]
    weak = [i for i in indicators if not i.strong]
    pattern = (len(weak) >= MIN_WEAK_FOR_PATTERN
               and any(i.name in MANIPULATION for i in weak))
    return Decision(strong, weak, pattern)


def detect_indicators(tr: Transcript, claims: list[Claim]) -> list[Indicator]:
    by = {t: [c for c in claims if c.type == t] for t in
          ("amount", "deadline", "payment_request", "name", "credential_request")}
    found: list[Indicator] = []

    asks = by["credential_request"]
    secret_asks = [c for c in asks if c.attrs.get("level", "secret") == "secret"]
    identity_asks = [c for c in asks if c.attrs.get("level") == "identity"]
    if secret_asks:
        secrets = sorted({c.key for c in secret_asks})
        found.append(Indicator(
            "credential_request", True,
            f"Caller asks for {', '.join(s.upper() if len(s) <= 4 else s for s in secrets)}. "
            "Legitimate banks and agencies never ask for these over the phone.",
            [_ev_from_claim(c) for c in secret_asks]))
    if identity_asks:
        found.append(Indicator(
            "identity_request", False,
            "Asks for identity details (" + ", ".join(sorted({c.key for c in identity_asks}))
            + "); genuine services ask too, so this only counts alongside other tactics.",
            [_ev_from_claim(c) for c in identity_asks]))
    unusual = [c for c in by["payment_request"] if c.attrs.get("unusual_channel")]
    if unusual:
        chans = sorted({c.attrs["channel"] for c in unusual})
        found.append(Indicator(
            "unusual_payment_channel", True,
            f"Payment requested via {', '.join(chans)}: untraceable or remote-access channels "
            "typical of scams.", [_ev_from_claim(c) for c in unusual]))

    urgency = [_ev_from_claim(c) for c in by["deadline"]]
    urgency += [e for e, _ in _scan(tr, _URGENCY_WORDS)]
    if urgency:
        found.append(Indicator("urgency", False, "Time pressure: "
                               + "; ".join(sorted({c.value for c in by["deadline"]}) or {"urgent"}),
                               urgency))
    auth = [c for c in by["name"]
            if (c.attrs.get("kind") == "organization" and _AUTHORITY_ORG.search(c.value))
            or (c.attrs.get("title") and _AUTHORITY_TITLE.search(c.attrs["title"]))]
    if auth:
        found.append(Indicator("authority", False,
                               "Caller claims authority: " + "; ".join(c.value for c in auth),
                               [_ev_from_claim(c) for c in auth]))
    for name, rx, label in (("threat", _THREAT, "Threat or fear appeal"),
                            ("secrecy", _SECRECY, "Isolation / secrecy request"),
                            ("lure", _LURE, "Too-good-to-be-true offer")):
        hits = _scan(tr, rx)
        if hits:
            phrases = "; ".join(sorted({t for _, t in hits}))
            found.append(Indicator(name, False, f"{label}: {phrases}", [e for e, _ in hits]))
    if by["payment_request"]:
        amounts = sorted({c.value for c in by["amount"]})
        found.append(Indicator("payment", False,
                               "Request to pay or transfer money"
                               + (f" ({', '.join(amounts)})" if amounts else ""),
                               [_ev_from_claim(c) for c in by["payment_request"]]))
    return found
