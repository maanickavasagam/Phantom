"""ECHO claim extraction (Day 16).

Turns a timestamped transcript into structured claim records:

    {type, value, key, start_s, end_s, segment_ids, supporting_text, attrs}

Types (docs/ECHO_SCOPE.md): amount, deadline, payment_request, name,
credential_request. v1 is rule-based on purpose: every claim can be explained
by the phrase that matched, and the rules are small enough to review. It will
miss indirect phrasing; that is a documented limitation.

`key` is a normalized form used for evaluation and de-duplication, e.g.
amount 50000 INR -> "50000", credential "OTP" -> "otp".
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from phantom.echo.transcribe import Transcript

# ------------------------------------------------------------------ numbers

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "a": 1, "an": 1, "half": 0.5,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000,
           "lac": 100_000, "crore": 10_000_000, "crores": 10_000_000, "million": 1_000_000,
           "k": 1_000}
_NUMWORD = "|".join(sorted([*_UNITS, *_SCALES], key=len, reverse=True))
NUM = (rf"(?:\d[\d,]*(?:\.\d+)?(?:\s*(?:{'|'.join(_SCALES)})\b)?"
       rf"|(?:(?:{_NUMWORD})(?:[\s-]+(?:and\s+)?(?:{_NUMWORD}))*)\b)")


def parse_number(text: str) -> float | None:
    """'50,000' -> 50000; 'fifty thousand' -> 50000; '2 lakh' -> 200000; 'two' -> 2."""
    t = text.lower().replace(",", "").strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]+)?", t)
    if m:
        scale = m.group(2)
        if scale and scale not in _SCALES:
            return None
        return float(m.group(1)) * (_SCALES[scale] if scale else 1)
    total, current = 0.0, 0.0
    for w in re.split(r"[\s-]+", t):
        if w == "and" or not w:
            continue
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _SCALES:
            current = (current or 1) * _SCALES[w]
            if _SCALES[w] >= 1000:
                total += current
                current = 0
        else:
            return None
    value = total + current
    return value if value else None


# ------------------------------------------------------------------- claims


@dataclass
class Claim:
    type: str
    value: str
    key: str
    start_s: float
    end_s: float
    segment_ids: list[int]
    supporting_text: str
    attrs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Span:
    text: str
    start: int
    end: int


class _Index:
    """Full transcript text with a char-offset -> word -> time mapping."""

    def __init__(self, tr: Transcript):
        self.tr = tr
        parts, self.word_spans, pos = [], [], 0
        for seg in tr.segments:
            words = seg.words or []
            if not words:  # no word timing: treat the segment as one word
                from phantom.echo.transcribe import Word
                words = [Word(seg.text, seg.start, seg.end, 1.0)]
            for w in words:
                # Whisper splits "25,000" into "25" + ",000" (and "5" + "." ...): tokens
                # starting with punctuation attach to the previous one, no space.
                if parts and not re.match(r"[,.!?;:%')]", w.text):
                    parts.append(" ")
                    pos += 1
                parts.append(w.text)
                self.word_spans.append((pos, pos + len(w.text), w, seg))
                pos += len(w.text)
        self.text = "".join(parts)

    def locate(self, start: int, end: int):
        hits = [(w, s) for a, b, w, s in self.word_spans if a < end and b > start]
        if not hits:
            return 0.0, 0.0, [], ""
        segs = list(dict.fromkeys(s.id for _, s in hits))
        seg_text = " ".join(s.text for s in self.tr.segments if s.id in segs)
        return hits[0][0].start, hits[-1][0].end, segs, seg_text


_CURRENCY_PRE = r"(?:₹|rs\.?|inr|\$|usd)"
_CURRENCY_POST = r"(?:rupees?|rs\.?|inr|dollars?|bucks|usd)"
_AMOUNT_RES = [
    re.compile(rf"{_CURRENCY_PRE}\s?(?P<n>{NUM})", re.I),
    re.compile(rf"(?P<n>{NUM})\s*{_CURRENCY_POST}\b", re.I),
    re.compile(rf"(?P<n>(?:\d[\d,.]*|{_NUMWORD})(?:[\s-]+(?:{_NUMWORD}))*"
               r"\s+(?:lakhs?|lac|crores?))\b", re.I),
]
_TIME_UNIT = r"(?:minutes?|mins?|hours?|hrs?|days?)"
_DEADLINE_RES = [
    re.compile(rf"\b(?:within|in)\s+(?:the\s+next\s+)?(?P<n>{NUM}|a few|few)"
               rf"\s+(?P<u>{_TIME_UNIT})\b", re.I),
    re.compile(rf"\b(?:only|just)\s+(?P<n>{NUM})\s+(?P<u>{_TIME_UNIT})"
               r"\s+(?:left|remaining)\b", re.I),
    re.compile(r"\b(?:by|before)\s+(?P<w>today|tonight|tomorrow(?:\s+morning)?|end of (?:the )?day|"
               r"\d{1,2}(?::\d{2})?\s?(?:a\.?m\.?|p\.?m\.?|o'?clock))", re.I),
    re.compile(r"\b(?P<w>immediately|right now|right away|at once|as soon as possible|asap|"
               r"today itself|without delay)\b", re.I),
    # "will be disconnected tonight", "is being blocked today"
    re.compile(r"\b(?:will be|is being|are being|gets?)\s+\w+\s+(?P<w>tonight|today|tomorrow)\b",
               re.I),
]
_CHANNELS = {
    "gift card": r"(?:google play|amazon|itunes|apple|steam)?\s*gift\s*cards?",
    "crypto": r"bitcoins?|crypto(?:currency)?|usdt|ethereum",
    "upi": r"upi|google\s?pay|g\s?pay|phone\s?pe|paytm|bhim",
    "wire transfer": r"wire transfer|western union|money\s?gram|swift",
    "bank transfer": r"bank transfer|neft|imps|rtgs|net\s?banking transfer",
    "qr code": r"qr code",
    "remote access app": r"any\s?desk|team\s?viewer|quick\s?support|remote access|"
                         r"remote (?:control|desktop|session)|screen ?shar\w*|"
                         r"access (?:to )?your (?:computer|device|laptop|pc|screen|system)",
}
_UNUSUAL = {"gift card", "crypto", "wire transfer", "qr code", "remote access app"}
_PAY_VERB = (r"\b(?:pay|paying|transfer|transferring|send|sending|deposit|wire|remit|scan|"
             r"install|download|buy|buying|purchase|recharge|invest)\b")
# Secrets: things a legitimate caller never asks for. A request for one is a
# strong indicator on its own.
_SECRETS = {
    "otp": r"\bo\.?t\.?p\b|one[\s-]time\s+(?:password|pin|code)",
    "pin": r"\b(?:upi\s+|atm\s+|m)?pin\b",
    "cvv": r"\bcvv\b|\bcvc\b|three[\s-]digit code",
    "password": r"\bpass\s?words?\b|\bpasscode\b|net\s?banking (?:id|login)",
    "verification code": r"verification code|security code|confirmation code",
    "card number": r"card (?:number|details|information|info)|(?:debit|credit) card number",
    "aadhaar": r"\baadhaa?r\b",
    "ssn": r"social security(?: number)?|\bssn\b",
}
# Identity data: also asked for by genuine services (a bank checking an account,
# a courier confirming an address), so a request is only a weak indicator.
_IDENTITY = {
    "account number": r"account number|bank account (?:details|information|info)|routing number",
    "date of birth": r"date of birth|\bdob\b|birth ?date",
    "personal details": r"personal (?:information|details|info)",
}
_ASK = (r"\b(?:read|tell|share|give|send|provide|confirm|say|enter|type|forward|need|verify|"
        r"update|repeat|what(?:'s| is))\b")
# "...the last four digits of MY card number?" is the recipient checking the
# caller, not being asked for a secret.
_SELF_OR_PARTIAL = re.compile(r"\b(?:my|our)\s+(?:\w+\s+){0,2}$|last (?:four|4) digits", re.I)
_NEGATION = (r"\b(?:never|don'?t|do not|not to|no one|nobody|will not|won'?t|should not|"
             r"shouldn'?t)\b")
_TITLE = (r"(?i:officer|inspector|constable|sub-inspector|mr\.?|mrs\.?|ms\.?|dr\.?|agent|"
          r"manager|executive|advocate)")
# The intro phrase is case-insensitive; the name itself must be Capitalised (as
# Whisper writes proper nouns), so "I am going" or "this is urgent" don't match.
_NAME_RES = [
    re.compile(r"\b(?i:this is|my name is|i am|i'm|it's|it is|speaking(?: here)? is)\s+"
               rf"(?P<v>(?:{_TITLE}\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"),
]
_NOT_NAMES = {"calling", "from", "the", "here", "sorry", "fine", "good", "urgent", "important",
              "regarding", "about", "your", "not", "just", "very", "really", "it", "this"}
_ORG_RE = re.compile(r"\b(?:calling|speaking|call(?:ing)? you|reminder|message|call|notice)\s+"
                     r"(?:is\s+)?(?:from|on behalf of)\s+(?:the\s+)?"
                     r"(?P<v>[A-Za-z][\w&'-]*(?:\s+[A-Za-z][\w&'-]*){0,5})", re.I)  # stops at . , ;
_ORG_STOP = re.compile(r"\s+(?:and|to|about|regarding|because|for|with|sir|madam|ma'am)\b.*$", re.I)


def _window(text: str, start: int, end: int, words_before: int, words_after: int) -> str:
    before = text[:start].split()[-words_before:] if words_before else []
    after = text[end:].split()[:words_after] if words_after else []
    return " ".join(before), " ".join(after)


def extract_claims(tr: Transcript) -> list[Claim]:
    idx = _Index(tr)
    text = idx.text
    claims: list[Claim] = []

    def add(type_, value, key, start, end, **attrs):
        s, e, segs, sup = idx.locate(start, end)
        claims.append(Claim(type_, value.strip(), key, s, e, segs, sup, attrs))

    # amounts
    taken: list[tuple[int, int]] = []
    for rx in _AMOUNT_RES:
        for m in rx.finditer(text):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            n = parse_number(m.group("n").replace(",", ""))
            if n is None or n <= 0:
                continue
            raw = m.group(0)
            currency = "USD" if re.search(r"\$|dollar|usd|bucks", raw, re.I) else "INR"
            taken.append((m.start(), m.end()))
            add("amount", raw, str(int(n)) if float(n).is_integer() else f"{n:.2f}",
                m.start(), m.end(), amount=n, currency=currency)

    # deadlines
    for rx in _DEADLINE_RES:
        for m in rx.finditer(text):
            gd = m.groupdict()
            if gd.get("u"):
                n = parse_number(gd["n"]) if gd["n"] not in ("a few", "few") else None
                u = gd["u"].lower()
                unit = "minute" if u.startswith("min") else "hour" if u.startswith("h") else "day"
                key = f"{int(n) if n else 'few'} {unit}"
            else:
                key = re.sub(r"\s+", " ", gd["w"].lower()).replace(".", "")
                key = re.sub(r"(\d)\s*(am|pm)\b", r"\1 \2", key)  # "5pm" == "5 pm"
            add("deadline", m.group(0), key, m.start(), m.end())

    # payment requests: a payment verb near a channel or an amount
    channel_spans: list[tuple[int, int]] = []
    for chan, pat in _CHANNELS.items():
        for m in re.finditer(pat, text, re.I):
            before, after = _window(text, m.start(), m.end(), 10, 8)
            if not re.search(_PAY_VERB, f"{before} {m.group(0)} {after}", re.I):
                continue
            channel_spans.append((m.start(), m.end()))
            add("payment_request", m.group(0), chan, m.start(), m.end(),
                channel=chan, unusual_channel=chan in _UNUSUAL)
    for m in re.finditer(_PAY_VERB, text, re.I):
        _, after = _window(text, m.start(), m.end(), 0, 8)
        # Same request as a channel claim already made ("pay ... by UPI", "UPI pay")? Skip.
        # Only a channel right around this verb counts, not one a sentence away.
        if any(m.start() - 25 <= a <= m.end() + len(after) + 1 for a, _ in channel_spans):
            continue
        negated = re.search(_NEGATION, _window(text, m.start(), m.end(), 4, 0)[0], re.I)
        if any(rx.search(after) for rx in _AMOUNT_RES) and not negated:
            add("payment_request", m.group(0) + " " + after, "payment", m.start(),
                m.end() + 1 + len(after), channel=None, unusual_channel=False)

    # credential requests: a secret / identity item + an ask nearby, no negation
    # ("never share your OTP"), and not the speaker's own ("my card number")
    for level, table in (("secret", _SECRETS), ("identity", _IDENTITY)):
        for secret, pat in table.items():
            for m in re.finditer(pat, text, re.I):
                before, after = _window(text, m.start(), m.end(), 8, 3)
                if re.search(_NEGATION, before, re.I):
                    continue
                if not re.search(_ASK, f"{before} {after}", re.I):
                    continue
                if _SELF_OR_PARTIAL.search(text[max(0, m.start() - 40):m.start()]) or \
                        _SELF_OR_PARTIAL.search(before):
                    continue
                add("credential_request", m.group(0), secret, m.start(), m.end(),
                    secret=secret, level=level)

    # self-identification and claimed organisation
    for rx in _NAME_RES:
        for m in rx.finditer(text):
            v = m.group("v")
            if v.split()[-1].lower() in _NOT_NAMES or v.split()[0].lower() in _NOT_NAMES:
                continue
            title = v.split()[0] if re.fullmatch(_TITLE, v.split()[0]) else None
            add("name", v, v.split()[-1].lower(), m.start("v"), m.end("v"), kind="person",
                title=title)
    for m in _ORG_RE.finditer(text):
        v = _ORG_STOP.sub("", m.group("v")).rstrip(".,")
        add("name", v, v.lower(), m.start("v"), m.start("v") + len(v), kind="organization")

    # De-duplicate: the same payment channel or the same secret asked for again later
    # in the call is the same request; other claims only merge at the same moment.
    seen, unique = set(), []
    for c in sorted(claims, key=lambda c: (c.start_s, c.type)):
        repeatable = c.type in ("payment_request", "credential_request") and c.key != "payment"
        k = (c.type, c.key) if repeatable else (c.type, c.key, round(c.start_s))
        if k not in seen:
            seen.add(k)
            unique.append(c)
    return unique
