"""BLACKBOX behaviour detectors (Day 22).

Explainable rules over an observed PageState. Each returns the behaviour type
with a short reason. They are deliberately specific so the benign decoys on
every site (a normal username/password login, a PDF brochure, a contact phone
number, a newsletter form, "sale ends Sunday" text) are NOT reported; when
they are, it is counted as a false positive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from phantom.blackbox.browser import PageState

_SECRET_FIELD = re.compile(r"\botp\b|one[- ]time|cvv|cvc|card.?number|atm.?pin|\bpin\b|aadhaa?r|"
                           r"upi.?pin|net.?banking|mpin", re.I)
_PAY_TEXT = re.compile(r"\b(?:fee|pay|payment|charges?)\b", re.I)
_PAY_CHANNEL = re.compile(r"\bupi\b|gift ?card|google play|itunes|bitcoin|crypto wallet", re.I)
_PAY_FIELD = re.compile(r"upi|gift|wallet", re.I)
_TIMER = re.compile(r"\b\d{1,2}:\d{2}\b")
_HURRY = re.compile(r"expires?|ends in|left|hurry|last chance|only \d+", re.I)
_EXEC_FILE = re.compile(r"\.(?:apk|exe|msi|bat|scr|dmg)(?:$|\?)", re.I)
_PHONE = re.compile(r"\+?\d[\d\s-]{8,}\d")
_SCARE = re.compile(r"infected|virus|malware|hacked|warning|data loss|call .{0,30}support", re.I)
_PERSONAL_FIELD = re.compile(r"name|phone|mobile|tel|dob|birth|email|address|card|otp", re.I)


@dataclass
class Detection:
    type: str
    page: str  # path where it was observed (redirect source for deceptive_redirect)
    reason: str


def _fields(form) -> str:
    return " ".join(f"{f['name']} {f['type']} {f['label']}" for f in form["fields"])


def detect(state: PageState, origin: str) -> list[Detection]:
    out: list[Detection] = []
    path = state.path

    for f in state.forms:
        fs = _fields(f)
        if _SECRET_FIELD.search(fs):
            out.append(Detection("credential_form", path,
                                 f"form asks for secrets: {_SECRET_FIELD.search(fs).group(0)}"))
        host = urlparse(f["action"])
        if f"{host.scheme}://{host.netloc}" != origin and _PERSONAL_FIELD.search(fs):
            out.append(Detection("exfil_form", path,
                                 f"form sends personal data to another site: {host.netloc}"))
    pay_form = any(_PAY_FIELD.search(_fields(f)) for f in state.forms)
    if _PAY_TEXT.search(state.text) and _PAY_CHANNEL.search(state.text) and pay_form:
        out.append(Detection("payment_demand", path, "asks for a fee via UPI / gift card"))
    if state.countdown or (_TIMER.search(state.text) and _HURRY.search(state.text)):
        out.append(Detection("fake_urgency", path, "countdown timer / artificial scarcity"))
    for lk in state.links:
        if _EXEC_FILE.search(lk["href"]):
            out.append(Detection("forced_download", path,
                                 f"pushes an executable download: {lk['href'].rsplit('/', 1)[-1]}"))
            break
    if state.redirected_from:
        src = urlparse(state.redirected_from).path
        has_login = any(f2["type"] == "password" for f in state.forms for f2 in f["fields"])
        if has_login or re.search(r"verif|secure|login", state.title, re.I):
            out.append(Detection("deceptive_redirect", src,
                                 f"silently redirects to a login page: {state.title}"))
    scare_text = state.overlay or state.text
    if _PHONE.search(scare_text) and _SCARE.search(scare_text):
        out.append(Detection("support_popup", path, "scareware message with a phone number"))
    return out
