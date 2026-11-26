"""BLACKBOX: is this URL the real brand, or a lookalike? (Week 5)

Given a URL, decide whether it is an official domain of one of the brands in
phantom.blackbox.brands, a likely fake impersonating one, merely suspicious,
or unrelated to any known brand. It NEVER opens the page (scope decision,
26 Nov 2026): it reads the URL itself, plus optional "safe lookups" that only
ask DNS whether the domain exists and the public RDAP registry when it was
registered.

Signals (each with a plain-language reason):
  strong  brand name inside a non-official domain (amazon-offers.in)
          typo-squat of a brand (flipkarrt.com, paytrn.com)
          look-alike characters (amaz0n.com, Cyrillic "а" in аmazon.com)
          brand in the subdomain of someone else's domain (amazon.in.verify.xyz)
          domain registered in the last 30 days (lookups)
  medium  exact brand name on an unofficial ending (flipkart.shop)
          link shortener hiding the destination
  weak    brand only in the path, risky ending (.xyz, .top...), IP address host,
          "@" trick, many subdomains/hyphens, phishing words (kyc, verify, reward),
          no HTTPS, domain under 6 months old (lookups)

Verdicts: official | likely_fake | suspicious | no_brand_link
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from urllib.parse import unquote, urlparse

from phantom.blackbox.brands import (
    BRAND_TLDS,
    BRANDS,
    HOSTING_PLATFORMS,
    NON_BRAND_WORDS,
    RESTRICTED_SUFFIXES,
    SHORT_TOKENS,
    Brand,
)

RISKY_TLDS = {"xyz", "top", "shop", "online", "site", "live", "click", "link", "buzz", "icu", "cfd",
              "sbs", "cyou", "rest", "monster", "quest", "lol", "mom", "bond", "win", "vip",
              "support", "loan", "work", "fit", "gq", "tk", "ml", "cf", "ga", "zip", "mov", "info"}
SHORTENERS = {"bit.ly", "tinyurl.com", "cutt.ly", "is.gd", "t.co", "rb.gy", "shorturl.at",
              "ow.ly", "tiny.cc", "rebrand.ly", "bl.ink", "s.id", "t.ly", "v.gd"}
PHISH_WORDS = re.compile(r"login|log-in|signin|sign-in|verify|verification|update|kyc|secure|"
                         r"account|wallet|reward|gift|offer|free|refund|lucky|winner|bonus|"
                         r"claim|password|otp|unlock|suspend|blocked|helpdesk|customer-?care", re.I)

# Character look-alikes folded to the Latin letter they imitate.
_CONFUSABLE = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g", "@": "a",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l", "ո": "n", "ս": "u", "ν": "v", "ο": "o",
    "α": "a", "ρ": "p", "τ": "t", "κ": "k", "ı": "i", "ł": "l", "ø": "o",
})


@dataclass
class Signal:
    name: str
    strength: str  # strong | medium | weak
    detail: str
    brand: str | None = None


@dataclass
class UrlReport:
    url: str
    host: str
    registrable: str
    verdict: str  # official | likely_fake | suspicious | no_brand_link | invalid
    brand: str | None
    summary: str
    signals: list[Signal] = field(default_factory=list)
    lookups: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ parsing


@lru_cache(maxsize=1)
def _extractor():
    import tldextract

    return tldextract.TLDExtract(suffix_list_urls=())  # bundled list: no network


def split_host(host: str) -> tuple[str, str, str]:
    """(subdomain, label, suffix), aware of restricted Indian suffixes the bundled
    public-suffix list may not know yet (bank.in from 2025)."""
    ex = _extractor()(host)
    sub, label, suffix = ex.subdomain, ex.domain, ex.suffix
    for rs in RESTRICTED_SUFFIXES:
        if f"{label}.{suffix}" == rs and sub:
            parts = sub.split(".")
            return ".".join(parts[:-1]), parts[-1], rs
    return sub, label, suffix


def skeleton(text: str) -> str:
    """Fold look-alike characters to plain Latin letters ("аmaz0n" -> "amazon")."""
    t = unicodedata.normalize("NFKC", text.lower()).translate(_CONFUSABLE)
    t = t.replace("rn", "m").replace("vv", "w").replace("cl", "d")
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


def osa_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein (optimal string alignment): typos incl. swapped letters."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _decode_idn(label: str) -> str:
    try:
        return label.encode("ascii").decode("idna") if "xn--" in label else label
    except (UnicodeError, ValueError):
        return label


# ------------------------------------------------------------------ brand matching


def official_brand(registrable: str, sub: str, label: str, suffix: str) -> Brand | None:
    for b in BRANDS.values():
        if registrable in b.official:
            return b
        if suffix in b.restricted and any(t in label for t in b.tokens):
            return b  # verified-only registry (bank.in, gov.in)
    if suffix in BRAND_TLDS:
        return BRANDS[BRAND_TLDS[suffix]]  # e.g. every *.sbi belongs to SBI
    return None


def _token_in(token: str, text: str) -> bool:
    if token in SHORT_TOKENS:  # short tokens only as a separate word: "sbi-kyc", not "sbitech"
        return re.search(rf"(?:^|[^a-z]){re.escape(token)}(?:$|[^a-z])", text) is not None
    return token in text


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _typo_inside(cands: list[str], core: str) -> str | None:
    """A one-edit typo of `core` embedded in a longer word. Cheap filters first
    (same first letter, most letter pairs shared) before the edit distance."""
    need = _bigrams(core)
    for p in cands:
        if len(p) <= len(core) or len(_bigrams(p) & need) < len(need) - 2:
            continue
        for n in (len(core) - 1, len(core), len(core) + 1):
            if n < 5 or n >= len(p):
                continue
            for i in range(len(p) - n + 1):
                sub = p[i:i + n]
                if sub[0] == core[0] and 0 < osa_distance(sub, core) <= 1:
                    return sub
    return None


@lru_cache(maxsize=50000)
def _mentions_cached(text: str) -> tuple:
    return tuple(_brand_mentions(text))


def brand_mentions(text: str) -> list[tuple[Brand, str, str]]:
    return list(_mentions_cached(text))


def _brand_mentions(text: str) -> list[tuple[Brand, str, str]]:
    """Brands whose name, typo or look-alike appears in a host label / text.
    Returns (brand, kind, matched) with kind in {token, lookalike, typo}."""
    raw = text.lower()
    for w in NON_BRAND_WORDS:  # "amazonia" is a rainforest, not the shop
        raw = raw.replace(w, "#")
    folded = skeleton(_decode_idn(raw))
    parts = [p for p in re.split(r"[.\-_]", folded) if p]
    joined = re.sub(r"[.\-_]", "", folded)  # "am-azon", "amaz.on" -> "amazon"
    hits: list[tuple[Brand, str, str]] = []
    for b in BRANDS.values():
        tok = next((t for t in b.tokens if _token_in(t, raw)), None)
        if tok:
            hits.append((b, "token", tok))
            continue
        tok = next((t for t in b.tokens if _token_in(t, folded)), None)
        if tok:
            hits.append((b, "lookalike", tok))
            continue
        tok = next((t for t in b.tokens if len(t) >= 5 and t in joined and t not in folded),
                   None)
        if tok:  # brand split by hyphens/dots: "am-azon.com", "amaz.on.in"
            hits.append((b, "lookalike", tok))
            continue
        for core in b.core:
            if len(core) < 5:
                continue  # too short for safe typo matching
            # One edit only: 2-edit matches hit real sites (mediapost~indiapost,
            # citibank~icicibank, onlinepbx~onlinesbi) on the tuning split. Names
            # written with non-Latin letters may be 2 edits away ("ɑnnazon"):
            # ordinary sites almost never mix scripts like that.
            idn = not _decode_idn(raw).isascii()
            limit = 2 if idn else 1
            cand = parts + [joined]
            match = next((p for p in cand if 0 < osa_distance(p, core) <= limit and len(p) >= 4),
                         None)
            # Typo hidden inside a longer word ("flipkarrtsale") only for distinctive
            # names of >= 8 letters: shorter ones (paytm, phonepe) and "online..."
            # names match ordinary words ("teacherspayteachers", "phonepower",
            # "onlinespiel") far too easily.
            if match is None and len(core) >= 8 and not core.startswith("online"):
                match = _typo_inside(cand, core)
            if match:
                hits.append((b, "lookalike" if idn else "typo", match))
                break
    return hits


# ------------------------------------------------------------------ main check


def check_url(url: str, lookups: bool = False, now: datetime | None = None) -> UrlReport:
    raw = url.strip()
    if not re.match(r"^[a-z][a-z0-9+.-]*://", raw, re.I):
        raw = "http://" + raw
    p = urlparse(raw)
    host = (p.hostname or "").rstrip(".")
    if not host or " " in host:
        return UrlReport(url, host, "", "invalid", None, "Not a valid web address.")
    signals: list[Signal] = []

    def add(name, strength, detail, brand=None):
        signals.append(Signal(name, strength, detail, brand))

    # host-level red flags
    is_ip = False
    try:
        ipaddress.ip_address(host)
        is_ip = True
        add("ip_address_host", "weak", f"uses a bare IP address ({host}) instead of a name")
    except ValueError:
        pass
    sub, label, suffix = ("", host, "") if is_ip else split_host(host)
    registrable = host if is_ip else (f"{label}.{suffix}" if suffix else label)
    off = None if is_ip else official_brand(registrable, sub, label, suffix)
    userinfo = p.netloc.rsplit("@", 1)[0] if "@" in p.netloc else ""
    if userinfo:
        named = brand_mentions(unquote(userinfo))
        if named:
            b = named[0][0]
            add("brand_in_userinfo", "strong",
                f"'{unquote(userinfo)}@' makes the link read like {b.name}, but browsers ignore "
                f"everything before '@': the real site is {host}", b.key)
        else:
            add("userinfo_trick", "weak", f"text before '@' is ignored by browsers: the real site "
                                          f"is {host}")
    if p.port and p.port not in (80, 443):
        add("unusual_port", "weak", f"unusual port {p.port}")
    if p.scheme == "http":
        add("no_https", "weak", "not encrypted (http)")

    platform = HOSTING_PLATFORMS.get(registrable)
    if off and platform and any(re.search(rf"(?:^|[.-]){k}(?:$|[.-])", sub) for k in platform):
        # s3.amazonaws.com, sites.google.com...: brand-owned domain, user-made content.
        add("hosted_on_brand_platform", "medium",
            f"{host} is a hosting service on {off.name}'s infrastructure: anyone can publish "
            f"pages there, so the domain does not prove the content is from {off.name}")
        off = None
    if off:
        report = _finish(url, host, registrable, signals, off, {})
        return report

    # brand impersonation signals
    label_hits = [] if is_ip else brand_mentions(label)
    for b, kind, m in label_hits:
        if kind == "token" and label == m and suffix not in ("",):
            add("brand_on_unofficial_ending", "medium",
                f"'{label}' is {b.name}'s name, but {registrable} is not one of its official "
                f"domains", b.key)
        elif kind == "token":
            add("brand_in_domain", "strong",
                f"the domain {registrable} contains '{m}' ({b.name}) but is not an official "
                f"{b.name} domain", b.key)
        elif kind == "lookalike":
            decoded = _decode_idn(label)
            what = "non-Latin look-alike letters" if decoded != label or not label.isascii() \
                else "look-alike characters (e.g. 0 for o, rn for m)"
            add("homoglyph", "strong", f"'{decoded}' imitates '{m}' ({b.name}) using {what}",
                b.key)
        else:
            add("typosquat", "strong",
                f"'{m}' in {registrable} is one typo away from {b.name} ('{b.core[0]}')", b.key)
    labelled = {b.key for b, _, _ in label_hits} | {s.brand for s in signals if s.brand}
    if sub and not label_hits:
        last = sub.rsplit(".", 1)[-1]
        split_hits = [h for h in brand_mentions(f"{last}.{label}")
                      if h[0].key not in labelled and not brand_mentions(last)]
        for b, _kind, m in split_hits[:1]:  # "amaz.on.in": brand cut across the dot
            add("brand_split_across_dots", "strong",
                f"'{last}.{label}' spells '{m}' ({b.name}) with a dot inside it; the real "
                f"site is {registrable}", b.key)
            labelled.add(b.key)
    for b, _kind, _m in brand_mentions(sub) if sub else []:
        if b.key not in labelled:
            add("brand_in_subdomain", "strong",
                f"'{sub}' makes it look like {b.name}, but the site actually belongs to "
                f"{registrable}", b.key)
    path = unquote(p.path + "?" + p.query).lower()
    for b, _kind, m in brand_mentions(path) if len(path) > 2 else []:
        if b.key not in labelled and not any(s.brand == b.key for s in signals):
            add("brand_in_path", "weak", f"mentions {b.name} ('{m}') in the link path, on an "
                                         f"unrelated domain {registrable}", b.key)

    # generic red flags
    if "xn--" in host:
        add("punycode", "weak", f"internationalised name (shown as '{_decode_idn(host)}')")
    if registrable in SHORTENERS:
        add("link_shortener", "medium", f"{registrable} is a link shortener: the real "
                                        f"destination is hidden")
    if suffix and suffix.split(".")[-1] in RISKY_TLDS:
        add("risky_tld", "weak", f".{suffix} is an ending heavily used for throwaway/scam sites")
    if sub.count(".") >= 3:
        add("many_subdomains", "weak", f"{sub.count('.') + 1} levels of subdomains")
    if label.replace("xn--", "").count("-") >= 2:  # punycode's own "--" doesn't count
        add("many_hyphens", "weak", f"hyphen-stuffed name '{label}'")
    words = sorted({w.lower() for w in PHISH_WORDS.findall(host + path)})
    if words:
        add("phishing_words", "weak", f"phishing-style words: {', '.join(words[:5])}")
    if len(url) > 120:
        add("long_url", "weak", f"very long link ({len(url)} characters)")

    info = safe_lookups(registrable, now) if lookups and not is_ip else {}
    if info.get("age_days") is not None:
        brandish = any(s.brand for s in signals)
        if info["age_days"] < 30:
            add("new_domain", "strong" if brandish else "medium",
                f"registered only {info['age_days']} days ago ({info['registered']})")
        elif info["age_days"] < 180:
            add("young_domain", "weak", f"registered {info['age_days']} days ago")
    if info.get("resolves") is False:
        add("does_not_resolve", "weak", "the domain does not currently exist in DNS "
                                        "(possibly taken down)")
    return _finish(url, host, registrable, signals, None, info)


def _finish(url, host, registrable, signals, official: Brand | None, info) -> UrlReport:
    if official:
        warn = "" if not signals else " (note: " + "; ".join(s.detail for s in signals) + ")"
        return UrlReport(url, host, registrable, "official", official.key,
                         f"Official {official.name} domain ({registrable}){warn}.", signals, info)
    strong = [s for s in signals if s.strength == "strong" and s.brand]
    brands = [s.brand for s in signals if s.brand]
    brand = max(set(brands), key=brands.count) if brands else None
    if strong:
        name = BRANDS[strong[0].brand].name
        return UrlReport(url, host, registrable, "likely_fake", strong[0].brand,
                         f"Likely FAKE: imitates {name} but is not an official {name} domain.",
                         signals, info)
    medium = [s for s in signals if s.strength in ("medium", "strong")]
    weak = [s for s in signals if s.strength == "weak"]
    if medium or (brand and weak) or len(weak) >= 3:
        who = f" (mentions {BRANDS[brand].name})" if brand else ""
        return UrlReport(url, host, registrable, "suspicious", brand,
                         f"Suspicious{who}: {len(signals)} warning sign(s); not verified as "
                         f"official.", signals, info)
    return UrlReport(url, host, registrable, "no_brand_link", None,
                     "Not linked to any brand we track, and no lookalike signs. This is not a "
                     "guarantee the site is safe.", signals, info)


# ------------------------------------------------------------------ safe lookups


def safe_lookups(registrable: str, now: datetime | None = None, timeout: float = 5.0) -> dict:
    """DNS existence + RDAP registration date. Never contacts the site itself.
    Privacy: sends the domain name to the system DNS resolver and rdap.org."""
    import socket
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    out: dict = {"domain": registrable}
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(socket.getaddrinfo, registrable, None)
        try:
            out["resolves"] = bool(fut.result(timeout=timeout))
        except Exception:  # noqa: BLE001 - NXDOMAIN, timeout, no network
            out["resolves"] = False
    try:
        # Use the operating system's certificate store (Windows/macOS), so HTTPS works
        # behind antivirus or corporate TLS inspection; fall back to certifi.
        try:
            import ssl

            import truststore

            verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except ImportError:
            verify = True
        r = httpx.get(f"https://rdap.org/domain/{registrable}", timeout=timeout, verify=verify,
                      follow_redirects=True, headers={"Accept": "application/rdap+json"})
        if r.status_code == 200:
            events = {e.get("eventAction"): e.get("eventDate") for e in r.json().get("events", [])}
            reg = events.get("registration")
            if reg:
                when = datetime.fromisoformat(reg.replace("Z", "+00:00"))
                out["registered"] = when.date().isoformat()
                out["age_days"] = ((now or datetime.now(UTC)) - when).days
            out["rdap_source"] = str(r.url)
        else:
            out["rdap_error"] = f"HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001 - offline, blocked, malformed
        out["rdap_error"] = type(e).__name__
    return out


# ------------------------------------------------------------------ explanation (UI)

MAIN_DOMAIN = {"amazon": "amazon.in", "flipkart": "flipkart.com", "myntra": "myntra.com",
               "meesho": "meesho.com", "paytm": "paytm.com", "phonepe": "phonepe.com",
               "gpay": "pay.google.com", "sbi": "onlinesbi.sbi", "hdfc": "hdfcbank.com",
               "icici": "icicibank.com", "irctc": "irctc.co.in", "indiapost": "indiapost.gov.in"}

# Every test the checker runs, in the order shown to the user, with the signal
# names that mean "this test found a problem".
CHECKS = [
    ("official", "Is it one of the brand's official domains?", ()),
    ("brand_name", "Does the domain contain a brand's name?",
     ("brand_in_domain", "brand_on_unofficial_ending")),
    ("typo", "Is the name one typo away from a brand (flipkarrt, amazn)?", ("typosquat",)),
    ("lookalike", "Look-alike letters (0 for o, rn for m, Cyrillic a)?",
     ("homoglyph", "punycode")),
    ("split", "Brand name cut by a dot or hyphen (amaz.on, am-azon)?",
     ("brand_split_across_dots",)),
    ("subdomain", "Brand name placed in front, on someone else's domain?",
     ("brand_in_subdomain",)),
    ("userinfo", "Fake address before an '@' (browsers ignore it)?",
     ("brand_in_userinfo", "userinfo_trick")),
    ("hosting", "User-made page on a brand's hosting service?", ("hosted_on_brand_platform",)),
    ("shortener", "Link shortener hiding the real destination?", ("link_shortener",)),
    ("ending", "Risky ending (.xyz, .top, .shop...)?", ("risky_tld",)),
    ("words", "Phishing words (login, kyc, verify, reward...)?", ("phishing_words",)),
    ("https", "Encrypted connection (https)?", ("no_https",)),
    ("ip_port", "Bare IP address or unusual port?", ("ip_address_host", "unusual_port")),
    ("shape", "Odd shape (many subdomains, many hyphens, very long)?",
     ("many_subdomains", "many_hyphens", "long_url")),
    ("path", "Brand name only in the path, on an unrelated domain?", ("brand_in_path",)),
    ("age", "Newly registered or missing domain (safe lookups)?",
     ("new_domain", "young_domain", "does_not_resolve")),
]


def explain(report: UrlReport, url: str) -> dict:
    """Data for the 'how it was judged' view: address parts, the closest real
    domain with a letter-by-letter diff, and the checklist of tests."""
    import difflib

    raw = url.strip()
    if not re.match(r"^[a-z][a-z0-9+.-]*://", raw, re.I):
        raw = "http://" + raw
    p = urlparse(raw)
    host = report.host
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False
    sub, label, suffix = ("", host, "") if is_ip else split_host(host)
    parts = {"scheme": p.scheme, "userinfo": p.netloc.rsplit("@", 1)[0] if "@" in p.netloc
             else "", "subdomain": sub, "domain": label, "suffix": suffix,
             "port": str(p.port or ""), "path": (p.path or "") + (f"?{p.query}" if p.query
                                                                   else ""),
             "decoded_domain": _decode_idn(report.registrable)}
    names = {s.name for s in report.signals}
    checks = []
    for key, question, sigs in CHECKS:
        if key == "official":
            state = "pass" if report.verdict == "official" else "fail" if report.brand else "n/a"
        elif key == "age" and not report.lookups:
            state = "not_run"
        else:
            state = "fail" if names & set(sigs) else "pass"
        checks.append({"key": key, "question": question, "state": state,
                       "signals": sorted(names & set(sigs))})
    comparison = None
    if report.brand and report.verdict != "official":
        fake = _decode_idn(report.registrable)
        if "brand_split_across_dots" in names:  # amaz.on.in: the imitation spans the dot
            fake = _decode_idn(host.removeprefix("www."))
        real = MAIN_DOMAIN.get(report.brand) or sorted(BRANDS[report.brand].official)[0]
        if names & {"typosquat", "homoglyph", "brand_split_across_dots"}:
            # letter-level imitation: line it up against the nearest real domain
            near = min(BRANDS[report.brand].official,
                       key=lambda d: (osa_distance(fake, d), len(d)))
            if osa_distance(fake, near) <= 3:
                real = near
        ops = [{"op": tag, "fake": fake[i1:i2], "real": real[j1:j2]}
               for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                   None, fake, real, autojunk=False).get_opcodes()]
        comparison = {"fake": fake, "real": real, "ops": ops,
                      "letter_level": bool(names & {"typosquat", "homoglyph",
                                                    "brand_split_across_dots"}),
                      "non_latin": sorted({c for c in fake if not c.isascii()})}
    return {"parts": parts, "checks": checks, "comparison": comparison}
