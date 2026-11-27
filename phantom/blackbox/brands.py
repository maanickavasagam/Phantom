"""Brand registry for BLACKBOX lookalike detection.

For each brand: the registrable domains it officially uses (including
CDN/short-link/sister domains, so first-party links are not reported as
fakes), and the name tokens a fake site would borrow.

Only domains we are confident the brand controls are listed (when unsure, a
domain is left out: a missing official domain shows up as a visible false
alarm, a wrong one would let a fake be "verified"). Keep this list reviewed: see docs/BLACKBOX_EVAL.md.

Restricted suffixes are handled separately (see RESTRICTED_SUFFIXES): only
verified entities can register under them, so a matching name there is
treated as official.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Brand:
    key: str
    name: str
    official: frozenset[str]  # registrable domains (e.g. "amazon.in")
    tokens: tuple[str, ...]  # name fragments a lookalike would reuse (lowercase, no spaces)
    core: tuple[str, ...]  # main labels used for typo-distance checks
    restricted: tuple[str, ...] = field(default=())  # e.g. ("bank.in",): label must match a token


_AMAZON_TLDS = ("in", "com", "co.uk", "de", "fr", "it", "es", "ca", "com.au", "co.jp", "ae",
                "sa", "sg", "nl", "se", "pl", "com.br", "com.mx", "com.tr", "eg", "cn", "ie",
                "co.za", "com.be", "dev")
# Amazon/AWS service domains. Added after the v1 false alarms on Tranco ranks
# 1-100,000 (the tuning split); held-out report uses ranks 100,001-200,000.
_AMAZON_SERVICES = (
    "amazonvideo.com", "amazonalexa.com", "amazonaws.com.cn", "amazonaws.cn", "amazonaws.eu",
    "amazonsilk.com", "amazonses.com", "amazoncognito.com", "images-amazon.com",
    "amazon-ads-attestation.com", "amazon-dss.com", "amazonworkspaces.com", "amazoncrl.com",
    "amazonlightsail.com", "jobsatamazon.co.uk", "assoc-amazon.com", "amazonfiretvapp.com",
    "amazonlogistics.eu", "payments-amazon.com", "amazonregistry.com", "aboutamazon.co.uk",
    "siege-amazon.com")
# Platforms where ANYONE can host content under a brand-owned domain: the domain
# proves nothing about who made the page (Stage-A: phishing on s3.amazonaws.com).
HOSTING_PLATFORMS = {
    "amazonaws.com": ("s3", "compute", "elasticbeanstalk", "execute-api", "cloudfront",
                      "awsapps", "amplifyapp"),
    "googleapis.com": ("storage",), "google.com": ("sites", "docs", "forms", "drive"),
}

BRANDS: dict[str, Brand] = {b.key: b for b in (
    Brand("amazon", "Amazon",
          frozenset({f"amazon.{t}" for t in _AMAZON_TLDS} | set(_AMAZON_SERVICES) | {
              "amzn.in", "amzn.to", "amzn.eu", "amzn.com", "a.co", "amazonpay.in", "amazon.jobs",
              "primevideo.com", "amazonaws.com", "media-amazon.com", "ssl-images-amazon.com",
              "amazon-adsystem.com", "amazontrust.com", "aboutamazon.com", "aboutamazon.in",
              "amazon.science", "audible.in", "audible.com", "alexa.com"}),
          ("amazon", "amzn", "amazonpay", "primevideo"), ("amazon", "amzn")),
    Brand("flipkart", "Flipkart",
          frozenset({"flipkart.com", "flixcart.com", "fkrt.it", "flipkartwholesale.com",
                     "shopsy.in"}),
          ("flipkart", "flipkrt", "flixcart", "fkrt"), ("flipkart",)),
    Brand("myntra", "Myntra",
          frozenset({"myntra.com", "myntassets.com"}),
          ("myntra",), ("myntra",)),
    Brand("meesho", "Meesho",
          frozenset({"meesho.com"}),
          ("meesho",), ("meesho",)),
    Brand("paytm", "Paytm",
          frozenset({"paytm.com", "paytm.in", "paytmbank.com", "paytmmall.com", "paytmmoney.com"}),
          ("paytm",), ("paytm",)),
    Brand("phonepe", "PhonePe",
          frozenset({"phonepe.com", "phon.pe"}),
          ("phonepe", "phonpe"), ("phonepe",)),
    Brand("gpay", "Google Pay",
          frozenset({"google.com", "google.co.in", "g.co", "goo.gl", "gstatic.com", "googleapis.com"}),
          ("gpay", "googlepay", "googlpay"), ("googlepay", "gpay")),
    Brand("sbi", "State Bank of India",
          frozenset({"sbi.co.in", "onlinesbi.com", "sbicard.com", "sbilife.co.in", "sbimf.com", "sbigeneral.in"}),
          ("onlinesbi", "yonosbi", "sbiyono", "statebankofindia", "sbionline", "sbi"),
          ("onlinesbi", "statebank"), restricted=("bank.in",)),
    Brand("hdfc", "HDFC Bank",
          frozenset({"hdfcbank.com", "hdfcbank.net", "hdfc.com", "hdfcsec.com",
                     "hdfclife.com", "hdfcergo.com", "hdfcfund.com"}),
          ("hdfcbank", "hdfc"), ("hdfcbank", "hdfc"), restricted=("bank.in",)),
    Brand("icici", "ICICI Bank",
          frozenset({"icicibank.com", "icicidirect.com", "iciciprulife.com",
                     "icicilombard.com", "icicipruamc.com"}),
          ("icicibank", "icici"), ("icicibank", "icici"), restricted=("bank.in",)),
    Brand("irctc", "IRCTC (Indian Railways)",
          frozenset({"irctc.co.in", "irctctourism.com", "indianrail.gov.in"}),
          ("irctc", "indianrail"), ("irctc",), restricted=("gov.in", "nic.in")),
    Brand("indiapost", "India Post",
          frozenset({"indiapost.gov.in", "ippbonline.com"}),
          ("indiapost", "speedpost", "ippb"), ("indiapost",), restricted=("gov.in", "nic.in")),
)}

# Registries where only verified organisations can register names:
#   bank.in  RBI-mandated domain for Indian banks (2025)
#   gov.in / nic.in  Government of India
#   sbi      SBI's own brand gTLD (every *.sbi name belongs to SBI)
RESTRICTED_SUFFIXES = ("bank.in", "gov.in", "nic.in")
BRAND_TLDS = {"sbi": "sbi"}

# Three-letter tokens hide inside ordinary words ("sbi" in "sbitech"), so they
# only count as a separate word. Longer ones (hdfc, icici...) count anywhere.
SHORT_TOKENS = {"sbi"}
# Ordinary words that contain a brand token but mean something else
# (Amazon river / Amazonas region), so they are not brand mentions.
NON_BRAND_WORDS = ("amazonas", "amazonia", "amazona", "amazone", "amazonian")
