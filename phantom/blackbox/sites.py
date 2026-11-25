"""BLACKBOX synthetic websites with planted behaviours (Day 20).

Generates small static websites (15-30 pages) with HARMLESS planted scam-like
behaviours and documented ground truth, plus benign decoys that look similar
(a normal login page, a PDF brochure, a contact phone number) so false
positives can be measured. Sites are served only on 127.0.0.1.

Train and test sites use DIFFERENT themes (vocabulary, page names, button
texts), so a learned explorer is evaluated on websites it has never seen the
like of, not on re-shuffled training sites.

    python -m phantom.blackbox.sites --out data/blackbox_sites --n-train 40 --n-test 24
"""

from __future__ import annotations

import argparse
import html
import json
import random
import threading
from dataclasses import asdict, dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BEHAVIOURS = ("credential_form", "payment_demand", "fake_urgency", "forced_download",
              "deceptive_redirect", "exfil_form", "support_popup")

THEMES = {
    # ---- train themes
    "shop": {"brand": ["MegaMart", "ShopKart", "DealZone"],
             "nav": ["Products", "Offers", "Orders", "Help"],
             "topics": ["Summer sale", "New arrivals", "Kitchen", "Electronics", "Fashion", "Toys"],
             "cta": ["Claim your voucher", "Unlock member price", "Get free gift", "Confirm order"]},
    "bank": {"brand": ["SecureBank", "TrustFin", "AxisPlus"],
             "nav": ["Accounts", "Cards", "Loans", "Support"],
             "topics": ["Savings", "Fixed deposits", "Home loans", "Credit cards", "Insurance", "NRI"],
             "cta": ["Update KYC now", "Verify your account", "Reactivate card", "Unblock netbanking"]},
    "lottery": {"brand": ["LuckyWin", "MegaDraw", "PrizeHub"],
                "nav": ["Results", "Winners", "Play", "FAQ"],
                "topics": ["Weekly draw", "Jackpot", "Scratch cards", "Past winners", "Rules", "Odds"],
                "cta": ["Claim your prize", "Collect winnings", "Check your ticket", "Release reward"]},
    "courier": {"brand": ["FastShip", "ParcelGo", "SwiftPost"],
                "nav": ["Track", "Ship", "Rates", "Contact"],
                "topics": ["Domestic", "International", "Express", "Customs", "Packaging", "Returns"],
                "cta": ["Reschedule delivery", "Pay customs duty", "Release parcel", "Confirm address"]},
    "jobs": {"brand": ["HireNow", "JobBridge", "WorkLink"],
             "nav": ["Jobs", "Companies", "Resume", "Advice"],
             "topics": ["Remote jobs", "Freshers", "IT jobs", "Part time", "Interview tips", "Salaries"],
             "cta": ["Accept offer letter", "Book interview slot", "Complete onboarding",
                     "Activate profile"]},
    # ---- test themes (never seen in training)
    "govt_benefits": {"brand": ["JanSeva Portal", "Citizen Aid", "Gov Relief"],
                      "nav": ["Schemes", "Eligibility", "Status", "Grievance"],
                      "topics": ["Pension", "Farmer subsidy", "Scholarship", "Ration card", "Housing",
                                 "Gas subsidy"],
                      "cta": ["Receive subsidy", "Link beneficiary ID", "Approve DBT transfer",
                              "Renew scheme enrolment"]},
    "crypto": {"brand": ["CoinVault", "BitNest", "ChainRise"],
               "nav": ["Markets", "Earn", "Wallet", "Academy"],
               "topics": ["Bitcoin", "Staking", "NFTs", "DeFi", "Airdrops", "Trading bots"],
               "cta": ["Withdraw your profit", "Boost staking yield", "Join airdrop",
                       "Upgrade to VIP tier"]},
    "tech_support": {"brand": ["PCFixPro", "TechAssist 24x7", "DeviceCare"],
                     "nav": ["Services", "Diagnostics", "Plans", "Contact"],
                     "topics": ["Virus removal", "Speed up PC", "Printer setup", "Data recovery",
                                "Router help", "Windows errors"],
                     "cta": ["Run free scan", "Fix errors now", "Extend protection",
                             "Connect a technician"]},
}
TRAIN_THEMES = ("shop", "bank", "lottery", "courier", "jobs")
TEST_THEMES = ("govt_benefits", "crypto", "tech_support")

_FILLER = ("We are committed to serving our customers with the best experience. "
           "Read our latest updates and explore everything we have to offer. ")


@dataclass
class Link:
    text: str
    href: str
    region: str  # nav | body | footer | button
    download: bool = False


@dataclass
class Form:
    action: str
    fields: list[tuple[str, str, str]]  # (name, type, label)
    submit: str


@dataclass
class Page:
    path: str
    title: str
    paragraphs: list[str] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    forms: list[Form] = field(default_factory=list)
    extra_html: str = ""
    meta_refresh: str | None = None


@dataclass
class Planted:
    type: str
    page: str  # path where the behaviour becomes observable
    entry_page: str  # page holding the first link of the chain
    entry_text: str
    depth: int


@dataclass
class Site:
    id: str
    theme: str
    split: str
    pages: dict[str, Page]
    planted: list[Planted]
    decoys: list[str]

    def edges(self) -> dict[str, list[str]]:
        """Directed page graph (links, form actions, redirects) for coverage and labels."""
        out = {}
        for p in self.pages.values():
            targets = [lk.href for lk in p.links if lk.href in self.pages]
            targets += [f.action for f in p.forms if f.action in self.pages]
            if p.meta_refresh and p.meta_refresh in self.pages:
                targets.append(p.meta_refresh)
            out[p.path] = targets
        return out

    def ground_truth(self) -> dict:
        return {"id": self.id, "theme": self.theme, "split": self.split,
                "n_pages": len(self.pages), "planted": [asdict(b) for b in self.planted],
                "decoys": self.decoys, "edges": self.edges()}


# ------------------------------------------------------------------ builder


class _Builder:
    def __init__(self, rng: random.Random, site_id: str, theme: str, split: str):
        self.rng, self.t = rng, THEMES[theme]
        self.site = Site(site_id, theme, split, {}, [], [])
        self.brand = rng.choice(self.t["brand"])
        self.n = 0

    def add(self, title: str, slug: str | None = None, paragraphs: int = 2) -> Page:
        # Neutral file names: the URL must not reveal what a page is (a learned
        # explorer could otherwise "find" behaviours by reading "verify12.html").
        self.n += 1
        path = "index.html" if slug == "index" else f"p{self.n}.html"
        p = Page(path, title, [_FILLER * self.rng.randint(1, 2) for _ in range(paragraphs)])
        self.site.pages[path] = p
        return p

    def link(self, src: Page, dst: Page, text: str, region: str) -> None:
        src.links.append(Link(text, dst.path, region))

    def build(self) -> Site:
        r, t = self.rng, self.t
        home = self.add(f"{self.brand} - Home", "index")
        sections = [self.add(f"{name} | {self.brand}", "section") for name in t["nav"]]
        for s, name in zip(sections, t["nav"], strict=True):
            self.link(home, s, name, "nav")
        about, terms, privacy, careers = (self.add(n, "info") for n in
                                          ("About us", "Terms of use", "Privacy policy", "Careers"))
        for pg, text in ((about, "About us"), (terms, "Terms"), (privacy, "Privacy"),
                         (careers, "Careers")):
            self.link(home, pg, text, "footer")
        topics = r.sample(t["topics"], 4)
        content = []
        for i, topic in enumerate(topics):
            sec = sections[i % len(sections)]
            pg = self.add(f"{topic} | {self.brand}", "topic")
            self.link(sec, pg, topic, "body")
            if r.random() < 0.6:
                self.link(home, pg, f"Explore {topic.lower()}", "body")
            content.append(pg)
            if r.random() < 0.6:  # article chains add depth and dead ends
                art = self.add(f"{topic}: guide", "article", 3)
                self.link(pg, art, "Read the full guide", "body")
                content.append(art)
        for pg in self.site.pages.values():  # every page links home
            if pg.path != "index.html":
                pg.links.append(Link("Home", "index.html", "nav"))
        self._decoys(home, sections, about)
        candidates = {0: [home], 1: sections + content[:2], 2: content}
        for kind in r.sample(BEHAVIOURS, r.randint(3, 4)):
            depth = r.choice([1, 1, 2, 2, 3])
            anchor = r.choice(candidates[min(depth - 1, 2)])
            self._plant(kind, anchor, depth)
        for pg in self.site.pages.values():
            r.shuffle(pg.links)
        return self.site

    # ---- benign look-alikes (should NOT be reported)
    def _decoys(self, home: Page, sections: list[Page], about: Page) -> None:
        r = self.rng
        login = self.add(f"Sign in | {self.brand}", "login")
        login.forms.append(Form("index.html", [("username", "text", "Username"),
                                               ("password", "password", "Password")], "Sign in"))
        self.link(home, login, "Sign in", "nav")
        self.site.decoys.append("plain_login_form")
        if r.random() < 0.7:
            r.choice(sections).links.append(Link("Download brochure (PDF)", "files/brochure.pdf",
                                                 "body", download=True))
            self.site.decoys.append("pdf_download")
        contact = self.add(f"Contact | {self.brand}", "contact")
        contact.paragraphs.append("Customer care: +91 80 4567 8900 (Mon-Sat, 9am-6pm).")
        self.link(home, contact, "Contact us", "footer")
        self.site.decoys.append("contact_phone_number")
        news = r.choice(sections)
        news.forms.append(Form("index.html", [("email", "email", "Email address")], "Subscribe"))
        self.site.decoys.append("newsletter_form")
        if r.random() < 0.5:
            r.choice(sections).paragraphs.append("Season sale ends this Sunday. Visit our stores.")
            self.site.decoys.append("sale_text_without_timer")
        about.links.append(Link("Our partners (external)", "https://www.example.org/", "body"))
        self.site.decoys.append("external_link")

    # ---- planted behaviours, `depth` clicks/submits away from `anchor`'s depth
    def _plant(self, kind: str, anchor: Page, depth: int) -> None:
        r = self.rng
        cta = r.choice(self.t["cta"])
        region = r.choice(["body", "body", "button", "footer"])
        chain = anchor
        for _ in range(max(0, depth - 1)):  # intermediate pages before the behaviour
            mid = self.add(f"{cta} | {self.brand}", "step")
            mid.paragraphs.insert(0, f"Continue to {cta.lower()}. This takes less than a minute.")
            self.link(chain, mid, cta if chain is anchor else "Continue", region if chain is anchor
                      else "button")
            chain = mid
        target = self.add(f"{cta} | {self.brand}", kind.split("_")[0])
        entry_text = cta if chain is anchor else "Continue"
        if kind == "credential_form":
            # Revealed only after submitting a harmless-looking form.
            start = target
            start.forms.append(Form("", [("full_name", "text", "Full name"),
                                         ("mobile", "tel", "Mobile number")], "Proceed"))
            verify = self.add(f"Verification | {self.brand}", "verify")
            verify.paragraphs.insert(0, "Enter the details below to complete verification.")
            verify.forms.append(Form("index.html", [
                ("otp", "text", "One-time password (OTP)"), ("card_number", "text", "Card number"),
                ("cvv", "password", "CVV"), ("atm_pin", "password", "ATM PIN")], "Verify"))
            start.forms[0].action = verify.path
            observable = verify.path
        elif kind == "payment_demand":
            target.paragraphs.insert(0, "To release your benefit, pay a one-time processing fee "
                                        "of Rs 499 via UPI or a Google Play gift card.")
            target.forms.append(Form("index.html", [("upi_id", "text", "UPI ID"),
                                                    ("gift_card_code", "text", "Gift card code")],
                                     "Pay fee"))
            observable = target.path
        elif kind == "fake_urgency":
            target.extra_html = ('<div class="countdown" data-deadline="300">Offer expires in '
                                 '<b>04:59</b>. Only 2 slots left!</div>')
            observable = target.path
        elif kind == "forced_download":
            target.paragraphs.insert(0, "Your app is out of date. Install the security update.")
            target.links.append(Link("Install security update", "files/secure-update.apk",
                                     "button", download=True))
            observable = target.path
        elif kind == "deceptive_redirect":
            landing = self.add(f"{self.brand} Secure Login Verification", "secure")
            landing.paragraphs.insert(0, "Session expired. Sign in again to continue.")
            landing.forms.append(Form("index.html", [("user", "text", "User ID"),
                                                     ("pass", "password", "Password")], "Login"))
            target.meta_refresh = landing.path
            target.title = "Redirecting..."
            observable = target.path
        elif kind == "exfil_form":
            target.paragraphs.insert(0, "Help us improve. Share your details to get a reward.")
            target.forms.append(Form("http://collector.track-data.test/submit",
                                     [("name", "text", "Name"), ("phone", "tel", "Phone"),
                                      ("dob", "date", "Date of birth")], "Submit"))
            observable = target.path
        else:  # support_popup
            target.extra_html = ('<div class="overlay alert">WARNING: Your device is infected '
                                 'with 3 viruses! Call Microsoft certified support now on '
                                 '+91 98765 43210 to avoid data loss.</div>')
            observable = target.path
        self.link(chain, target, entry_text, region if chain is anchor else "button")
        self.site.planted.append(Planted(kind, observable, anchor.path, cta, depth))


# ------------------------------------------------------------------ render


def render(page: Page) -> str:
    e = html.escape
    head = f"<meta charset='utf-8'><title>{e(page.title)}</title>"
    if page.meta_refresh:
        head += f"<meta http-equiv='refresh' content='0; url={e(page.meta_refresh)}'>"
    nav = " ".join(f"<a href='{e(lk.href)}'>{e(lk.text)}</a>" for lk in page.links
                   if lk.region == "nav")
    foot = " ".join(f"<a href='{e(lk.href)}'>{e(lk.text)}</a>" for lk in page.links
                    if lk.region == "footer")
    def _attrs(lk: Link) -> str:  # no backslashes inside f-strings: CI runs Python 3.11
        return (" download" if lk.download else "") + (' class="btn"' if lk.region == "button"
                                                       else "")

    body_links = "".join(
        f"<p><a href='{e(lk.href)}'{_attrs(lk)}>{e(lk.text)}</a></p>"
        for lk in page.links if lk.region in ("body", "button"))
    forms = ""
    for f in page.forms:
        inputs = "".join(f"<label>{e(lbl)} <input name='{e(n)}' type='{e(t)}'></label><br>"
                         for n, t, lbl in f.fields)
        forms += (f"<form method='get' action='{e(f.action)}'>{inputs}"
                  f"<button type='submit'>{e(f.submit)}</button></form>")
    paras = "".join(f"<p>{e(p)}</p>" for p in page.paragraphs)
    return (f"<!doctype html><html><head>{head}</head><body><nav>{nav}</nav>"
            f"<main><h1>{e(page.title)}</h1>{paras}{page.extra_html}{body_links}{forms}</main>"
            f"<footer>{foot}</footer></body></html>")


def generate(out: Path, n_train: int, n_test: int, seed: int = 20) -> list[dict]:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    truths = []
    for split, n, themes in (("train", n_train, TRAIN_THEMES), ("test", n_test, TEST_THEMES)):
        for i in range(n):
            theme = themes[i % len(themes)]
            site = _Builder(rng, f"{split}_{i:03d}_{theme}", theme, split).build()
            d = out / "www" / site.id  # only www/ is served; ground truth lives outside it
            (d / "files").mkdir(parents=True, exist_ok=True)
            for p in site.pages.values():
                (d / p.path).write_text(render(p), encoding="utf-8")
            (d / "files" / "brochure.pdf").write_bytes(b"%PDF-1.4 dummy brochure\n")
            (d / "files" / "secure-update.apk").write_bytes(b"harmless dummy file, not an app\n")
            gt = site.ground_truth()
            (out / "truth").mkdir(exist_ok=True)
            (out / "truth" / f"{site.id}.json").write_text(json.dumps(gt, indent=1))
            truths.append(gt)
    (out / "index.json").write_text(json.dumps([{k: t[k] for k in ("id", "theme", "split")}
                                                for t in truths], indent=1))
    return truths


# ------------------------------------------------------------------ server


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep evaluation output clean
        pass


def load_truth(root: Path, site_id: str) -> dict:
    return json.loads((root / "truth" / f"{site_id}.json").read_text())


def serve(root: Path) -> tuple[ThreadingHTTPServer, str]:
    """Serve root/www (the sites only, never the ground truth) on 127.0.0.1, random
    free port. Returns (server, base_url)."""
    handler = partial(_QuietHandler, directory=str(root / "www"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/blackbox_sites")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20)
    a = ap.parse_args()
    truths = generate(Path(a.out), a.n_train, a.n_test, a.seed)
    n_b = sum(len(t["planted"]) for t in truths)
    print(f"wrote {len(truths)} sites ({n_b} planted behaviours) to {a.out}")
