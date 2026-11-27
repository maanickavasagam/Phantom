"""Sandboxed screenshot preview of a suspected website (opt-in, one click).

The user never loads the page. Our server opens it in a throwaway headless
browser with these restrictions:

- JavaScript OFF (no scripts, no exploit kits, no fingerprinting);
- fresh context: no cookies, no storage, service workers blocked, no
  permissions, downloads refused;
- only http/https to PUBLIC addresses: every request (including redirects and
  images) is checked, and anything that resolves to a private, loopback,
  link-local or reserved address is blocked (no probing our own network);
- at most MAX_REQUESTS requests, media/fonts/websockets blocked, 15 s limit;
- nothing is clicked or typed; no form is ever submitted.

It records a screenshot, the page title, where redirects really end, and
whether the page asks for a password, then re-checks the FINAL address with
the URL checker and compares the page's claimed brand with its domain.

Privacy / risk notes (shown in the UI): the site's server sees one visit from
our machine's IP address, which can tell a scammer that the link was opened.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from functools import lru_cache
from urllib.parse import urlparse

from phantom.blackbox.urlcheck import brand_mentions, check_url
from phantom.common.evidence_store import EvidenceStore

TIMEOUT_MS = 15_000  # whole visit
GOTO_MS = 10_000  # first HTML must arrive within this
SHOT_MS = 3_000
MAX_REQUESTS = 60
VIEWPORT = {"width": 1280, "height": 800}
BLOCKED_TYPES = {"media", "font", "websocket", "eventsource", "manifest", "other"}
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36")


class PreviewRefused(ValueError):
    """The URL is not allowed to be previewed (scheme, private address...)."""


@lru_cache(maxsize=512)
def _host_status(host: str) -> str:
    """'public', 'private' (any address non-public) or 'unresolved'."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (OSError, UnicodeError):
        return "unresolved"
    addrs = {info[4][0].split("%")[0] for info in infos}
    if not addrs:
        return "unresolved"
    for a in addrs:
        ip = ipaddress.ip_address(a)
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified or not ip.is_global):
            return "private"
    return "public"


def _public_host(host: str) -> bool:
    return bool(host) and _host_status(host) == "public"


def refusal_reason(url: str) -> str | None:
    """Why url may not be previewed, or None if it may."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return "only http/https web addresses can be previewed"
    status = _host_status(p.hostname)
    if status == "unresolved":
        return (f"{p.hostname} does not exist in DNS (not registered, or already taken "
                "down), so there is no page to show")
    if status == "private":
        return f"{p.hostname} points to a private or local network address: not allowed"
    return None


def _launch(playwright):
    args = ["--disable-extensions", "--disable-plugins", "--no-first-run",
            "--disable-background-networking", "--disable-sync", "--mute-audio"]
    try:
        return playwright.chromium.launch(headless=True, args=args)
    except Exception:
        try:
            return playwright.chromium.launch(channel="msedge", headless=True, args=args)
        except Exception as error:
            raise RuntimeError("could not start a headless browser (install Playwright "
                               "Chromium or Microsoft Edge)") from error


def safe_preview(url: str, store: EvidenceStore) -> dict:
    """Open url in the sandbox; return what was seen. Raises PreviewRefused."""
    raw = url.strip()
    if "://" not in raw:
        raw = "https://" + raw
    reason = refusal_reason(raw)
    if reason:
        raise PreviewRefused(reason)
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    stats = {"requests": 0, "blocked": []}

    def guard(route):
        req = route.request
        u = urlparse(req.url)
        reason = None
        if u.scheme in ("data", "blob"):
            return route.continue_()
        if u.scheme not in ("http", "https"):
            reason = "non-web scheme"
        elif req.resource_type in BLOCKED_TYPES:
            reason = f"{req.resource_type} blocked"
        elif stats["requests"] >= MAX_REQUESTS:
            reason = "request limit"
        elif not _public_host(u.hostname or ""):
            reason = "private or unknown address"
        if reason:
            if len(stats["blocked"]) < 20:
                stats["blocked"].append(f"{reason}: {req.url[:120]}")
            return route.abort()
        stats["requests"] += 1
        return route.continue_()

    t0 = time.monotonic()
    out: dict = {"requested_url": raw, "sandbox": {
        "javascript": False, "cookies": False, "downloads": False, "forms_submitted": False,
        "max_requests": MAX_REQUESTS, "timeout_s": TIMEOUT_MS / 1000}}
    with sync_playwright() as playwright:
        browser = _launch(playwright)
        try:
            context = browser.new_context(
                java_script_enabled=False, accept_downloads=False, service_workers="block",
                viewport=VIEWPORT, user_agent=USER_AGENT, locale="en-IN", permissions=[],
                ignore_https_errors=False)
            context.set_default_timeout(SHOT_MS)
            context.route("**/*", guard)
            page = context.new_page()
            response = None
            try:
                response = page.goto(raw, wait_until="domcontentloaded", timeout=GOTO_MS)
                left = TIMEOUT_MS - (time.monotonic() - t0) * 1000 - SHOT_MS
                if left > 500:  # let images/CSS arrive, but never past the limit
                    try:
                        page.wait_for_load_state("load", timeout=left)
                    except PlaywrightError:
                        pass
            except PlaywrightError as error:
                msg = str(error).splitlines()[0][:200]
                out["load_error"] = msg
                if "ERR_CERT" in msg or "SSL" in msg:
                    out["certificate_error"] = True
            chain = []
            req = response.request if response else None
            while req is not None:
                chain.append(req.url)
                req = req.redirected_from
            out["redirects"] = list(reversed(chain))
            out["final_url"] = page.url if page.url.startswith("http") else None
            out["status"] = response.status if response else None
            if response is None:  # nothing arrived: no page to inspect or photograph
                out["title"] = ""
            else:
                try:
                    out["title"] = (page.title() or "")[:200]
                    # read the HTML once instead of querying a live page (which waits
                    # for navigations to settle)
                    html = page.content().lower()
                    out["password_fields"] = len(re.findall(
                        r"<input[^>]*type\s*=\s*[\"']?password", html))
                    out["forms"] = html.count("<form")
                    png = page.screenshot(full_page=False, type="png", timeout=SHOT_MS,
                                          animations="disabled")
                    out["screenshot_id"] = store.put(
                        png, kind="screenshot", source=f"preview of {raw[:200]}",
                        meta={"final_url": out["final_url"], "title": out["title"]})
                except PlaywrightError as error:
                    out.setdefault("load_error", str(error).splitlines()[0][:200])
        finally:
            browser.close()
    out["requests"] = stats["requests"]
    out["blocked_requests"] = stats["blocked"]
    out["seconds"] = round(time.monotonic() - t0, 2)
    out["observations"] = _observations(raw, out)
    return out


def _observations(requested: str, seen: dict) -> list[dict]:
    """What the visit adds to the address-only verdict."""
    obs: list[dict] = []
    final = seen.get("final_url")
    first = check_url(requested)
    if final:
        last = check_url(final)
        seen["final_check"] = {"verdict": last.verdict, "brand": last.brand,
                               "registrable": last.registrable, "summary": last.summary}
        if last.registrable and last.registrable != first.registrable:
            obs.append({"name": "redirects_elsewhere", "strength": "medium",
                        "detail": f"the link ends on a different site: {last.registrable} "
                                  f"(started at {first.registrable})"})
    title = seen.get("title") or ""
    claimed = brand_mentions(title.lower().replace(" ", "")) if title else []
    site_brand = (seen.get("final_check") or {}).get("brand") or first.brand
    site_official = (seen.get("final_check") or {}).get("verdict", first.verdict) == "official"
    for b, _kind, _m in claimed[:1]:
        if not (site_official and site_brand == b.key):
            obs.append({"name": "page_claims_brand", "strength": "strong",
                        "detail": f"the page title '{title[:80]}' presents itself as {b.name}, "
                                  f"but the site is not an official {b.name} domain"})
    if seen.get("password_fields"):
        verdict = (seen.get("final_check") or {}).get("verdict", first.verdict)
        risky = bool(claimed) or verdict in ("likely_fake", "suspicious")
        if not site_official:  # a login box on the brand's own site is normal
            obs.append({
                "name": "asks_for_password", "strength": "medium" if risky else "weak",
                "detail": f"the page has {seen['password_fields']} password box(es)"
                          + (": a fake site asking for a password is how accounts get stolen"
                             if risky else " (normal for a login page; matters only if the "
                                           "site is not who it claims to be)")})
    if seen.get("certificate_error"):
        obs.append({"name": "certificate_error", "strength": "medium",
                    "detail": "the site's HTTPS certificate is invalid"})
    if seen.get("load_error"):
        obs.append({"name": "did_not_load", "strength": "weak",
                    "detail": f"the page could not be loaded ({seen['load_error'][:120]})"})
    return obs
