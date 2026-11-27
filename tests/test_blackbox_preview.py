"""BLACKBOX 'how it was judged' explanation and the sandboxed preview's safety
rules. No network: DNS answers are faked and the browser is not started."""

import pytest
from fastapi.testclient import TestClient

from phantom.blackbox import preview
from phantom.blackbox.preview import PreviewRefused, _observations, refusal_reason, safe_preview
from phantom.blackbox.urlcheck import CHECKS, check_url, explain


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    table = {"example.com": "93.184.215.14", "amazon-sale-offer.xyz": "45.33.32.156",
             "intranet.corp": "10.0.0.5", "localhost": "127.0.0.1",
             "rebind.example": "169.254.169.254"}

    def getaddrinfo(host, port, *a, **k):
        if host in table:
            return [(2, 1, 6, "", (table[host], 0))]
        try:  # literal IPs resolve to themselves
            import ipaddress
            ipaddress.ip_address(host)
            return [(2, 1, 6, "", (host, 0))]
        except ValueError:
            raise OSError("NXDOMAIN") from None

    preview._host_status.cache_clear()
    monkeypatch.setattr(preview.socket, "getaddrinfo", getaddrinfo)
    yield
    preview._host_status.cache_clear()


# ------------------------------------------------------------- preview safety rules
@pytest.mark.parametrize("url,why", [
    ("http://127.0.0.1:8000/admin", "private"),
    ("http://localhost/", "private"),
    ("http://10.0.0.5/", "private"),
    ("http://intranet.corp/", "private"),
    ("http://169.254.169.254/latest/meta-data/", "private"),  # cloud metadata endpoint
    ("http://[::1]/", "private"),
    ("file:///C:/Windows/win.ini", "http/https"),
    ("javascript:alert(1)", "http/https"),
    ("ftp://example.com/x", "http/https"),
    ("https://sbi-kyc-taken-down.online/", "does not exist"),
])
def test_unsafe_or_missing_targets_are_refused(url, why):
    reason = refusal_reason(url)
    assert reason and why in reason
    with pytest.raises(PreviewRefused):
        safe_preview(url, store=None)  # refused before any browser starts


def test_public_site_is_allowed():
    assert refusal_reason("https://example.com/") is None
    assert refusal_reason("http://amazon-sale-offer.xyz/login") is None


def test_redirect_into_private_network_is_blocked_by_the_request_guard():
    # the guard re-checks every request (redirect targets, images...) the same way
    assert not preview._public_host("rebind.example")
    assert not preview._public_host("localhost")
    assert preview._public_host("example.com")


# ------------------------------------------------------------- what the visit adds
def test_page_claiming_a_brand_on_a_fake_domain():
    seen = {"final_url": "http://amazon-sale-offer.xyz/login", "title": "Amazon.in Sign-In",
            "password_fields": 1}
    obs = {o["name"]: o for o in _observations("http://amazon-sale-offer.xyz/login", seen)}
    assert obs["page_claims_brand"]["strength"] == "strong"
    assert obs["asks_for_password"]["strength"] == "medium"


def test_redirect_to_another_site_is_reported():
    seen = {"final_url": "https://flipkart-prize.top/win", "title": ""}
    obs = {o["name"] for o in _observations("https://bit.ly/abc", seen)}
    assert "redirects_elsewhere" in obs
    assert seen["final_check"]["verdict"] == "likely_fake"


def test_login_box_on_official_site_is_not_a_warning():
    seen = {"final_url": "https://www.amazon.in/ap/signin", "title": "Amazon Sign-In",
            "password_fields": 1}
    assert _observations("https://www.amazon.in/ap/signin", seen) == []


def test_login_box_on_unrelated_site_is_only_weak():
    seen = {"final_url": "https://github.com/login", "title": "Sign in to GitHub",
            "password_fields": 1}
    obs = _observations("https://github.com/login", seen)
    assert [(o["name"], o["strength"]) for o in obs] == [("asks_for_password", "weak")]


# ------------------------------------------------------------- how it was judged
def test_explain_homoglyph_shows_the_swapped_letter():
    url = "https://xn--mazon-3ve.com/login"
    x = explain(check_url(url), url)
    cmp = x["comparison"]
    assert cmp["real"] == "amazon.com" and cmp["letter_level"]
    assert [o for o in cmp["ops"] if o["op"] != "equal"] == [
        {"op": "replace", "fake": "\u0430", "real": "a"}]
    assert cmp["non_latin"] == ["\u0430"]
    assert x["parts"]["decoded_domain"] == "\u0430mazon.com"


def test_explain_userinfo_trick_parts():
    url = "https://www.amazon.in@evil-site.ru/x"
    x = explain(check_url(url), url)
    assert x["parts"]["userinfo"] == "www.amazon.in"
    assert x["parts"]["domain"] == "evil-site" and x["parts"]["suffix"] == "ru"
    assert x["comparison"]["real"] == "amazon.in" and not x["comparison"]["letter_level"]
    failed = {c["key"] for c in x["checks"] if c["state"] == "fail"}
    assert {"official", "userinfo"} <= failed


def test_explain_checklist_covers_every_signal_and_official_passes():
    url = "https://www.flipkart.com/"
    x = explain(check_url(url), url)
    assert x["comparison"] is None
    assert x["checks"][0] == {"key": "official", "question": CHECKS[0][1], "state": "pass",
                              "signals": []}
    assert next(c for c in x["checks"] if c["key"] == "age")["state"] == "not_run"


def test_explain_split_brand_compares_whole_host():
    x = explain(check_url("amaz.on.in"), "amaz.on.in")
    assert x["comparison"]["fake"] == "amaz.on.in" and x["comparison"]["real"] == "amazon.in"


# ------------------------------------------------------------- API
def test_api_preview_refusal_and_screenshot_kind(tmp_path, monkeypatch):
    from phantom.api import main
    from phantom.common.evidence_store import EvidenceStore

    monkeypatch.setattr(main, "store", EvidenceStore(tmp_path / "ev"))
    client = TestClient(main.app)
    r = client.post("/blackbox/preview", json={"url": "http://127.0.0.1:8000/"})
    assert r.status_code == 400 and "private" in r.json()["detail"]
    doc = main.store.put(b"<html>not an image</html>", kind="document", source="x.html")
    assert client.get(f"/evidence/{doc}/screenshot").status_code == 400
    png = main.store.put(b"\x89PNG\r\n\x1a\nfake", kind="screenshot", source="preview")
    shot = client.get(f"/evidence/{png}/screenshot")
    assert shot.status_code == 200 and shot.headers["content-type"] == "image/png"
    assert "default-src 'none'" in shot.headers["content-security-policy"]
    check = client.post("/blackbox/check-url", json={"url": "flipkarrt.com"}).json()
    assert check["artifacts"]["explain"]["comparison"]["real"] == "flipkart.com"
