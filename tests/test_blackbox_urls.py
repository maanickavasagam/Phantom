import pytest

from phantom.blackbox.urlcheck import check_url, osa_distance, skeleton, split_host
from phantom.blackbox.urltool import check_url_tool
from phantom.common.evidence_store import EvidenceStore


@pytest.mark.parametrize("url,brand", [
    ("https://www.amazon.in/dp/B0C", "amazon"), ("https://amzn.to/3abc", "amazon"),
    ("https://www.flipkart.com/offers", "flipkart"), ("https://onlinesbi.sbi/", "sbi"),
    ("https://hdfc.bank.in", "hdfc"), ("https://www.irctc.co.in", "irctc"),
    ("https://www.indiapost.gov.in", "indiapost"), ("https://amazonaws.com", "amazon"),
])
def test_official_domains(url, brand):
    r = check_url(url)
    assert (r.verdict, r.brand) == ("official", brand)


@pytest.mark.parametrize("url,brand,signal", [
    ("http://amaz0n-offers.shop/login", "amazon", "homoglyph"),
    ("https://flipkarrt.com", "flipkart", "typosquat"),
    ("https://flipkart-bigbillion-sale.xyz/claim", "flipkart", "brand_in_domain"),
    ("https://amazon.in.verify-account.top/kyc", "amazon", "brand_in_subdomain"),
    ("https://xn--mazon-3ve.com", "amazon", "homoglyph"),
    ("https://sbi-kyc-update.online/", "sbi", "brand_in_domain"),
    ("https://paytrn.in/reward", "paytm", "homoglyph"),
    ("https://www.amazon.in@evil-site.ru/login", "amazon", "brand_in_userinfo"),
    ("https://irctc-tatkal.co.in", "irctc", "brand_in_domain"),
])
def test_lookalikes_are_likely_fake(url, brand, signal):
    r = check_url(url)
    assert (r.verdict, r.brand) == ("likely_fake", brand)
    assert signal in {s.name for s in r.signals}


@pytest.mark.parametrize("url,verdict", [
    ("https://myntra.shop", "suspicious"),          # exact name, unofficial ending
    ("https://bit.ly/3xYz", "suspicious"),          # shortener hides destination
    ("https://example.com/amazon/login", "suspicious"),
    ("http://192.168.1.10/hdfc/login", "suspicious"),
    ("https://www.wikipedia.org", "no_brand_link"),
    ("https://sbitech.com", "no_brand_link"),       # "sbi" inside a word is not SBI
    ("https://github.com/login", "no_brand_link"),
])
def test_other_verdicts(url, verdict):
    assert check_url(url).verdict == verdict


def test_helpers():
    assert skeleton("аmaz0n") == "amazon"  # Cyrillic a + zero
    assert osa_distance("flipkart", "filpkart") == 1  # swapped letters = one typo
    assert split_host("hdfc.bank.in") == ("", "hdfc", "bank.in")


def test_tool_result_and_invalid(tmp_path):
    store = EvidenceStore(tmp_path / "ev")
    res = check_url_tool("https://flipkarrt.com", store)
    assert res.findings[0].check == "verdict_likely_fake"
    assert res.findings[0].outcome == "suspicious"
    assert res.evidence_ids and res.artifacts["url_check"]["brand"] == "flipkart"
    assert check_url_tool("", store).errors[0].code == "invalid_url"
    assert check_url_tool("http://" + "a" * 3000 + ".com", store).errors


def test_api_and_coordinator(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from phantom.api import main
    from phantom.coordinator.engine import coordinate

    monkeypatch.setattr(main, "store", main.EvidenceStore(tmp_path / "ev"))
    client = TestClient(main.app)
    body = client.post("/blackbox/check-url", json={"url": "https://phonepe-cashback.in"}).json()
    assert body["artifacts"]["url_check"]["verdict"] == "likely_fake"
    report = coordinate("Is this website link genuine?", main.store,
                        website_url="https://www.flipkart.com")
    assert report["trace"][0]["status"] == "completed"
    assert report["findings"][0]["check"] == "verdict_official"


@pytest.mark.parametrize("url,verdict,brand", [
    # user content on a brand's hosting service is not "official"
    ("https://s3.amazonaws.com/bucket/amazon-login.html", "suspicious", None),
    ("https://hdfcpagekyc-08.web.app/", "likely_fake", "hdfc"),  # brand joined to other text
    ("am-azon.com", "likely_fake", "amazon"),  # brand split by a hyphen
    ("amaz.on.in", "likely_fake", "amazon"),  # brand split by a dot
    ("xn--nnazon-25b.in", "likely_fake", "amazon"),  # non-Latin letters, 2 edits
    ("amazonia.bo", "no_brand_link", None),  # the rainforest, not the shop
    ("amazonvideo.com", "official", "amazon"),
    ("citibank.com", "no_brand_link", None),  # 2 edits from icicibank: not a typo
    ("mediapost.com", "no_brand_link", None),
    ("phonepower.com", "no_brand_link", None),
    ("teacherspayteachers.com", "no_brand_link", None),
])
def test_v2_regressions(url, verdict, brand):
    r = check_url(url)
    assert r.verdict == verdict, (url, r.verdict, [s.name for s in r.signals])
    if brand:
        assert r.brand == brand
