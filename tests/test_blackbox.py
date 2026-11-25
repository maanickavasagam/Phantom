import json
import random

import pytest

pytest.importorskip("playwright")

from phantom.blackbox.browser import SafeBrowser  # noqa: E402
from phantom.blackbox.crawl_check import crawl  # noqa: E402
from phantom.blackbox.explore import FixedRuleStrategy, RandomStrategy, run_episode  # noqa: E402
from phantom.blackbox.sites import generate, load_truth, serve  # noqa: E402


@pytest.fixture(scope="module")
def sites(tmp_path_factory):
    root = tmp_path_factory.mktemp("bb")
    generate(root, n_train=2, n_test=1, seed=3)
    srv, base = serve(root)
    try:
        with SafeBrowser(base) as b:
            yield root, base, b
    except Exception as e:  # browser binary not installed in this environment
        pytest.skip(f"playwright browser unavailable: {e}")
    finally:
        srv.shutdown()


def _ids(root):
    return [s["id"] for s in json.loads((root / "index.json").read_text())]


def test_ground_truth_is_not_served(sites):
    root, base, b = sites
    sid = _ids(root)[0]
    assert (root / "truth" / f"{sid}.json").exists()
    st = b.load(f"{base}/{sid}/index.html")
    assert "ground_truth" not in st.text
    import urllib.error
    import urllib.request

    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"{base}/../truth/{sid}.json")


def test_page_names_do_not_reveal_behaviours(sites):
    root, _, _ = sites
    for sid in _ids(root):
        for page in load_truth(root, sid)["edges"]:
            assert page == "index.html" or page[0] == "p", page


def test_allowlist_and_no_files_or_external_forms(sites):
    root, base, b = sites
    with pytest.raises(PermissionError):
        b.load("https://www.example.org/")
    for sid in _ids(root):
        st = b.load(f"{base}/{sid}/index.html")
        for a in b.actions(st):
            assert a.target.startswith(base)
            assert not a.target.lower().split("?")[0].endswith((".apk", ".pdf"))


def test_full_crawl_finds_planted_and_no_decoys(sites):
    root, base, b = sites
    for sid in _ids(root):
        r = crawl(root, sid, b, base)
        assert r["missed"] == [] and r["false"] == []


def test_episode_respects_budget(sites):
    root, base, b = sites
    sid = _ids(root)[0]
    for strat in (RandomStrategy(), FixedRuleStrategy()):
        ep = run_episode(b, base, sid, strat, 5, random.Random(0))
        assert len(ep.steps) <= 5
