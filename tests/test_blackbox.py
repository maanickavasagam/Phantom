from phantom.blackbox.investigate import action_features, allowed_url, scenario_html


def test_local_allowlist():
    assert allowed_url("http://127.0.0.1:8000/synthetic/fees/start")
    assert not allowed_url("https://example.com/synthetic/fees/start")
    assert not allowed_url("http://127.0.0.1:8000/other")
    assert not allowed_url("http://evil.test@127.0.0.1:8000/synthetic/fees/start")


def test_synthetic_ground_truth_and_features():
    assert 'data-behaviour="pricing"' in scenario_html("fees", "clue")
    assert "data-behaviour" not in scenario_html("fees", "safe")
    assert action_features("Review fee details") != action_features("Overview")
