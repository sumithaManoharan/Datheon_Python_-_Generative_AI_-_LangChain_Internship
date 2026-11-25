"""Tavily search tests. The real client is blocked by conftest; fakes only."""

import pytest

from cartsense.config import ConfigError, load_config
from cartsense.tools import ToolError, make_tavily_search, web_search

PRODUCT = "https://www.amazon.in/Bosch/dp/B07JW77MVJ"


def _install_fake(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            calls.append({"api_key": api_key})

        def search(self, query: str, **kwargs: object) -> dict[str, object]:
            calls.append({"query": query, **kwargs})
            hit = {"title": "T", "url": PRODUCT, "content": "Price Rs 52,021"}
            return {"results": [hit]}

    monkeypatch.setattr("tavily.TavilyClient", FakeClient)
    return calls


def test_tavily_results_use_the_common_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    hits = make_tavily_search("k")("bosch dishwasher", 5)
    assert hits == [{"title": "T", "href": PRODUCT, "body": "Price Rs 52,021"}]


def test_store_sets_include_domains_and_basic_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake(monkeypatch)
    make_tavily_search("k", "amazon")("bosch", 3)
    options = calls[-1]
    assert options["include_domains"] == ["amazon.in"]
    assert options["search_depth"] == "basic"
    assert options["max_results"] == 3


def test_unknown_store_is_rejected() -> None:
    with pytest.raises(ToolError):
        make_tavily_search("k", "ebay")


def test_web_search_can_use_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake(monkeypatch)
    results = web_search(
        "bosch", 5, provider="tavily", tavily_api_key="secret", sleep=lambda _: None
    )
    assert results == [{"title": "T", "url": PRODUCT, "snippet": "Price Rs 52,021"}]
    assert calls[0] == {"api_key": "secret"}


def _config(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    monkeypatch.setattr("cartsense.config.load_dotenv", lambda *a, **k: None)
    for name in ("TAVILY_API_KEY", "CARTSENSE_SEARCH_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def test_provider_defaults_follow_the_tavily_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    assert load_config().search_provider == "ddgs"
    _config(monkeypatch, TAVILY_API_KEY="t")
    assert load_config().search_provider == "tavily"


def test_tavily_without_a_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, CARTSENSE_SEARCH_PROVIDER="tavily")
    with pytest.raises(ConfigError, match="TAVILY_API_KEY"):
        load_config()


def test_unknown_provider_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, CARTSENSE_SEARCH_PROVIDER="bing")
    with pytest.raises(ConfigError, match="CARTSENSE_SEARCH_PROVIDER"):
        load_config()
