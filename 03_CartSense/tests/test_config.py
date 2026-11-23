"""Tests for configuration loading."""

import pytest

from cartsense.config import ConfigError, load_config


def test_search_region_defaults_to_india(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.delenv("CARTSENSE_SEARCH_REGION", raising=False)
    assert load_config().search_region == "in-en"


def test_search_region_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("CARTSENSE_SEARCH_REGION", "UK-EN")
    assert load_config().search_region == "uk-en"


@pytest.mark.parametrize("value", ["india", "in_en", "in-eng", ""])
def test_bad_search_region_fails_fast(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("CARTSENSE_SEARCH_REGION", value)
    with pytest.raises(ConfigError, match="CARTSENSE_SEARCH_REGION"):
        load_config()
