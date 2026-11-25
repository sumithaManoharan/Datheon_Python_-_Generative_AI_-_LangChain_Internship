"""Shared test fixtures: no test may spend Tavily credits."""

import pytest


@pytest.fixture(autouse=True)
def _block_real_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the real Tavily client fail loudly; tests that need one install a fake."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("tests must not call the real Tavily API")

    monkeypatch.setattr("tavily.TavilyClient", blocked)
