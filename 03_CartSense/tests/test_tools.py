"""Tests for the agent tools. No network; files only inside tmp_path."""

import json
from pathlib import Path

import pytest

from cartsense.tools import (
    MAX_SNIPPET_LENGTH,
    MAX_WISHLIST_BYTES,
    ToolError,
    calculator,
    read_wishlist,
    web_search,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data/synthetic/wishlists/phone.json"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(1000 + 200 * 3) / 2", 800),
        ("-5 + 2", -3),
        ("2 ** 10", 1024),
        ("7 // 2", 3),
        ("10 % 4", 2),
        ("0.1 + 0.2", 0.3),
    ],
)
def test_calculator_valid(expression: str, expected: float) -> None:
    assert calculator(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('x')",
        "x + 1",
        "True + 1",
        "[1, 2]",
        "2 +",
        "",
        "   ",
        "1 / 0",
        "10 ** 99999",
        "9 ** 9 ** 9",
        "1e308 * 10",
        "(-8) ** 0.5",
        "1 + " * 100 + "1",
    ],
)
def test_calculator_rejects_unsafe_or_invalid(expression: str) -> None:
    with pytest.raises(ToolError):
        calculator(expression)


def test_read_wishlist_missing_returns_none(tmp_path: Path) -> None:
    assert read_wishlist("phone", tmp_path) is None


def test_read_wishlist_reads_valid_file(tmp_path: Path) -> None:
    (tmp_path / "phone.json").write_text(SAMPLE.read_text())
    wishlist = read_wishlist("Phone ", tmp_path)
    assert wishlist is not None
    assert wishlist.category == "phone"


def test_read_wishlist_maps_spaces_to_underscores(tmp_path: Path) -> None:
    data = json.loads(SAMPLE.read_text())
    data["category"] = "washing machine"
    (tmp_path / "washing_machine.json").write_text(json.dumps(data))
    wishlist = read_wishlist("Washing  Machine", tmp_path)
    assert wishlist is not None
    assert wishlist.category == "washing machine"


@pytest.mark.parametrize("category", ["../secrets", "a/b", "", "..", "x.json", "a" * 60])
def test_read_wishlist_rejects_bad_category(tmp_path: Path, category: str) -> None:
    with pytest.raises(ToolError):
        read_wishlist(category, tmp_path)


def test_read_wishlist_rejects_corrupt_and_oversized_files(tmp_path: Path) -> None:
    (tmp_path / "phone.json").write_text("{not json")
    with pytest.raises(ToolError):
        read_wishlist("phone", tmp_path)
    (tmp_path / "phone.json").write_text("x" * (MAX_WISHLIST_BYTES + 1))
    with pytest.raises(ToolError):
        read_wishlist("phone", tmp_path)


def _flaky_search(failures: int, hits: list[dict[str, str]]):
    """Build a fake search function that fails `failures` times, then succeeds."""
    calls = {"count": 0}

    def fake(query: str, limit: int) -> list[dict[str, str]]:
        calls["count"] += 1
        if calls["count"] <= failures:
            raise RuntimeError("rate limited")
        return hits

    return fake, calls


def test_web_search_retries_then_succeeds() -> None:
    hits = [{"title": "T", "href": "https://example.com", "body": "B" * 1000}]
    fake, calls = _flaky_search(2, hits)
    sleeps: list[float] = []
    results = web_search("best phone", search_fn=fake, sleep=sleeps.append)
    assert calls["count"] == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]
    assert results[0]["url"] == "https://example.com"
    assert len(results[0]["snippet"]) == MAX_SNIPPET_LENGTH


def test_web_search_gives_up_with_tool_error() -> None:
    fake, calls = _flaky_search(99, [])
    sleeps: list[float] = []
    with pytest.raises(ToolError):
        web_search("best phone", search_fn=fake, sleep=sleeps.append)
    assert calls["count"] == 3
    assert len(sleeps) == 2


def test_web_search_rejects_empty_query_without_calling_backend() -> None:
    fake, calls = _flaky_search(0, [])
    with pytest.raises(ToolError):
        web_search("   ", search_fn=fake, sleep=lambda _: None)
    assert calls["count"] == 0
