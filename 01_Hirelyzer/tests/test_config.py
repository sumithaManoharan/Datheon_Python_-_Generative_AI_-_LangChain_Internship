"""Tests for settings loading and fail-fast validation."""

from dataclasses import FrozenInstanceError

import pytest
from hirelyzer.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TEXT_MODEL,
    load_settings,
)
from hirelyzer.errors import ConfigError

KEY = {"OPENROUTER_API_KEY": "test-key"}


def test_defaults_apply_when_only_the_key_is_set() -> None:
    settings = load_settings(KEY)

    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.text_model == DEFAULT_TEXT_MODEL
    assert settings.temperature == 0.0
    assert settings.max_retries == DEFAULT_MAX_RETRIES


@pytest.mark.parametrize("env", [{}, {"OPENROUTER_API_KEY": "   "}], ids=["missing", "blank"])
def test_missing_api_key_fails_fast_with_a_readable_message(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match="No API key found"):
        load_settings(env)


def test_openai_key_is_used_as_a_fallback() -> None:
    assert load_settings({"OPENAI_API_KEY": "k2"}).api_key == "k2"


def test_openrouter_key_wins_when_both_are_set() -> None:
    env = {"OPENROUTER_API_KEY": "k1", "OPENAI_API_KEY": "k2"}

    assert load_settings(env).api_key == "k1"


def test_environment_overrides_defaults() -> None:
    settings = load_settings(
        {
            **KEY,
            "HIRELYZER_BASE_URL": "https://api.openai.com/v1",
            "HIRELYZER_TEXT_MODEL": "gpt-4o-mini",
            "HIRELYZER_MAX_RETRIES": "5",
            "HIRELYZER_TEMPERATURE": "0.5",
        }
    )

    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.text_model == "gpt-4o-mini"
    assert settings.max_retries == 5
    assert settings.temperature == 0.5


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("HIRELYZER_TEMPERATURE", "abc", "must be a float"),
        ("HIRELYZER_TEMPERATURE", "3", "between 0 and 2"),
        ("HIRELYZER_MAX_RETRIES", "x", "must be a int"),
        ("HIRELYZER_MAX_RETRIES", "-1", "cannot be negative"),
        ("HIRELYZER_REPAIR_ATTEMPTS", "-2", "cannot be negative"),
        ("HIRELYZER_MAX_FILE_MB", "0", "must be positive"),
        ("HIRELYZER_MAX_TEXT_CHARS", "0", "must be positive"),
    ],
)
def test_invalid_values_are_rejected(name: str, value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_settings({**KEY, name: value})


def test_file_size_limit_is_converted_to_bytes() -> None:
    assert load_settings({**KEY, "HIRELYZER_MAX_FILE_MB": "2"}).max_file_bytes == 2 * 1024 * 1024


def test_api_key_is_hidden_from_repr() -> None:
    assert "test-key" not in repr(load_settings(KEY))


def test_settings_are_immutable() -> None:
    settings = load_settings(KEY)

    with pytest.raises(FrozenInstanceError):
        settings.temperature = 1.0  # type: ignore[misc]
