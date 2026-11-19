"""Tests for docqa.config."""

import pytest
from docqa.config import PROJECT_ROOT, Settings, load_settings
from docqa.errors import ConfigError
from pydantic import ValidationError


def test_defaults_when_environment_is_empty():
    s = load_settings({})
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (800, 100, 4)
    assert s.log_level == "INFO"
    assert s.api_key is None
    assert s.embedding_model == "openai/text-embedding-3-small"
    assert (s.embed_batch_size, s.max_attempts, s.retry_base_delay) == (64, 3, 1.0)
    assert s.relevance_threshold == 0.25


def test_environment_values_override_defaults():
    s = load_settings(
        {
            "DOCQA_CHUNK_SIZE": "500",
            "DOCQA_CHUNK_OVERLAP": "50",
            "DOCQA_TOP_K": "6",
            "DOCQA_EMBEDDING_MODEL": "some/model",
            "DOCQA_RETRY_BASE_DELAY": "0.5",
        }
    )
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (500, 50, 6)
    assert s.embedding_model == "some/model"
    assert s.retry_base_delay == 0.5


def test_empty_environment_variable_falls_back_to_default():
    assert load_settings({"DOCQA_CHUNK_SIZE": ""}).chunk_size == 800


def test_default_paths_live_under_the_project_root():
    s = load_settings({})
    assert s.source_dir.is_relative_to(PROJECT_ROOT)
    assert s.store_dir.is_relative_to(PROJECT_ROOT)


def test_explicit_mapping_ignores_the_real_environment(monkeypatch):
    monkeypatch.setenv("DOCQA_TOP_K", "9")
    assert load_settings({}).top_k == 4


def test_default_reads_the_real_environment(monkeypatch):
    monkeypatch.setenv("DOCQA_TOP_K", "9")
    assert load_settings().top_k == 9


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DOCQA_CHUNK_SIZE", "abc"),
        ("DOCQA_CHUNK_SIZE", "0"),
        ("DOCQA_TOP_K", "0"),
        ("DOCQA_TOP_K", "21"),
        ("DOCQA_RELEVANCE_THRESHOLD", "1.5"),
        ("DOCQA_RELEVANCE_THRESHOLD", "-0.1"),
        ("DOCQA_TEMPERATURE", "3"),
        ("DOCQA_LOG_LEVEL", "LOUD"),
        ("DOCQA_MAX_FILE_MB", "0"),
        ("DOCQA_MAX_ATTEMPTS", "0"),
        ("DOCQA_EMBED_BATCH_SIZE", "0"),
        ("DOCQA_RETRY_BASE_DELAY", "-1"),
    ],
)
def test_invalid_value_raises_config_error_naming_the_variable(name: str, value: str):
    with pytest.raises(ConfigError, match=name):
        load_settings({name: value})


def test_overlap_not_smaller_than_size_raises():
    env = {"DOCQA_CHUNK_SIZE": "100", "DOCQA_CHUNK_OVERLAP": "100"}
    with pytest.raises(ConfigError, match="chunk_overlap"):
        load_settings(env)


def test_every_problem_is_reported_at_once():
    with pytest.raises(ConfigError) as info:
        load_settings({"DOCQA_TOP_K": "0", "DOCQA_LOG_LEVEL": "LOUD"})
    message = str(info.value)
    assert "DOCQA_TOP_K" in message
    assert "DOCQA_LOG_LEVEL" in message


def test_missing_api_key_raises_only_when_demanded():
    settings = load_settings({})
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        settings.require_api_key()


def test_api_key_is_returned_but_hidden_from_repr():
    settings = load_settings({"OPENROUTER_API_KEY": "sk-test-123"})
    assert settings.require_api_key() == "sk-test-123"
    assert "sk-test-123" not in repr(settings)
    assert "sk-test-123" not in str(settings)


def test_settings_are_frozen():
    settings = load_settings({})
    with pytest.raises(ValidationError):
        settings.top_k = 9


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        Settings(bogus=1)
