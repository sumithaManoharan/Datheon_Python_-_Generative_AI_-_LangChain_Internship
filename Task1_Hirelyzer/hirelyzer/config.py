"""Runtime settings for Hirelyzer, read from environment variables.

Nothing here runs at import time. Call load_settings() once at startup.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from hirelyzer.errors import ConfigError

# Defaults live here as named constants, not buried inside functions.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEXT_MODEL = "openai/gpt-oss-20b"
DEFAULT_VISION_MODEL = "qwen/qwen3.7-flash"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_REPAIR_ATTEMPTS = 2
DEFAULT_MAX_FILE_MB = 10
DEFAULT_MAX_TEXT_CHARS = 60_000

# The first variable that is set wins, so OpenRouter or OpenAI keys both work.
API_KEY_VARS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")


@dataclass(frozen=True)
class Settings:
    """All runtime settings, validated once at startup and never changed."""

    api_key: str = field(repr=False)  # repr=False keeps the key out of logs
    base_url: str
    text_model: str
    vision_model: str
    temperature: float
    max_retries: int
    max_repair_attempts: int
    max_file_mb: int
    max_text_chars: int

    @property
    def max_file_bytes(self) -> int:
        """Size limit in bytes, derived from max_file_mb."""
        return self.max_file_mb * 1024 * 1024


def _find_api_key(env: Mapping[str, str]) -> str:
    """Return the first non-empty API key, or fail fast with a readable message."""
    for name in API_KEY_VARS:
        value = env.get(name, "").strip()
        if value:
            return value
    raise ConfigError(
        "No API key found. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) "
        "in your environment or in a .env file."
    )


def _read_text(env: Mapping[str, str], name: str, default: str) -> str:
    """Read a string setting, falling back to the default when unset or blank."""
    return env.get(name, "").strip() or default


def _read_number[T: (int, float)](
    env: Mapping[str, str], name: str, default: T, cast: Callable[[str], T]
) -> T:
    """Read an int or float setting; turn a bad value into a ConfigError."""
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a {cast.__name__}, got {raw!r}") from exc


def _validate(settings: Settings) -> None:
    """Reject values that parse fine but make no sense."""
    if not 0.0 <= settings.temperature <= 2.0:
        raise ConfigError(
            f"HIRELYZER_TEMPERATURE must be between 0 and 2, got {settings.temperature}"
        )
    if settings.max_retries < 0 or settings.max_repair_attempts < 0:
        raise ConfigError("Retry settings cannot be negative.")
    if settings.max_file_mb <= 0 or settings.max_text_chars <= 0:
        raise ConfigError("Size limits must be positive numbers.")


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build validated Settings from the environment (or from a dict in tests)."""
    env = os.environ if env is None else env
    settings = Settings(
        api_key=_find_api_key(env),
        base_url=_read_text(env, "HIRELYZER_BASE_URL", DEFAULT_BASE_URL),
        text_model=_read_text(env, "HIRELYZER_TEXT_MODEL", DEFAULT_TEXT_MODEL),
        vision_model=_read_text(env, "HIRELYZER_VISION_MODEL", DEFAULT_VISION_MODEL),
        temperature=_read_number(env, "HIRELYZER_TEMPERATURE", DEFAULT_TEMPERATURE, float),
        max_retries=_read_number(env, "HIRELYZER_MAX_RETRIES", DEFAULT_MAX_RETRIES, int),
        max_repair_attempts=_read_number(
            env, "HIRELYZER_REPAIR_ATTEMPTS", DEFAULT_MAX_REPAIR_ATTEMPTS, int
        ),
        max_file_mb=_read_number(env, "HIRELYZER_MAX_FILE_MB", DEFAULT_MAX_FILE_MB, int),
        max_text_chars=_read_number(env, "HIRELYZER_MAX_TEXT_CHARS", DEFAULT_MAX_TEXT_CHARS, int),
    )
    _validate(settings)
    return settings
