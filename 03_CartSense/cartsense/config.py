"""Environment-driven configuration for CartSense."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_MAX_STEPS = 40
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEARCH_REGION = "in-en"
REGION_PATTERN = r"^[a-z]{2}-[a-z]{2}$"


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    """Validated settings, built once at startup."""

    api_key: str
    base_url: str
    model: str
    max_steps: int
    temperature: float
    search_region: str = DEFAULT_SEARCH_REGION
    search_provider: str = "ddgs"
    tavily_api_key: str = field(default="", repr=False)


def _read_int(name: str, default: int) -> int:
    """Read an integer env var; raise ConfigError if it is not a valid int."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from None


def _read_float(name: str, default: float) -> float:
    """Read a float env var; raise ConfigError if it is not a valid number."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from None


def load_config() -> Config:
    """Load .env, validate everything, and return a Config.

    Fails fast with a readable ConfigError instead of a later API error.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    max_steps = _read_int("CARTSENSE_MAX_STEPS", DEFAULT_MAX_STEPS)
    if max_steps < 1:
        raise ConfigError("CARTSENSE_MAX_STEPS must be at least 1.")

    temperature = _read_float("CARTSENSE_TEMPERATURE", DEFAULT_TEMPERATURE)
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError("CARTSENSE_TEMPERATURE must be between 0.0 and 2.0.")

    region = os.environ.get("CARTSENSE_SEARCH_REGION", DEFAULT_SEARCH_REGION).strip().lower()
    if not re.fullmatch(REGION_PATTERN, region):
        raise ConfigError("CARTSENSE_SEARCH_REGION must look like 'in-en' or 'us-en'.")

    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    provider = os.environ.get("CARTSENSE_SEARCH_PROVIDER", "").strip().lower()
    provider = provider or ("tavily" if tavily_key else "ddgs")
    if provider not in ("ddgs", "tavily"):
        raise ConfigError("CARTSENSE_SEARCH_PROVIDER must be ddgs or tavily.")
    if provider == "tavily" and not tavily_key:
        raise ConfigError("TAVILY_API_KEY is not set. Add it to .env or use ddgs.")

    return Config(
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip(),
        model=os.environ.get("CARTSENSE_MODEL", DEFAULT_MODEL).strip(),
        max_steps=max_steps,
        temperature=temperature,
        search_region=region,
        search_provider=provider,
        tavily_api_key=tavily_key,
    )
