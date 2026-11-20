"""Runtime configuration for the DocQA pipeline.

Settings are built from environment variables. The environment is injected as
a mapping, so tests pass a plain dict and never touch the real environment.
Everything is validated once, up front, so a bad value fails fast with a
readable message instead of crashing deep inside the pipeline.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from docqa.errors import ConfigError

# Project root = the folder that contains the `docqa` package (no absolute paths).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 10
# Tuned on the InsureLM eval: weakest answerable top hit 0.31, strongest unrelated 0.17.
DEFAULT_RELEVANCE_THRESHOLD = 0.25
DEFAULT_MAX_QUESTION_CHARS = 500
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_FILE_MB = 20
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 1.0

# Maps each Settings field to the environment variable that overrides it.
ENV_VARS: dict[str, str] = {
    "chunk_size": "DOCQA_CHUNK_SIZE",
    "chunk_overlap": "DOCQA_CHUNK_OVERLAP",
    "top_k": "DOCQA_TOP_K",
    "relevance_threshold": "DOCQA_RELEVANCE_THRESHOLD",
    "max_question_chars": "DOCQA_MAX_QUESTION_CHARS",
    "llm_model": "DOCQA_LLM_MODEL",
    "base_url": "DOCQA_BASE_URL",
    "temperature": "DOCQA_TEMPERATURE",
    "api_key": "OPENROUTER_API_KEY",
    "source_dir": "DOCQA_SOURCE_DIR",
    "pdf_dir": "DOCQA_PDF_DIR",
    "store_dir": "DOCQA_STORE_DIR",
    "log_level": "DOCQA_LOG_LEVEL",
    "max_file_mb": "DOCQA_MAX_FILE_MB",
    "embedding_model": "DOCQA_EMBEDDING_MODEL",
    "embed_batch_size": "DOCQA_EMBED_BATCH_SIZE",
    "max_attempts": "DOCQA_MAX_ATTEMPTS",
    "retry_base_delay": "DOCQA_RETRY_BASE_DELAY",
}


class Settings(BaseModel):
    """Validated, immutable pipeline settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    relevance_threshold: float = Field(default=DEFAULT_RELEVANCE_THRESHOLD, ge=0, le=1)
    max_question_chars: int = Field(default=DEFAULT_MAX_QUESTION_CHARS, gt=0)
    llm_model: str = Field(default=DEFAULT_LLM_MODEL, min_length=1)
    base_url: str = Field(default=DEFAULT_BASE_URL, min_length=1)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0, le=2)
    # repr=False keeps the secret out of printed settings and log lines.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = DEFAULT_LOG_LEVEL
    api_key: str | None = Field(default=None, repr=False)
    source_dir: Path = PROJECT_ROOT / "data" / "source_md"
    pdf_dir: Path = PROJECT_ROOT / "data" / "pdf"
    store_dir: Path = PROJECT_ROOT / "data" / "chroma"
    max_file_mb: int = Field(default=DEFAULT_MAX_FILE_MB, gt=0)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, min_length=1)
    embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, ge=1, le=2048)
    max_attempts: int = Field(default=DEFAULT_MAX_ATTEMPTS, ge=1, le=10)
    retry_base_delay: float = Field(default=DEFAULT_RETRY_BASE_DELAY, ge=0, le=60)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> Self:
        """Reject an overlap that is not strictly smaller than the chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller "
                f"than chunk_size ({self.chunk_size})"
            )
        return self

    def require_api_key(self) -> str:
        """Return the API key, or raise ConfigError if it was never set.

        Ingestion and retrieval work offline, so the key is only demanded
        right before the first LLM call.
        """
        if not self.api_key:
            raise ConfigError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.api_key


def _format_validation_error(exc: ValidationError) -> str:
    """Turn a pydantic ValidationError into one readable line per problem."""
    lines = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"])
        # Show the env var the user can actually fix; model-level errors have no field.
        name = ENV_VARS.get(field, field) or "settings"
        lines.append(f"{name}: {err['msg']}")
    return "Invalid configuration:\n  " + "\n  ".join(lines)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from the environment, raising ConfigError on bad values.

    Args:
        env: Mapping to read from. Defaults to ``os.environ``; tests pass a dict.
    """
    source = os.environ if env is None else env
    # Only pass variables that are set and non-empty, so model defaults apply
    # to everything else.
    raw = {field: source[name] for field, name in ENV_VARS.items() if source.get(name)}
    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
