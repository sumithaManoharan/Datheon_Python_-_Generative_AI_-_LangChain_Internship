"""Embedding providers behind a small interface, so tests can inject a fake."""

import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import openai
from openai import OpenAI

from docqa.errors import ConfigError, EmbeddingError
from docqa.retry import call_with_retry, is_transient_api_error

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns text into vectors.

    `name` identifies the model; the store refuses to mix vectors from different names.
    """

    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many texts, returning one vector per text in the same order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


class OpenAIEmbedder:
    """Embedder for any OpenAI-compatible endpoint (OpenAI, OpenRouter, ...).

    The client is injected. Requests are sent in batches and retried on transient
    errors; any final failure is raised as `EmbeddingError`.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        batch_size: int = 64,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if batch_size < 1:
            raise ConfigError(f"batch_size must be at least 1, got {batch_size}")
        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._sleep = sleep
        self.name = model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches, keeping the input order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(list(texts[start : start + self._batch_size])))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string."""
        return self._embed_batch([text])[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Send one request with retry and validate the response shape."""
        try:
            response = call_with_retry(
                lambda: self._client.embeddings.create(model=self._model, input=batch),
                is_transient=is_transient_api_error,
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
                sleep=self._sleep,
            )
        except openai.OpenAIError as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        # The API returns an index per item; sort so order never depends on the server.
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(batch):
            raise EmbeddingError(f"expected {len(batch)} embeddings, got {len(items)}")
        return [list(item.embedding) for item in items]
