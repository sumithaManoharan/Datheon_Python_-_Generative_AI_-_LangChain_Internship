"""Thin wrapper around the OpenAI-compatible client: retries, backoff and telemetry hooks."""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import openai

from hirelyzer.config import Settings
from hirelyzer.errors import LLMCallError
from hirelyzer.telemetry import RequestRecord, TelemetryRecorder

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 60.0


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Seconds to wait before retry number `attempt` (1 = first retry), with full jitter."""
    ceiling = min(cap, base * 2 ** (attempt - 1))
    return random.uniform(0, ceiling)


def is_transient(exc: Exception) -> bool:
    """Return True if the error is worth retrying."""
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code == 429 or exc.status_code >= 500:
            return True
    return False


@dataclass(frozen=True)
class LLMResponse:
    """What one successful call returns: the text, token usage and how many tries it took."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    attempts: int


class LLMClient:
    """Makes chat calls with JSON mode, temperature control and retry on transient errors."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        recorder: TelemetryRecorder | None = None,
    ) -> None:
        """Take the SDK client, sleep function and recorder as arguments so tests can fake them."""
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._sleep = sleep
        self._recorder = recorder

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        recorder: TelemetryRecorder | None = None,
        model: str | None = None,
    ) -> "LLMClient":
        """Build the real client. The SDK's own retries are off so ours are the visible ones."""
        sdk_client = openai.OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=REQUEST_TIMEOUT_S,
            max_retries=0,
        )
        return cls(
            sdk_client,
            model or settings.text_model,
            settings.temperature,
            settings.max_retries,
            recorder=recorder,
        )

    def complete(
        self, messages: list[dict[str, str]], source: str = "", pipeline: str = "text"
    ) -> LLMResponse:
        """Send the messages, retrying transient failures; raise LLMCallError when giving up."""
        total_attempts = self._max_retries + 1  # first try plus the allowed retries
        started = time.perf_counter()

        def emit(
            attempt: int,
            success: bool,
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
            error: str = "",
        ) -> None:
            """Report this request to the recorder (if any), with the total elapsed time."""
            if self._recorder is None:
                return
            self._recorder.record(
                RequestRecord(
                    source=source,
                    model=self._model,
                    pipeline=pipeline,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retries=attempt - 1,
                    success=success,
                    error=error,
                )
            )

        for attempt in range(1, total_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                )
            except openai.OpenAIError as exc:
                # Give up on permanent errors (auth, bad request) or when tries run out.
                if not is_transient(exc) or attempt == total_attempts:
                    message = f"LLM request failed after {attempt} attempt(s): {exc}"
                    emit(attempt, False, error=message)
                    raise LLMCallError(message) from exc
                delay = backoff_delay(attempt)
                logger.warning(
                    "Transient LLM error on attempt %d/%d (%s); retrying in %.1fs",
                    attempt,
                    total_attempts,
                    type(exc).__name__,
                    delay,
                )
                self._sleep(delay)
                continue

            # The call succeeded, but the reply can still be unusable.
            usage = response.usage  # some providers omit usage, so default to 0
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            content = response.choices[0].message.content if response.choices else None
            if not content or not content.strip():
                message = "The model returned an empty response."
                emit(attempt, False, prompt_tokens, completion_tokens, message)
                raise LLMCallError(message)

            emit(attempt, True, prompt_tokens, completion_tokens)
            return LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                attempts=attempt,
            )

        # The loop always returns or raises; this line only satisfies type checkers.
        raise LLMCallError("LLM request failed without a result.")
