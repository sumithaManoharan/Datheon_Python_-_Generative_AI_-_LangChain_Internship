"""Retry helper with exponential backoff and jitter for transient API errors."""

import logging
import random
import time
from collections.abc import Callable

import openai

from docqa.errors import ConfigError

logger = logging.getLogger(__name__)

HTTP_SERVER_ERROR = 500


def is_transient_api_error(exc: Exception) -> bool:
    """True for errors worth retrying: rate limits, timeouts, connection and 5xx errors.

    Authentication and bad-request errors are permanent, so they are not retried.
    """
    if isinstance(exc, openai.RateLimitError | openai.APIConnectionError):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= HTTP_SERVER_ERROR


def call_with_retry[T](
    func: Callable[[], T],
    *,
    is_transient: Callable[[Exception], bool],
    max_attempts: int,
    base_delay: float,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> T:
    """Call `func`, retrying transient failures with exponential backoff and jitter.

    The delay doubles each attempt (capped at `max_delay`) and is scaled by a random
    factor between 0.5 and 1.0. Non-transient errors, and the last attempt's error,
    are re-raised unchanged. `sleep` and `jitter` are injectable so tests run instantly.
    """
    if max_attempts < 1:
        raise ConfigError(f"max_attempts must be at least 1, got {max_attempts}")
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            if attempt == max_attempts or not is_transient(exc):
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay *= 0.5 + jitter() / 2
            logger.warning(
                "attempt %d/%d failed (%s); retrying in %.1fs", attempt, max_attempts, exc, delay
            )
            sleep(delay)
    raise AssertionError("unreachable")
