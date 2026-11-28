"""Tests for retry classification, backoff and the LLM client (no network)."""

from typing import Any

import httpx
import openai
import pytest
from hirelyzer.errors import LLMCallError
from hirelyzer.llm import LLMClient, backoff_delay, is_transient
from hirelyzer.telemetry import TelemetryRecorder

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    ("cls", "code", "expected"),
    [
        (openai.RateLimitError, 429, True),
        (openai.InternalServerError, 500, True),
        (openai.InternalServerError, 503, True),
        (openai.AuthenticationError, 401, False),
        (openai.BadRequestError, 400, False),
        (openai.PermissionDeniedError, 403, False),
        (openai.NotFoundError, 404, False),
    ],
)
def test_is_transient_by_status(
    make_sdk_error: Any, cls: type[openai.APIStatusError], code: int, expected: bool
) -> None:
    assert is_transient(make_sdk_error(cls, code)) is expected


def test_timeouts_and_connection_errors_are_transient(http_request: httpx.Request) -> None:
    assert is_transient(openai.APITimeoutError(request=http_request))
    assert is_transient(openai.APIConnectionError(request=http_request))


def test_unrelated_exceptions_are_not_transient() -> None:
    assert not is_transient(ValueError("a bug in our own code"))


def test_backoff_delay_is_bounded_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the "random" part to its upper bound so the ceiling itself is visible.
    monkeypatch.setattr("hirelyzer.llm.random.uniform", lambda low, high: high)
    assert [backoff_delay(n) for n in range(1, 8)] == [1, 2, 4, 8, 16, 30, 30]


def test_backoff_delay_has_jitter_below_the_ceiling() -> None:
    assert all(0 <= backoff_delay(3) <= 4 for _ in range(50))


def test_success_on_first_try_sends_expected_parameters(
    make_sdk: Any, make_sdk_response: Any, sleeps: list[float]
) -> None:
    sdk = make_sdk([make_sdk_response(prompt_tokens=7, completion_tokens=3)])
    client = LLMClient(sdk, "test-model", temperature=0.0, sleep=sleeps.append)

    response = client.complete(MESSAGES)

    assert (response.attempts, response.prompt_tokens, response.completion_tokens) == (1, 7, 3)
    assert sleeps == []
    request = sdk.calls[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0.0
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"] == MESSAGES


def test_rate_limit_twice_then_success(
    make_sdk: Any, make_sdk_error: Any, make_sdk_response: Any, sleeps: list[float]
) -> None:
    errors = [make_sdk_error(openai.RateLimitError, 429) for _ in range(2)]
    sdk = make_sdk([*errors, make_sdk_response()])

    response = LLMClient(sdk, "m", max_retries=3, sleep=sleeps.append).complete(MESSAGES)

    assert response.attempts == 3
    assert len(sdk.calls) == 3
    assert len(sleeps) == 2


def test_timeout_then_success(
    make_sdk: Any, make_sdk_response: Any, http_request: httpx.Request, sleeps: list[float]
) -> None:
    sdk = make_sdk([openai.APITimeoutError(request=http_request), make_sdk_response()])

    response = LLMClient(sdk, "m", sleep=sleeps.append).complete(MESSAGES)

    assert response.attempts == 2


def test_auth_error_is_not_retried(make_sdk: Any, make_sdk_error: Any, sleeps: list[float]) -> None:
    sdk = make_sdk([make_sdk_error(openai.AuthenticationError, 401)])

    with pytest.raises(LLMCallError):
        LLMClient(sdk, "m", sleep=sleeps.append).complete(MESSAGES)

    assert len(sdk.calls) == 1
    assert sleeps == []


def test_retries_exhausted_raises_with_original_cause(
    make_sdk: Any, make_sdk_error: Any, sleeps: list[float]
) -> None:
    sdk = make_sdk([make_sdk_error(openai.InternalServerError, 500) for _ in range(4)])

    with pytest.raises(LLMCallError, match="4 attempt") as excinfo:
        LLMClient(sdk, "m", max_retries=3, sleep=sleeps.append).complete(MESSAGES)

    assert isinstance(excinfo.value.__cause__, openai.InternalServerError)
    assert len(sdk.calls) == 4
    assert len(sleeps) == 3  # no wasted wait after the final attempt


def test_zero_retries_means_a_single_attempt(
    make_sdk: Any, make_sdk_error: Any, sleeps: list[float]
) -> None:
    sdk = make_sdk([make_sdk_error(openai.RateLimitError, 429)])

    with pytest.raises(LLMCallError):
        LLMClient(sdk, "m", max_retries=0, sleep=sleeps.append).complete(MESSAGES)

    assert len(sdk.calls) == 1


@pytest.mark.parametrize("content", ["", "   ", None], ids=["empty", "blank", "none"])
def test_empty_reply_is_a_hard_failure_not_a_retry(
    make_sdk: Any, make_sdk_response: Any, sleeps: list[float], content: str | None
) -> None:
    sdk = make_sdk([make_sdk_response(content=content)])

    with pytest.raises(LLMCallError, match="empty"):
        LLMClient(sdk, "m", sleep=sleeps.append).complete(MESSAGES)

    assert len(sdk.calls) == 1


def test_missing_usage_defaults_to_zero_tokens(make_sdk: Any, make_sdk_response: Any) -> None:
    sdk = make_sdk([make_sdk_response(with_usage=False)])

    response = LLMClient(sdk, "m").complete(MESSAGES)

    assert (response.prompt_tokens, response.completion_tokens) == (0, 0)


def test_recorder_logs_retries_tokens_and_source(
    make_sdk: Any, make_sdk_error: Any, make_sdk_response: Any, sleeps: list[float]
) -> None:
    recorder = TelemetryRecorder()
    sdk = make_sdk([make_sdk_error(openai.RateLimitError, 429), make_sdk_response()])

    LLMClient(sdk, "m", sleep=sleeps.append, recorder=recorder).complete(MESSAGES, source="a.pdf")

    (record,) = recorder.records
    assert record.success
    assert (record.retries, record.source, record.model) == (1, "a.pdf", "m")
    assert (record.prompt_tokens, record.completion_tokens) == (10, 5)
    assert record.latency_ms >= 0


def test_recorder_logs_failures(make_sdk: Any, make_sdk_error: Any, sleeps: list[float]) -> None:
    recorder = TelemetryRecorder()
    sdk = make_sdk([make_sdk_error(openai.AuthenticationError, 401)])

    with pytest.raises(LLMCallError):
        LLMClient(sdk, "m", sleep=sleeps.append, recorder=recorder).complete(MESSAGES)

    (record,) = recorder.records
    assert not record.success
    assert "failed" in record.error
