"""Tests for docqa.retry."""

import httpx
import openai
import pytest
from docqa.errors import ConfigError
from docqa.retry import call_with_retry, is_transient_api_error

REQUEST = httpx.Request("POST", "https://example.test/v1")


def _status_error(cls, status: int):
    return cls("boom", response=httpx.Response(status, request=REQUEST), body=None)


class Flaky:
    """Callable that raises the scripted errors in order, then returns 'ok'."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def _is_timeout(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError)


def test_success_on_first_try_does_not_sleep():
    sleeps: list[float] = []
    func = Flaky([])
    result = call_with_retry(
        func, is_transient=_is_timeout, max_attempts=3, base_delay=1.0, sleep=sleeps.append
    )
    assert result == "ok"
    assert func.calls == 1
    assert sleeps == []


def test_transient_errors_back_off_exponentially_then_succeed():
    sleeps: list[float] = []
    func = Flaky([TimeoutError(), TimeoutError()])
    result = call_with_retry(
        func,
        is_transient=_is_timeout,
        max_attempts=3,
        base_delay=1.0,
        sleep=sleeps.append,
        jitter=lambda: 1.0,
    )
    assert result == "ok"
    assert func.calls == 3
    assert sleeps == [1.0, 2.0]


def test_jitter_scales_the_delay_down():
    sleeps: list[float] = []
    call_with_retry(
        Flaky([TimeoutError(), TimeoutError()]),
        is_transient=_is_timeout,
        max_attempts=3,
        base_delay=1.0,
        sleep=sleeps.append,
        jitter=lambda: 0.0,
    )
    assert sleeps == [0.5, 1.0]


def test_delay_is_capped():
    sleeps: list[float] = []
    call_with_retry(
        Flaky([TimeoutError()] * 4),
        is_transient=_is_timeout,
        max_attempts=5,
        base_delay=1.0,
        max_delay=3.0,
        sleep=sleeps.append,
        jitter=lambda: 1.0,
    )
    assert sleeps == [1.0, 2.0, 3.0, 3.0]


def test_non_transient_error_is_raised_immediately():
    sleeps: list[float] = []
    func = Flaky([ValueError("bad")])
    with pytest.raises(ValueError):
        call_with_retry(
            func, is_transient=_is_timeout, max_attempts=3, base_delay=1.0, sleep=sleeps.append
        )
    assert func.calls == 1
    assert sleeps == []


def test_attempts_exhausted_raises_the_last_error():
    sleeps: list[float] = []
    func = Flaky([TimeoutError()] * 3)
    with pytest.raises(TimeoutError):
        call_with_retry(
            func, is_transient=_is_timeout, max_attempts=3, base_delay=1.0, sleep=sleeps.append
        )
    assert func.calls == 3
    assert len(sleeps) == 2


def test_invalid_max_attempts_raises_config_error():
    with pytest.raises(ConfigError):
        call_with_retry(Flaky([]), is_transient=_is_timeout, max_attempts=0, base_delay=1.0)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_status_error(openai.RateLimitError, 429), True),
        (_status_error(openai.InternalServerError, 500), True),
        (_status_error(openai.InternalServerError, 503), True),
        (openai.APIConnectionError(request=REQUEST), True),
        (openai.APITimeoutError(request=REQUEST), True),
        (_status_error(openai.AuthenticationError, 401), False),
        (_status_error(openai.BadRequestError, 400), False),
        (ValueError("not an api error"), False),
    ],
)
def test_is_transient_api_error(exc, expected):
    assert is_transient_api_error(exc) is expected
