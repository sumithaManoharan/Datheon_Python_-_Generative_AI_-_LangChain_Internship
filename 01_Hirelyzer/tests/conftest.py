"""Shared test doubles: a fake OpenAI-style client, a fake LLM, and SDK object builders."""

from collections.abc import Callable, Iterator, Sequence
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from hirelyzer.llm import LLMResponse
from hirelyzer.prompts import PromptBuilder
from hirelyzer.telemetry import telemetry_logger


class FakeSDKClient:
    """Mimics openai.OpenAI().chat.completions.create, returning or raising scripted items."""

    def __init__(self, script: Sequence[Any]) -> None:
        """Take the scripted results; an Exception in the script is raised, not returned."""
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []  # keyword arguments of every request
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        """Record the request, then play the next scripted item."""
        self.calls.append(kwargs)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeLLM:
    """Stands in for LLMClient inside the extractor; the script holds reply texts or errors."""

    def __init__(self, script: Sequence[str | Exception]) -> None:
        """Take the scripted replies."""
        self._script = list(script)
        self.calls: list[list[dict[str, str]]] = []  # the messages of every request

    def complete(
        self, messages: list[dict[str, str]], source: str = "", pipeline: str = "text"
    ) -> LLMResponse:
        """Record the messages, then return or raise the next scripted item."""
        self.calls.append(messages)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(content=item, prompt_tokens=10, completion_tokens=5, attempts=1)


def build_sdk_response(
    content: str | None = "{}",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    with_usage: bool = True,
) -> SimpleNamespace:
    """Build an object shaped like an SDK chat completion."""
    message = SimpleNamespace(content=content)
    usage = (
        SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if with_usage
        else None
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture
def http_request() -> httpx.Request:
    """A dummy HTTP request, needed to construct SDK exceptions."""
    return httpx.Request("POST", "https://example.com")


@pytest.fixture
def make_sdk_error(http_request: httpx.Request) -> Callable[..., openai.APIStatusError]:
    """Factory: make_sdk_error(openai.RateLimitError, 429) builds a realistic SDK error."""

    def build(cls: type[openai.APIStatusError], status_code: int) -> openai.APIStatusError:
        response = httpx.Response(status_code, request=http_request)
        return cls("boom", response=response, body=None)

    return build


@pytest.fixture
def make_sdk_response() -> Callable[..., SimpleNamespace]:
    """Factory for fake SDK chat completions."""
    return build_sdk_response


@pytest.fixture
def make_sdk() -> type[FakeSDKClient]:
    """Factory for a fake SDK client: make_sdk([error, error, response])."""
    return FakeSDKClient


@pytest.fixture
def make_llm() -> type[FakeLLM]:
    """Factory for a fake LLM: make_llm(["bad json", '{"is_resume": false}'])."""
    return FakeLLM


@pytest.fixture
def sleeps() -> list[float]:
    """Collects backoff delays; pass sleeps.append as the sleep function so tests run instantly."""
    return []


@pytest.fixture(scope="session")
def prompt_builder() -> PromptBuilder:
    """The real prompt builder, built once from the files in prompts/."""
    return PromptBuilder.from_files()


@pytest.fixture(autouse=True)
def _reset_telemetry_logger() -> Iterator[None]:
    """After every test, detach any file handler so tests never leak into each other."""
    yield
    for handler in telemetry_logger.handlers[:]:
        telemetry_logger.removeHandler(handler)
        handler.close()
    telemetry_logger.propagate = True
