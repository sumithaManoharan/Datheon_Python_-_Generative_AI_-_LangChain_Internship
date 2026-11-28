"""Tests for streaming answers (docqa.answer.AnswerGenerator.answer_stream)."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from docqa.answer import HISTORY_RULE, REFUSAL_MESSAGE, AnswerGenerator, Turn
from docqa.errors import InvalidQuestionError, LLMCallError
from docqa.retriever import RetrievedChunk
from docqa.store import SearchHit

REQUEST = httpx.Request("POST", "https://example.test/v1")


def _piece(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def _usage_chunk(prompt: int, completion: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[], usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    )


def _status_error(cls, status: int):
    return cls("boom", response=httpx.Response(status, request=REQUEST), body=None)


def _rc(text: str, source: str = "docs/a.md", **kwargs) -> RetrievedChunk:
    return RetrievedChunk(
        hit=SearchHit(text=text, source=source, distance=0.2, **kwargs), relevance=0.8
    )


def _two_chunks() -> list[RetrievedChunk]:
    return [
        _rc("Founded in 2015.", section="Doc > One", start_line=3, end_line=5),
        _rc("Based in Austin.", section="Doc > Two", start_line=7, end_line=7),
    ]


class _Failing:
    """A stream that yields some pieces, then raises."""

    def __init__(self, pieces: list[str], error: Exception) -> None:
        self._pieces = pieces
        self._error = error

    def __iter__(self):
        for text in self._pieces:
            yield _piece(text)
        raise self._error


class FakeStreamClient:
    """Stands in for an OpenAI client; each create() call returns or raises the next item."""

    def __init__(self, *results) -> None:
        self._results = list(results)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return iter(result) if isinstance(result, list) else result


def _generator(client, sleeps=None) -> AnswerGenerator:
    sleeps = [] if sleeps is None else sleeps
    return AnswerGenerator(
        client, "test-model", "SYSTEM", max_question_chars=100, sleep=sleeps.append
    )


def test_pieces_arrive_in_order_and_the_answer_is_ready_at_the_end():
    client = FakeStreamClient([_piece("Founded "), _piece("in 2015 [1]."), _usage_chunk(40, 7)])
    stream = _generator(client).answer_stream("When?", _two_chunks())

    assert list(stream) == ["Founded ", "in 2015 [1]."]

    answer = stream.answer
    assert answer.text == "Founded in 2015 [1]."
    assert answer.refused is False
    assert [c.number for c in answer.citations] == [1]
    assert (answer.prompt_tokens, answer.completion_tokens) == (40, 7)


def test_the_stream_is_requested_with_usage_reporting():
    client = FakeStreamClient([_piece("Hi [1].")])
    stream = _generator(client).answer_stream("When?", _two_chunks())
    list(stream)

    call = client.calls[0]
    assert call["stream"] is True
    assert call["stream_options"] == {"include_usage": True}
    assert call["model"] == "test-model"


def test_answer_is_not_available_before_the_stream_is_read():
    client = FakeStreamClient([_piece("Hi [1].")])
    stream = _generator(client).answer_stream("When?", _two_chunks())

    with pytest.raises(RuntimeError):
        _ = stream.answer


def test_no_tokens_reported_gives_none():
    client = FakeStreamClient([_piece("Hi [1].")])
    stream = _generator(client).answer_stream("When?", _two_chunks())
    list(stream)

    assert stream.answer.prompt_tokens is None
    assert stream.answer.completion_tokens is None


def test_no_retrieved_chunks_refuses_without_an_llm_call():
    client = FakeStreamClient()
    stream = _generator(client).answer_stream("When?", [])

    assert list(stream) == [REFUSAL_MESSAGE]
    assert stream.answer.refused is True
    assert stream.answer.citations == ()
    assert client.calls == []


def test_a_streamed_model_refusal_becomes_the_fixed_refusal():
    client = FakeStreamClient(
        [_piece("I could not find this "), _piece("in the loaded documents.")]
    )
    stream = _generator(client).answer_stream("When?", _two_chunks())
    list(stream)

    assert stream.answer.refused is True
    assert stream.answer.text == REFUSAL_MESSAGE
    assert stream.answer.citations == ()


def test_markers_are_normalized_only_in_the_final_answer():
    client = FakeStreamClient([_piece("Austin 【2】.")])
    stream = _generator(client).answer_stream("Where?", _two_chunks())

    assert list(stream) == ["Austin 【2】."]
    assert stream.answer.text == "Austin [2]."
    assert [c.number for c in stream.answer.citations] == [2]


def test_an_empty_stream_raises_when_it_ends():
    client = FakeStreamClient([_piece(""), _piece("   ")])
    stream = _generator(client).answer_stream("When?", _two_chunks())

    with pytest.raises(LLMCallError):
        list(stream)
    with pytest.raises(RuntimeError):
        _ = stream.answer


def test_an_error_mid_stream_becomes_llm_call_error_and_leaves_no_answer():
    failing = _Failing(["Founded "], openai.APIConnectionError(request=REQUEST))
    client = FakeStreamClient(failing)
    stream = _generator(client).answer_stream("When?", _two_chunks())
    received = []

    with pytest.raises(LLMCallError):
        for piece in stream:
            received.append(piece)

    assert received == ["Founded "]
    with pytest.raises(RuntimeError):
        _ = stream.answer


def test_opening_the_stream_is_retried_on_a_transient_error():
    sleeps: list[float] = []
    client = FakeStreamClient(_status_error(openai.RateLimitError, 429), [_piece("Hi [1].")])
    stream = _generator(client, sleeps).answer_stream("When?", _two_chunks())

    assert list(stream) == ["Hi [1]."]
    assert len(client.calls) == 2
    assert len(sleeps) == 1


def test_history_adds_the_history_rule_and_earlier_messages():
    client = FakeStreamClient([_piece("Hi [1].")])
    history = [Turn(question="Who founded it?", answer="Ann [1].")]
    stream = _generator(client).answer_stream("When?", _two_chunks(), history)
    list(stream)

    messages = client.calls[0]["messages"]
    assert messages[0]["content"] == "SYSTEM" + HISTORY_RULE
    assert messages[1] == {"role": "user", "content": "Who founded it?"}
    assert messages[2] == {"role": "assistant", "content": "Ann."}
    assert messages[-1]["content"].endswith("Question: When?")


def test_an_invalid_question_makes_no_llm_call():
    client = FakeStreamClient()

    with pytest.raises(InvalidQuestionError):
        _generator(client).answer_stream("   ", _two_chunks())

    assert client.calls == []
