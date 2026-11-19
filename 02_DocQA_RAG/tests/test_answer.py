"""Tests for docqa.answer."""

from pathlib import Path

import httpx
import openai
import pytest
from docqa.answer import (
    REFUSAL_MESSAGE,
    AnswerGenerator,
    Citation,
    load_system_prompt,
    validate_question,
)
from docqa.errors import ConfigError, InvalidQuestionError, LLMCallError
from docqa.retriever import RetrievedChunk
from docqa.store import SearchHit

REQUEST = httpx.Request("POST", "https://example.test/v1")


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


def _generator(llm, sleeps=None, **kwargs) -> AnswerGenerator:
    sleeps = [] if sleeps is None else sleeps
    return AnswerGenerator(
        llm,
        "test-model",
        "SYSTEM",
        max_question_chars=100,
        sleep=sleeps.append,
        **kwargs,
    )


# ---------- question guards ----------


def test_question_is_stripped():
    assert validate_question("  hello?  ", 100) == "hello?"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_raises(question: str):
    with pytest.raises(InvalidQuestionError):
        validate_question(question, 100)


def test_question_length_limit():
    assert validate_question("x" * 100, 100) == "x" * 100
    with pytest.raises(InvalidQuestionError):
        validate_question("x" * 101, 100)


def test_invalid_question_makes_no_llm_call(llm):
    with pytest.raises(InvalidQuestionError):
        _generator(llm).answer("   ", _two_chunks())
    assert llm.calls == []


# ---------- prompt file ----------


def test_real_prompt_loads_with_the_refusal_filled_in():
    prompt = load_system_prompt()
    assert REFUSAL_MESSAGE in prompt
    assert "{refusal}" not in prompt


def test_missing_prompt_file_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_system_prompt(tmp_path / "nope.txt")


def test_empty_prompt_file_raises(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("  \n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_system_prompt(path)


# ---------- answering ----------


def test_no_evidence_refuses_without_calling_the_llm(llm):
    answer = _generator(llm).answer("anything?", [])
    assert answer.refused
    assert answer.text == REFUSAL_MESSAGE
    assert answer.citations == ()
    assert llm.calls == []


def test_request_carries_model_temperature_prompt_context_and_question(llm):
    _generator(llm).answer("When was it founded?", _two_chunks())
    call = llm.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0.0
    assert call["messages"][0] == {"role": "system", "content": "SYSTEM"}
    user = call["messages"][1]["content"]
    assert "[1] a.md (Doc > One, lines 3-5)\nFounded in 2015." in user
    assert "[2] a.md (Doc > Two, line 7)\nBased in Austin." in user
    assert user.endswith("Question: When was it founded?")


def test_only_cited_passages_become_citations(llm):
    llm.reply = "It was founded in 2015 [1]."
    answer = _generator(llm).answer("When?", _two_chunks())
    assert not answer.refused
    assert [c.number for c in answer.citations] == [1]
    assert answer.citations[0].section == "Doc > One"
    assert (answer.citations[0].start_line, answer.citations[0].end_line) == (3, 5)


def test_multiple_citation_markers_are_all_kept(llm):
    llm.reply = "Founded in 2015 [1] and based in Austin [2]."
    answer = _generator(llm).answer("Tell me.", _two_chunks())
    assert [c.number for c in answer.citations] == [1, 2]


def test_answer_without_markers_cites_every_retrieved_passage(llm):
    llm.reply = "Founded in 2015."
    answer = _generator(llm).answer("When?", _two_chunks())
    assert [c.number for c in answer.citations] == [1, 2]


def test_out_of_range_marker_is_ignored(llm):
    llm.reply = "Founded in 2015 [9]."
    answer = _generator(llm).answer("When?", _two_chunks())
    assert [c.number for c in answer.citations] == [1, 2]


@pytest.mark.parametrize(
    "reply", [REFUSAL_MESSAGE, REFUSAL_MESSAGE.rstrip("."), f"  {REFUSAL_MESSAGE}\n"]
)
def test_model_refusal_clears_citations(llm, reply: str):
    llm.reply = reply
    answer = _generator(llm).answer("Who is the CEO of Mars Inc?", _two_chunks())
    assert answer.refused
    assert answer.text == REFUSAL_MESSAGE
    assert answer.citations == ()


def test_pdf_citation_reports_the_page(llm):
    llm.reply = "See the report [1]."
    chunks = [_rc("Revenue grew.", source="docs/report.pdf", page=4)]
    citation = _generator(llm).answer("Revenue?", chunks).citations[0]
    assert citation.source == "report.pdf"
    assert citation.page == 4
    assert citation.location() == "page 4"


def test_token_counts_are_recorded(llm):
    answer = _generator(llm).answer("When?", _two_chunks())
    assert (answer.prompt_tokens, answer.completion_tokens) == (11, 5)


def test_snippet_collapses_whitespace_and_truncates(llm):
    chunks = [_rc("word   " * 100)]
    snippet = _generator(llm).answer("Q?", chunks).citations[0].snippet
    assert "  " not in snippet
    assert snippet.endswith("...")
    assert len(snippet) <= 203


def test_citation_location_formats():
    assert (
        Citation(1, "a.md", "s", section="A > B", start_line=3, end_line=9).location()
        == "A > B, lines 3-9"
    )
    assert Citation(1, "a.md", "s", start_line=4, end_line=4).location() == "line 4"
    assert Citation(1, "a.md", "s").location() == ""


# ---------- retry and failures ----------


def test_rate_limit_is_retried_then_succeeds(llm):
    sleeps: list[float] = []
    llm.failures = [_status_error(openai.RateLimitError, 429)]
    answer = _generator(llm, sleeps).answer("When?", _two_chunks())
    assert not answer.refused
    assert len(llm.calls) == 2
    assert len(sleeps) == 1


def test_auth_error_is_not_retried(llm):
    llm.failures = [_status_error(openai.AuthenticationError, 401)]
    with pytest.raises(LLMCallError):
        _generator(llm).answer("When?", _two_chunks())
    assert len(llm.calls) == 1


def test_attempts_exhausted_raises_llm_call_error(llm):
    llm.failures = [_status_error(openai.InternalServerError, 503)] * 3
    with pytest.raises(LLMCallError) as info:
        _generator(llm, max_attempts=3).answer("When?", _two_chunks())
    assert len(llm.calls) == 3
    assert isinstance(info.value.__cause__, openai.InternalServerError)


@pytest.mark.parametrize("reply", [None, "", "   "])
def test_empty_model_reply_raises(llm, reply):
    llm.reply = reply
    with pytest.raises(LLMCallError):
        _generator(llm).answer("When?", _two_chunks())


@pytest.mark.parametrize(
    ("reply", "expected_text", "expected_numbers"),
    [
        ("Founded in 2015【1】", "Founded in 2015[1]", [1]),
        ("Founded [1, 2].", "Founded [1][2].", [1, 2]),
        ("Based in Austin［2］", "Based in Austin[2]", [2]),
    ],
)
def test_alternative_citation_markers_are_normalised(llm, reply, expected_text, expected_numbers):
    llm.reply = reply
    answer = _generator(llm).answer("Q?", _two_chunks())
    assert answer.text == expected_text
    assert [c.number for c in answer.citations] == expected_numbers
