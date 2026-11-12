"""Tests for the extraction pipeline: validation, the repair loop and the result envelope."""

import json
from typing import Any

import pytest
from hirelyzer.errors import LLMCallError
from hirelyzer.extractor import ExtractionResult, ResumeExtractor
from hirelyzer.prompts import PromptBuilder

NON_RESUME = '{"is_resume": false}'
VALID_RESUME = json.dumps(
    {"is_resume": True, "language": "English", "name": "Jordan Avery", "skills": ["Python", "SQL"]}
)
BAD_OUTPUTS = [
    '{"is_resume": true, "email": "not-an-email"}',
    '{"is_resume": true, "nickname": "extra keys are forbidden"}',
    '{"is_resume": true, "experience": [{"start_date": "March 2022"}]}',
    '{"is_resume": true, "name": "Jordan"',  # truncated JSON
    "I could not read that document.",
]


def run_extraction(
    prompt_builder: PromptBuilder, llm: Any, max_repairs: int = 2
) -> ExtractionResult:
    """Run one extraction against a fake LLM."""
    extractor = ResumeExtractor(llm, prompt_builder, max_repairs)
    return extractor.extract("some resume text", source="demo.pdf")


def test_valid_resume_succeeds_first_try(prompt_builder: PromptBuilder, make_llm: Any) -> None:
    result = run_extraction(prompt_builder, make_llm([VALID_RESUME]))

    assert result.status == "success"
    assert result.attempts == 1
    assert result.errors == []
    assert result.data is not None
    assert result.data.name == "Jordan Avery"
    assert result.source == "demo.pdf"
    assert result.latency_ms >= 0


def test_non_resume_succeeds_without_a_repair(prompt_builder: PromptBuilder, make_llm: Any) -> None:
    result = run_extraction(prompt_builder, make_llm([NON_RESUME]))

    assert result.status == "success"
    assert result.attempts == 1
    assert result.data is not None
    assert result.data.is_resume is False


def test_prompt_contains_the_document_text(prompt_builder: PromptBuilder, make_llm: Any) -> None:
    llm = make_llm([NON_RESUME])

    run_extraction(prompt_builder, llm)

    system, user = llm.calls[0]
    assert system["role"] == "system"
    assert "some resume text" in user["content"]


@pytest.mark.parametrize("bad", BAD_OUTPUTS)
def test_repair_loop_recovers_after_one_bad_reply(
    prompt_builder: PromptBuilder, make_llm: Any, bad: str
) -> None:
    llm = make_llm([bad, VALID_RESUME])

    result = run_extraction(prompt_builder, llm)

    assert result.status == "success"
    assert result.attempts == 2
    assert len(result.errors) == 1  # the first failure is kept as evidence
    repair_request = llm.calls[1]
    assert len(repair_request) == 4  # system, user, rejected reply, error message
    assert repair_request[2] == {"role": "assistant", "content": bad}
    assert "rejected" in repair_request[3]["content"]


def test_repair_loop_gives_up_after_the_limit(prompt_builder: PromptBuilder, make_llm: Any) -> None:
    llm = make_llm([BAD_OUTPUTS[0]] * 3)

    result = run_extraction(prompt_builder, llm, max_repairs=2)

    assert result.status == "failed"
    assert result.attempts == 3
    assert len(result.errors) == 3
    assert result.data is None


def test_zero_repairs_means_one_attempt(prompt_builder: PromptBuilder, make_llm: Any) -> None:
    llm = make_llm([BAD_OUTPUTS[0]])

    result = run_extraction(prompt_builder, llm, max_repairs=0)

    assert result.status == "failed"
    assert result.attempts == 1
    assert len(llm.calls) == 1


def test_llm_failure_stops_immediately_without_repair(
    prompt_builder: PromptBuilder, make_llm: Any
) -> None:
    llm = make_llm([LLMCallError("boom")])

    result = run_extraction(prompt_builder, llm)

    assert result.status == "failed"
    assert result.attempts == 1
    assert result.errors == ["boom"]
    assert len(llm.calls) == 1


def test_tokens_accumulate_across_attempts(prompt_builder: PromptBuilder, make_llm: Any) -> None:
    result = run_extraction(prompt_builder, make_llm([BAD_OUTPUTS[0], VALID_RESUME]))

    assert result.prompt_tokens == 20
    assert result.completion_tokens == 10
