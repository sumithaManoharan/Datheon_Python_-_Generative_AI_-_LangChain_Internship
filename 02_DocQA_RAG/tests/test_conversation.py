"""Tests for docqa.conversation and the history helpers in docqa.answer."""

import pytest
from docqa.answer import CONDENSE_PROMPT, AnswerGenerator, Turn, strip_citations
from docqa.conversation import DEFAULT_MAX_TURNS, ConversationMemory
from docqa.errors import LLMCallError


def _generator(llm) -> AnswerGenerator:
    return AnswerGenerator(
        llm, "test-model", "SYSTEM", max_question_chars=100, sleep=lambda _seconds: None
    )


# ---------- memory ----------


def test_the_default_memory_holds_six_turns():
    assert ConversationMemory().max_turns == DEFAULT_MAX_TURNS == 6


def test_turns_are_kept_oldest_first():
    memory = ConversationMemory()
    memory.add("q1", "a1")
    memory.add("q2", "a2")
    assert [(t.question, t.answer) for t in memory.turns] == [("q1", "a1"), ("q2", "a2")]
    assert len(memory) == 2


def test_the_oldest_turn_is_dropped_beyond_the_limit():
    memory = ConversationMemory(max_turns=3)
    for number in range(1, 6):
        memory.add(f"q{number}", f"a{number}")
    assert [t.question for t in memory.turns] == ["q3", "q4", "q5"]


def test_clear_forgets_everything():
    memory = ConversationMemory()
    memory.add("q", "a")
    memory.clear()
    assert len(memory) == 0
    assert memory.turns == ()


def test_turns_is_a_snapshot_not_a_live_view():
    memory = ConversationMemory()
    snapshot = memory.turns
    memory.add("q", "a")
    assert snapshot == ()


@pytest.mark.parametrize("bad", [0, -1])
def test_a_limit_below_one_is_rejected(bad: int):
    with pytest.raises(ValueError):
        ConversationMemory(max_turns=bad)


# ---------- strip_citations ----------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Avery [1].", "Avery."),
        ("A [1][2] and B [3].", "A and B."),
        ("A [1, 2].", "A."),
        ("No markers.", "No markers."),
    ],
)
def test_strip_citations_removes_old_passage_numbers(text: str, expected: str):
    assert strip_citations(text) == expected


# ---------- condense ----------


def test_condense_without_history_returns_the_question_and_makes_no_call(llm):
    assert _generator(llm).condense("Who founded it?", ()) == "Who founded it?"
    assert llm.calls == []


def test_condense_asks_the_model_for_a_standalone_question(llm):
    llm.reply = "What is Jordan's job title?\n"
    history = (Turn("Who won the award?", "Jordan won it [1]."),)
    result = _generator(llm).condense("What is their job title?", history)
    assert result == "What is Jordan's job title?"
    system, user = llm.calls[0]["messages"]
    assert system["content"] == CONDENSE_PROMPT
    assert "Q: Who won the award?" in user["content"]
    assert "A: Jordan won it." in user["content"]
    assert "Follow-up question: What is their job title?" in user["content"]


def test_condense_falls_back_to_the_original_when_the_rewrite_is_too_long(llm):
    llm.reply = "x" * 200
    history = (Turn("q", "a"),)
    assert _generator(llm).condense("And then?", history) == "And then?"


def test_condense_with_an_empty_reply_raises(llm):
    llm.reply = ""
    with pytest.raises(LLMCallError):
        _generator(llm).condense("And then?", (Turn("q", "a"),))
