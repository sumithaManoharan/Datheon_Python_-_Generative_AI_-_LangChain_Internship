"""Tests for ask_user, save_wishlist, the interview flow and non-interactive tools."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from test_agent import _call, _observations, _run

from cartsense import agent as agent_module
from cartsense.agent import Requirements, build_requirements, build_tools, mode_note
from cartsense.config import Config
from cartsense.tools import (
    MAX_ANSWER_LENGTH,
    MAX_QUESTIONS,
    NO_ANSWER,
    ToolError,
    UserPrompter,
    read_wishlist,
    save_wishlist,
)

CORE_TOOLS = ["read_wishlist", "web_search", "calculator"]
LONG_CONFIG = Config(api_key="k", base_url="u", model="m", max_steps=10, temperature=0.0)


def _answers(*items: str):
    """A fake input function that returns the given answers in order."""
    queue = list(items)

    def fake_input(prompt: str) -> str:
        return queue.pop(0)

    return fake_input


def _wishlist(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "category": "phone",
        "budget": 25000,
        "years_of_use": 3,
        "warranty_years": 1,
        "priorities": ["battery life", "camera quality"],
        "dealbreakers": ["no fast charging"],
        "specs": {"ram": "8 GB"},
    }
    return {**data, **changes}


# --- UserPrompter -----------------------------------------------------------


def test_prompter_enforces_the_question_cap() -> None:
    prompter = UserPrompter(_answers(*["a"] * MAX_QUESTIONS))
    for _ in range(MAX_QUESTIONS):
        assert prompter.ask("Question?") == "a"
    with pytest.raises(ToolError, match="limit"):
        prompter.ask("One too many?")


def test_prompter_handles_blank_and_long_answers() -> None:
    prompter = UserPrompter(_answers("   ", "x" * 2000))
    assert prompter.ask("Q1?") == NO_ANSWER
    assert len(prompter.ask("Q2?")) == MAX_ANSWER_LENGTH


def test_prompter_reports_end_of_input() -> None:
    def closed(prompt: str) -> str:
        raise EOFError

    with pytest.raises(ToolError, match="No input"):
        UserPrompter(closed).ask("Budget?")


@pytest.mark.parametrize("question", ["", "   ", "q" * 301])
def test_prompter_rejects_bad_questions(question: str) -> None:
    with pytest.raises(ToolError):
        UserPrompter(_answers("a")).ask(question)


# --- save_wishlist ----------------------------------------------------------


def test_save_wishlist_writes_a_file_that_reads_back(tmp_path: Path) -> None:
    folder = tmp_path / "wishlists"
    path = save_wishlist(_wishlist(), folder)
    assert path == folder / "phone.json"
    loaded = read_wishlist("phone", folder)
    assert loaded is not None
    assert loaded.budget == 25000


def test_save_wishlist_maps_spaces_and_writes_only_one_json_file(tmp_path: Path) -> None:
    save_wishlist(_wishlist(category="Washing Machine"), tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["washing_machine.json"]


def test_save_wishlist_overwrites_an_existing_one(tmp_path: Path) -> None:
    save_wishlist(_wishlist(), tmp_path)
    save_wishlist(_wishlist(budget=30000), tmp_path)
    loaded = read_wishlist("phone", tmp_path)
    assert loaded is not None
    assert loaded.budget == 30000


@pytest.mark.parametrize(
    "data",
    [
        _wishlist(budget=-5),
        _wishlist(bogus="x"),
        _wishlist(category="../escape"),
        _wishlist(category="a/b"),
        _wishlist(priorities=["a", "b", "c"]),
        _wishlist(specs={f"k{i}": "x" * 2000 for i in range(10)}),
        "not a dict",
    ],
)
def test_save_wishlist_rejects_bad_data_and_writes_nothing(tmp_path: Path, data: Any) -> None:
    with pytest.raises(ToolError):
        save_wishlist(data, tmp_path)
    assert list(tmp_path.iterdir()) == []


# --- tool lists and modes ---------------------------------------------------


def test_default_tools_have_no_interactive_tools() -> None:
    assert [tool.name for tool in build_tools()] == CORE_TOOLS


def test_prompter_adds_ask_and_save_tools(tmp_path: Path) -> None:
    tools = build_tools(wishlist_dir=tmp_path, prompter=UserPrompter(_answers()))
    assert [tool.name for tool in tools] == [
        "read_wishlist",
        "ask_user",
        "save_wishlist",
        "web_search",
        "calculator",
    ]


def test_mode_notes_differ() -> None:
    assert "NOT available" in mode_note(False)
    assert "NOT available" not in mode_note(True)


# --- the agent with these tools --------------------------------------------


def test_interview_flow_saves_the_wishlist(tmp_path: Path) -> None:
    tools = build_tools(wishlist_dir=tmp_path, prompter=UserPrompter(_answers("25000", "3 years")))
    script = [
        _call("ask_user", {"question": "Budget?"}, "c1"),
        _call("ask_user", {"question": "Years of use?"}, "c2"),
        _call("save_wishlist", {"data": _wishlist()}, "c3"),
        AIMessage(content="Saved."),
    ]
    _, result = _run(script, tools=tools, config=LONG_CONFIG)
    assert result.status == "done"
    assert _observations(result) == ["25000", "3 years", "Saved wishlist to phone.json."]
    assert (tmp_path / "phone.json").is_file()


def test_invalid_save_becomes_an_observation(tmp_path: Path) -> None:
    tools = build_tools(wishlist_dir=tmp_path, prompter=UserPrompter(_answers()))
    script = [
        _call("save_wishlist", {"data": _wishlist(budget=-5)}),
        AIMessage(content="Will fix."),
    ]
    _, result = _run(script, tools=tools)
    assert _observations(result)[0].startswith("Error: Invalid wishlist")
    assert list(tmp_path.iterdir()) == []


def test_non_interactive_agent_cannot_ask_or_save() -> None:
    def must_not_run(prompt: str) -> str:
        raise AssertionError("ask_user must not run in non-interactive mode")

    script = [
        _call("ask_user", {"question": "Budget?"}, "c1"),
        _call("save_wishlist", {"data": _wishlist()}, "c2"),
        AIMessage(content="Proceeding with defaults."),
    ]
    _, result = _run(script, tools=build_tools())
    assert result.status == "done"
    assert "ask_user" in _observations(result)[0]
    assert "save_wishlist" in _observations(result)[1]


# --- the final-answer gate ---------------------------------------------------


def test_final_answer_without_required_tools_is_rejected() -> None:
    script = [
        AIMessage(content="Made-up answer."),
        _call("calculator", {"expression": "1 + 1"}),
        AIMessage(content="Real answer: 2."),
    ]
    _, result = _run(
        script, config=LONG_CONFIG, requirements=Requirements(frozenset({"calculator"}))
    )
    assert result.status == "done"
    assert result.final_answer == "Real answer: 2."
    assert all("Made-up" not in step.text for step in result.steps)
    assert any(step.text.startswith("CartSense rejected") for step in result.steps)


def test_agent_that_keeps_skipping_steps_fails() -> None:
    script = [AIMessage(content=f"Answer {i}.") for i in range(3)]
    model, result = _run(
        script, config=LONG_CONFIG, requirements=Requirements(frozenset({"calculator"}))
    )
    assert result.status == "failed"
    assert "calculator" in result.final_answer
    assert model.calls == 3


def test_new_wishlist_requires_questions_and_a_save(tmp_path: Path) -> None:
    tools = build_tools(wishlist_dir=tmp_path, prompter=UserPrompter(_answers("1", "2")))
    script = [
        AIMessage(content="Early answer."),
        _call("ask_user", {"question": "Budget?"}, "c1"),
        _call("ask_user", {"question": "Years?"}, "c2"),
        _call("save_wishlist", {"data": _wishlist()}, "c3"),
        AIMessage(content="Done."),
    ]
    req = Requirements(min_questions=2, save_new_wishlist=True)
    _, result = _run(script, config=LONG_CONFIG, tools=tools, requirements=req)
    assert result.status == "done"
    assert (tmp_path / "phone.json").is_file()
    assert any("CartSense rejected" in step.text for step in result.steps)


def test_existing_wishlist_needs_no_interview(tmp_path: Path) -> None:
    save_wishlist(_wishlist(), tmp_path)
    tools = build_tools(wishlist_dir=tmp_path, prompter=UserPrompter(_answers()))
    script = [_call("read_wishlist", {"category": "phone"}), AIMessage(content="Reuse.")]
    req = Requirements(min_questions=6, save_new_wishlist=True)
    _, result = _run(script, config=LONG_CONFIG, tools=tools, requirements=req)
    assert result.status == "done"


def test_urls_not_found_in_tool_results_are_rejected() -> None:
    script = [AIMessage(content="See https://fake.example/x"), AIMessage(content="No links.")]
    _, result = _run(script, config=LONG_CONFIG, requirements=Requirements(check_urls=True))
    assert result.final_answer == "No links."
    assert any("fake.example" in step.text for step in result.steps)


def test_urls_from_search_results_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [{"title": "T", "url": "https://example.com/p", "snippet": "S"}]
    monkeypatch.setattr(agent_module, "web_search", lambda query, max_results=5, **kw: hits)
    script = [
        _call("web_search", {"query": "phone"}),
        AIMessage(content="Source: https://example.com/p."),
    ]
    _, result = _run(script, config=LONG_CONFIG, requirements=Requirements(check_urls=True))
    assert result.status == "done"


def test_requirements_depend_on_mode() -> None:
    live, quiet = build_requirements(True), build_requirements(False)
    assert live.min_questions > 0 and live.save_new_wishlist
    assert quiet.min_questions == 0 and not quiet.save_new_wishlist
    assert quiet.tools == frozenset({"web_search", "calculator"})


def test_ask_user_is_closed_after_the_wishlist_is_saved(tmp_path: Path) -> None:
    tools = build_tools(wishlist_dir=tmp_path, prompter=UserPrompter(_answers("1")))
    script = [
        _call("save_wishlist", {"data": _wishlist()}, "c1"),
        _call("ask_user", {"question": "One more?"}, "c2"),
        AIMessage(content="Done."),
    ]
    _, result = _run(script, config=LONG_CONFIG, tools=tools)
    assert "interview is over" in _observations(result)[1]


# --- link, figure and query checks ------------------------------------------


def test_answer_without_a_link_is_rejected() -> None:
    script = [AIMessage(content="No link."), AIMessage(content="See https://example.com/p")]
    _, result = _run(script, config=LONG_CONFIG, requirements=Requirements(require_link=True))
    assert result.final_answer == "See https://example.com/p"
    assert any("source link" in step.text for step in result.steps)


def test_invented_price_in_the_answer_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [{"title": "Chair", "url": "https://example.com/c", "snippet": "Only Rs 2,999"}]
    monkeypatch.setattr(agent_module, "web_search", lambda query, max_results=5, **kw: hits)
    script = [
        _call("web_search", {"query": "chair"}),
        AIMessage(content="Buy it for ₹14,999."),
        AIMessage(content="Buy it for ₹2,999."),
    ]
    _, result = _run(script, config=LONG_CONFIG, requirements=Requirements(check_figures=True))
    assert result.final_answer == "Buy it for ₹2,999."
    assert any("price 14,999" in step.text for step in result.steps)


def test_unsupported_prices_and_table_scores_are_found() -> None:
    from cartsense.agent import _unsupported_figures

    table = (
        "| Model | Price | Value Score |\n|---|---|---|\n| A | ₹2,999 | 87.5 |\n| B | ₹2,500 | 70 |"
    )
    assert _unsupported_figures(table, {2999.0}, {87.5}) == ["price 2,500", "score 70"]
    assert _unsupported_figures("Value score: 87.5 for ₹2,999", {2999.0}, {87.5}) == []


def test_search_query_with_an_invented_price_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[str] = []

    def fake(query: str, max_results: int = 5, **kwargs: Any):
        runs.append(query)
        return []

    monkeypatch.setattr(agent_module, "web_search", fake)
    script = [
        _call("web_search", {"query": "chair price 14999 INR"}, "c1"),
        _call("web_search", {"query": "chair FHP1411Z9B 2026"}, "c2"),
        AIMessage(content="Ok."),
    ]
    _, result = _run(script, config=LONG_CONFIG)
    assert runs == ["chair FHP1411Z9B 2026"]
    assert _observations(result)[0].startswith("Error: the number 14999")


# --- reuse-or-update confirmation -------------------------------------------


def test_search_is_blocked_until_the_buyer_is_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[str] = []

    def fake(query: str, max_results: int = 5, **kwargs: Any):
        runs.append(query)
        return []

    monkeypatch.setattr(agent_module, "web_search", fake)
    save_wishlist(_wishlist(), tmp_path)
    tools = build_tools(wishlist_dir=tmp_path, prompter=UserPrompter(_answers("reuse")))
    script = [
        _call("read_wishlist", {"category": "phone"}, "c1"),
        _call("web_search", {"query": "phone"}, "c2"),
        _call("ask_user", {"question": "Reuse or update?"}, "c3"),
        _call("web_search", {"query": "phone"}, "c4"),
        AIMessage(content="Ok."),
    ]
    _, result = _run(script, config=LONG_CONFIG, tools=tools)
    assert _observations(result)[1].startswith("Error: a saved wishlist exists")
    assert runs == ["phone"]


def test_existing_wishlist_without_a_question_is_rejected(tmp_path: Path) -> None:
    save_wishlist(_wishlist(), tmp_path)
    tools = build_tools(wishlist_dir=tmp_path, prompter=UserPrompter(_answers("reuse")))
    script = [
        _call("read_wishlist", {"category": "phone"}, "c1"),
        AIMessage(content="Early answer."),
        _call("ask_user", {"question": "Reuse or update?"}, "c2"),
        AIMessage(content="Final answer."),
    ]
    req = Requirements(confirm_existing=True)
    _, result = _run(script, config=LONG_CONFIG, tools=tools, requirements=req)
    assert result.final_answer == "Final answer."
    assert any("reuse or update" in step.text for step in result.steps)


def test_existing_wishlist_with_a_question_passes(tmp_path: Path) -> None:
    save_wishlist(_wishlist(), tmp_path)
    tools = build_tools(wishlist_dir=tmp_path, prompter=UserPrompter(_answers("reuse")))
    script = [
        _call("read_wishlist", {"category": "phone"}, "c1"),
        _call("ask_user", {"question": "Reuse or update?"}, "c2"),
        AIMessage(content="Done."),
    ]
    _, result = _run(
        script, config=LONG_CONFIG, tools=tools, requirements=Requirements(confirm_existing=True)
    )
    assert result.status == "done"
