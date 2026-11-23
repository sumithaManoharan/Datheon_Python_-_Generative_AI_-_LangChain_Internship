"""Tests for the LangChain agent with a scripted fake model. No network."""

import json
from typing import Any

import openai
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from pydantic import Field

from cartsense import agent as agent_module
from cartsense.agent import (
    AgentError,
    Step,
    build_tools,
    format_step,
    load_system_prompt,
    run_agent,
)
from cartsense.config import Config

CONFIG = Config(api_key="k", base_url="u", model="m", max_steps=3, temperature=0.0)


class ScriptedModel(BaseChatModel):
    """A fake chat model that replays scripted messages or raises scripted exceptions."""

    script: list[Any]
    calls: int = 0
    bound: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        self.bound = [getattr(tool, "name", "?") for tool in tools]
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def _call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    """A model message that asks for one tool call."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _run(script: list[Any], config: Config = CONFIG, **kwargs: Any):
    model = ScriptedModel(script=script)
    kwargs.setdefault("system_prompt", "test prompt")
    return model, run_agent(model, config, "buy a phone", **kwargs)


def _observations(result) -> list[str]:
    return [s.text for s in result.steps if s.kind == "observation"]


def test_direct_final_answer() -> None:
    _, result = _run([AIMessage(content="Buy the blue one.")])
    assert result.status == "done"
    assert result.final_answer == "Buy the blue one."
    assert [s.kind for s in result.steps] == ["final"]


def test_tool_call_then_final_answer() -> None:
    script = [_call("calculator", {"expression": "(1000 + 200 * 3) / 2"}), AIMessage(content="800")]
    _, result = _run(script)
    assert [s.kind for s in result.steps] == ["action", "observation", "final"]
    assert _observations(result) == ["800.0"]


def test_thought_is_recorded_when_model_speaks_before_a_tool_call() -> None:
    first = AIMessage(
        content="Let me add.",
        tool_calls=[{"name": "calculator", "args": {"expression": "1 + 1"}, "id": "c1"}],
    )
    _, result = _run([first, AIMessage(content="Two.")])
    assert result.steps[0] == Step("thought", "Let me add.")


def test_on_step_listener_receives_steps_in_order() -> None:
    seen: list[str] = []
    script = [_call("calculator", {"expression": "1 + 1"}), AIMessage(content="Two.")]
    _run(script, on_step=lambda step: seen.append(step.kind))
    assert seen == ["action", "observation", "final"]


def test_tool_error_becomes_observation_and_loop_continues() -> None:
    script = [_call("calculator", {"expression": "1 / 0"}), AIMessage(content="Recovered.")]
    _, result = _run(script)
    assert _observations(result) == ["Error: Division by zero."]
    assert result.status == "done"


def test_scoring_mode_calls_tco() -> None:
    args = {"function": "tco", "arguments": {"price": 1000, "years_of_use": 2}}
    _, result = _run([_call("calculator", args), AIMessage(content="Ok.")])
    assert _observations(result) == ["1000.0"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"function": "tco", "arguments": {"price": -1, "years_of_use": 2}}, "Bad arguments"),
        ({"function": "tco", "arguments": {"bogus": 1}}, "Bad arguments"),
        ({"function": "rm_rf", "arguments": {}}, "Unknown function"),
    ],
)
def test_bad_scoring_calls_become_observations(args: dict[str, Any], expected: str) -> None:
    _, result = _run([_call("calculator", args), AIMessage(content="Ok.")])
    assert expected in _observations(result)[0]


def test_web_search_tool_returns_json_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [{"title": "T", "url": "https://example.com", "snippet": "S"}]
    monkeypatch.setattr(agent_module, "web_search", lambda query, max_results=5: hits)
    _, result = _run([_call("web_search", {"query": "best phone"}), AIMessage(content="Ok.")])
    assert json.loads(_observations(result)[0]) == hits


def test_web_search_outage_becomes_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(query: str, max_results: int = 5):
        raise agent_module.ToolError("Web search is unavailable right now.")

    monkeypatch.setattr(agent_module, "web_search", broken)
    _, result = _run([_call("web_search", {"query": "x"}), AIMessage(content="Ok.")])
    assert _observations(result) == ["Error: Web search is unavailable right now."]


def test_read_wishlist_tool_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "read_wishlist", lambda category, directory: None)
    _, result = _run([_call("read_wishlist", {"category": "phone"}), AIMessage(content="Ok.")])
    assert _observations(result) == ["No saved wishlist for 'phone'."]


def test_step_limit_stops_the_loop() -> None:
    script = [_call("calculator", {"expression": "1 + 1"}, f"c{i}") for i in range(30)]
    model, result = _run(script)
    assert result.status == "max_steps"
    assert model.calls < 30


def test_empty_final_answer_is_a_failed_result() -> None:
    _, result = _run([AIMessage(content="   ")])
    assert result.status == "failed"


def test_model_api_error_raises_agent_error() -> None:
    with pytest.raises(AgentError):
        _run([openai.OpenAIError("bad request")])


@pytest.mark.parametrize("request_text", ["", "   ", "x" * 501])
def test_bad_requests_are_rejected(request_text: str) -> None:
    with pytest.raises(AgentError):
        run_agent(ScriptedModel(script=[]), CONFIG, request_text, system_prompt="p")


def test_only_the_given_tools_are_offered_to_the_model() -> None:
    only = [
        StructuredTool.from_function(func=lambda: "0", name="noop", description="Does nothing.")
    ]
    model, _ = _run([AIMessage(content="Ok.")], tools=only)
    assert model.bound == ["noop"]


def test_default_tools_are_the_three_core_tools() -> None:
    assert [tool.name for tool in build_tools()] == ["read_wishlist", "web_search", "calculator"]


def test_system_prompt_file_loads_and_bad_files_fail(tmp_path) -> None:
    assert "CartSense" in load_system_prompt()
    with pytest.raises(AgentError):
        load_system_prompt(tmp_path / "missing.md")
    (tmp_path / "empty.md").write_text("  ")
    with pytest.raises(AgentError):
        load_system_prompt(tmp_path / "empty.md")


def test_format_step_labels() -> None:
    assert format_step(Step("action", "x()")) == "Action: x()"
    assert format_step(Step("final", "done")) == "Final answer: done"


def test_duplicate_tool_call_is_not_run_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[str] = []

    def fake(query: str, max_results: int = 5):
        runs.append(query)
        return [{"title": "T", "url": "https://example.com", "snippet": "S"}]

    monkeypatch.setattr(agent_module, "web_search", fake)
    script = [
        _call("web_search", {"query": "x"}, "c1"),
        _call("web_search", {"query": "x"}, "c2"),
        AIMessage(content="Ok."),
    ]
    _, result = _run(script)
    assert runs == ["x"]
    assert "Duplicate call" in _observations(result)[1]


def test_search_region_is_passed_to_web_search(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def fake(query: str, max_results: int = 5, region: str = "unset"):
        seen["region"] = region
        return []

    monkeypatch.setattr(agent_module, "web_search", fake)
    tools = build_tools(region="uk-en")
    _run([_call("web_search", {"query": "phone"}), AIMessage(content="Ok.")], tools=tools)
    assert seen["region"] == "uk-en"
