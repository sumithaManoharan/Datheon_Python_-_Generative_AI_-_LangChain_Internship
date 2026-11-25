"""Tests for the command-line entry point. No network, files only in tmp_path."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from test_agent import CONFIG, ScriptedModel

from cartsense import cli
from cartsense.agent import AgentResult, Requirements
from cartsense.config import ConfigError


class FakeStdin:
    """A stand-in for sys.stdin that reports whether it is a terminal."""

    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _fail_input(prompt: str) -> str:
    raise AssertionError("input must not be called")


def _main(argv: list[str], tmp_path: Path, script: list, *, tty: bool = False):
    model = ScriptedModel(script=script)
    code = cli.main(
        argv,
        model=model,
        config=CONFIG,
        input_fn=_fail_input,
        stdin=FakeStdin(tty),
        wishlist_dir=tmp_path / "wishlists",
        output_dir=tmp_path / "output",
        requirements=Requirements(),
    )
    return code, model


def test_parse_args_joins_words_and_reads_flags() -> None:
    args = cli.parse_args(["--no-interactive", "buy", "a", "phone"])
    assert args.request == "buy a phone"
    assert args.no_interactive is True
    assert args.category is None


def test_parse_args_requires_a_request() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args([])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I want to buy a washing machine", "washing machine"),
        ("buy a phone", "phone"),
        ("Get a NEW laptop!", "laptop"),
        ("buy ../../etc", "etc"),
        ("???", "recommendation"),
    ],
)
def test_guess_category(text: str, expected: str) -> None:
    assert cli.guess_category(text) == expected


@pytest.mark.parametrize(
    ("no_interactive", "tty", "expected"),
    [(False, True, True), (True, True, False), (False, False, False), (True, False, False)],
)
def test_is_interactive(no_interactive: bool, tty: bool, expected: bool) -> None:
    assert cli.is_interactive(no_interactive, FakeStdin(tty)) is expected


def test_save_recommendation_writes_markdown(tmp_path: Path) -> None:
    result = AgentResult("done", "Buy the blue one.", [])
    path = cli.save_recommendation(result, "buy a phone", "washing machine", tmp_path / "out")
    assert path.name == "washing_machine.md"
    assert "Buy the blue one." in path.read_text()


def test_non_interactive_run_hides_ask_user_and_saves_output(tmp_path: Path, capsys) -> None:
    code, model = _main(
        ["--no-interactive", "buy a phone"], tmp_path, [AIMessage(content="Buy X.")]
    )
    assert code == cli.EXIT_OK
    assert model.bound == ["read_wishlist", "web_search", "calculator"]
    assert "Buy X." in (tmp_path / "output" / "phone.md").read_text()
    assert "Final answer: Buy X." in capsys.readouterr().out


def test_piped_stdin_is_non_interactive_without_the_flag(tmp_path: Path) -> None:
    _, model = _main(["buy a phone"], tmp_path, [AIMessage(content="Ok.")], tty=False)
    assert "ask_user" not in model.bound
    assert "save_wishlist" not in model.bound


def test_terminal_run_offers_the_interview_tools(tmp_path: Path) -> None:
    _, model = _main(["buy a phone"], tmp_path, [AIMessage(content="Ok.")], tty=True)
    assert "ask_user" in model.bound
    assert "save_wishlist" in model.bound


def test_failed_run_returns_nonzero_and_writes_no_file(tmp_path: Path) -> None:
    code, _ = _main(["buy a phone"], tmp_path, [AIMessage(content=""), AIMessage(content="")])
    assert code == cli.EXIT_FAILED
    assert not (tmp_path / "output").exists()


def test_config_error_gives_exit_code_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken() -> None:
        raise ConfigError("OPENROUTER_API_KEY is not set.")

    monkeypatch.setattr(cli, "load_config", broken)
    assert cli.main(["buy a phone"], stdin=FakeStdin(False)) == cli.EXIT_CONFIG
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_calculation_log_lists_only_successful_calculator_calls() -> None:
    from cartsense.agent import Step

    steps = [
        Step("action", 'calculator({"expression": "1 + 1"})'),
        Step("observation", "2.0"),
        Step("action", 'calculator({"expression": "1 / 0"})'),
        Step("observation", "Error: Division by zero."),
        Step("action", 'web_search({"query": "x"})'),
        Step("observation", "[]"),
    ]
    log = cli.calculation_log(steps)
    assert "2.0" in log
    assert "Division" not in log
    assert "web_search" not in log


def _search_steps():
    from cartsense.agent import Step

    hit = '[{"title": "Chair", "url": "https://example.com/a", "snippet": "x"}]'
    return [
        Step("action", 'web_search({"query": "a"})'),
        Step("observation", hit),
        Step("action", 'web_search({"query": "b"})'),
        Step("observation", hit),
        Step("action", 'web_search({"query": "c"})'),
        Step("observation", "Error: Web search is unavailable right now."),
    ]


def test_search_links_are_deduplicated_and_errors_skipped() -> None:
    assert cli.search_links(_search_steps()) == [("Chair", "https://example.com/a")]


def test_saved_file_has_a_where_to_buy_section(tmp_path: Path) -> None:
    result = AgentResult("done", "Buy it.", _search_steps())
    path = cli.save_recommendation(result, "buy a chair", "chair", tmp_path)
    assert "[Chair](https://example.com/a)" in path.read_text()
