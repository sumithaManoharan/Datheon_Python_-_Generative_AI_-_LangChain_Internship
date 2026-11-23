"""Tests for the command-line entry point. No network, files only in tmp_path."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from test_agent import CONFIG, ScriptedModel

from cartsense import cli
from cartsense.agent import AgentResult
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
