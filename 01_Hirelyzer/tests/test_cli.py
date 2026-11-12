"""Tests for the command-line entry point, using a fake LLM and temporary folders."""

import io
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from hirelyzer import cli
from hirelyzer.errors import LLMCallError
from hirelyzer.extractor import ExtractionResult, ResumeExtractor

NON_RESUME = '{"is_resume": false}'


@pytest.fixture
def install_llm(
    monkeypatch: pytest.MonkeyPatch, prompt_builder: Any, make_llm: Any
) -> Callable[[Sequence[str | Exception]], Any]:
    """Install a fake API key and a scripted fake LLM into the CLI."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)  # tests must never read a real .env
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def install(script: Sequence[str | Exception]) -> Any:
        llm = make_llm(script)
        monkeypatch.setattr(
            cli,
            "build_extractor",
            lambda settings, recorder: ResumeExtractor(llm, prompt_builder, 2),
        )
        return llm

    return install


def run_cli(tmp_path: Path, *args: str) -> int:
    """Run main() with output and telemetry redirected into the temporary folder."""
    output = ["--output-dir", str(tmp_path / "out"), "--telemetry-log", str(tmp_path / "t.jsonl")]
    return cli.main([*args, *output])


def test_text_input_succeeds_and_writes_json(tmp_path: Path, install_llm: Any) -> None:
    install_llm([NON_RESUME])

    assert run_cli(tmp_path, "--text", "hello") == 0

    data = json.loads((tmp_path / "out" / "text-input.json").read_text(encoding="utf-8"))
    assert data["is_resume"] is False


def test_file_input_writes_json_named_after_the_file(tmp_path: Path, install_llm: Any) -> None:
    install_llm([NON_RESUME])
    document = tmp_path / "good.txt"
    document.write_text("some text", encoding="utf-8")

    assert run_cli(tmp_path, str(document)) == 0

    assert (tmp_path / "out" / "good.txt.json").exists()


def test_folder_input_processes_only_supported_files(tmp_path: Path, install_llm: Any) -> None:
    install_llm([NON_RESUME, NON_RESUME])
    folder = tmp_path / "docs"
    folder.mkdir()
    for name in ("a.txt", "b.txt", "c.xyz"):
        (folder / name).write_text("some text", encoding="utf-8")

    assert run_cli(tmp_path, str(folder)) == 0

    written = sorted(path.name for path in (tmp_path / "out").glob("*.json"))
    assert written == ["a.txt.json", "b.txt.json"]


def test_stdin_input(tmp_path: Path, install_llm: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    install_llm([NON_RESUME])
    monkeypatch.setattr("sys.stdin", io.StringIO("some text"))

    assert run_cli(tmp_path, "-") == 0

    assert (tmp_path / "out" / "stdin.json").exists()


def test_one_bad_file_does_not_stop_the_batch(tmp_path: Path, install_llm: Any) -> None:
    install_llm([NON_RESUME])  # only the good file reaches the model
    good = tmp_path / "good.txt"
    good.write_text("some text", encoding="utf-8")

    exit_code = run_cli(tmp_path, str(tmp_path / "missing.txt"), str(good))

    assert exit_code == 1
    assert (tmp_path / "out" / "good.txt.json").exists()


def test_llm_failure_gives_exit_code_1_and_no_output(tmp_path: Path, install_llm: Any) -> None:
    install_llm([LLMCallError("boom")])

    assert run_cli(tmp_path, "--text", "hello") == 1

    assert not (tmp_path / "out").exists()


def test_missing_api_key_exits_with_code_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert run_cli(tmp_path, "--text", "hello") == 2

    assert "Configuration error" in capsys.readouterr().err


def test_folder_without_supported_files_exits_with_code_2(
    tmp_path: Path, install_llm: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    install_llm([])
    empty = tmp_path / "empty"
    empty.mkdir()

    assert run_cli(tmp_path, str(empty)) == 2

    assert "No supported documents" in capsys.readouterr().err


def test_parse_args_requires_an_input() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_args([])

    assert excinfo.value.code == 2


def test_parse_args_accepts_text_only() -> None:
    assert cli.parse_args(["--text", "hello"]).text == "hello"


def test_expand_path_filters_and_sorts_a_folder(tmp_path: Path) -> None:
    for name in ("b.txt", "a.pdf", "c.xyz"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    assert [path.name for path in cli.expand_path(tmp_path)] == ["a.pdf", "b.txt"]
    assert cli.expand_path(tmp_path / "nope.pdf") == [tmp_path / "nope.pdf"]


def test_format_summary_reports_counts_tokens_and_failures() -> None:
    ok = ExtractionResult(
        source="a.txt", status="success", prompt_tokens=10, completion_tokens=5, latency_ms=100.0
    )
    bad = ExtractionResult(source="b.txt", errors=["File not found"])

    summary = cli.format_summary([ok, bad])

    assert "1 succeeded, 1 failed" in summary
    assert "FAILED b.txt: File not found" in summary
    assert "Total tokens: 15" in summary


def test_write_result_refuses_a_result_without_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        cli.write_result(ExtractionResult(source="x"), tmp_path)
