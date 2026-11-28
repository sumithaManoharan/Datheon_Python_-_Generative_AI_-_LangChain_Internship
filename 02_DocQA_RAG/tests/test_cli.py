"""Tests for docqa.cli."""

import json
from pathlib import Path

import pytest
from docqa.answer import REFUSAL_MESSAGE, Answer, Citation
from docqa.cli import main, parse_args
from docqa.config import load_settings
from docqa.errors import ConfigError, InvalidQuestionError
from docqa.pipeline import IngestSummary
from docqa.telemetry import QueryRecord, TelemetryLog

SETTINGS = load_settings({})
ANSWER = Answer(
    text="Founded in 2015 [1].",
    citations=(
        Citation(
            number=1,
            source="about.md",
            snippet="Insurellm was founded in 2015.",
            section="About Insurellm",
            start_line=1,
            end_line=5,
        ),
    ),
    refused=False,
    prompt_tokens=10,
    completion_tokens=5,
)


class FakePipeline:
    """Records calls and returns canned results, or raises a scripted error."""

    def __init__(self, summary=None, answer=None, error=None):
        self.summary = summary
        self.answer = answer
        self.error = error
        self.ingested: list[Path] = []
        self.questions: list[str] = []
        self.memory = None
        self.memory_sizes: list[int] = []
        self.fail_on: set[str] = set()

    def ingest(self, source_dir):
        self.ingested.append(source_dir)
        if self.error:
            raise self.error
        return self.summary

    def ask(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.answer

    def chat(self, question, memory):
        self.questions.append(question)
        self.memory = memory
        self.memory_sizes.append(len(memory))
        if question in self.fail_on:
            raise InvalidQuestionError("boom")
        memory.add(question, self.answer.text)
        return self.answer


def _run(argv, pipeline) -> int:
    return main(argv, settings=SETTINGS, pipeline_factory=lambda _settings: pipeline)


# ---------- ingest ----------


def test_ingest_success_exits_zero_and_reports_counts(capsys):
    pipeline = FakePipeline(summary=IngestSummary(3, 1, (), 40))
    assert _run(["ingest"], pipeline) == 0
    out = capsys.readouterr().out
    assert "3 embedded" in out
    assert "1 unchanged" in out
    assert "40 chunks" in out
    assert pipeline.ingested == [SETTINGS.source_dir]


def test_ingest_source_dir_option_overrides_the_setting():
    pipeline = FakePipeline(summary=IngestSummary(1, 0, (), 5))
    _run(["ingest", "--source-dir", "docs"], pipeline)
    assert pipeline.ingested == [Path("docs")]


def test_ingest_with_a_failed_file_exits_one_and_lists_it(capsys):
    pipeline = FakePipeline(summary=IngestSummary(1, 0, ("bad.pdf: corrupt",), 10))
    assert _run(["ingest"], pipeline) == 1
    out = capsys.readouterr().out
    assert "1 failed" in out
    assert "bad.pdf: corrupt" in out


# ---------- ask ----------


def test_ask_prints_the_answer_and_its_sources(capsys):
    pipeline = FakePipeline(answer=ANSWER)
    assert _run(["ask", "When was it founded?"], pipeline) == 0
    out = capsys.readouterr().out
    assert "Founded in 2015 [1]." in out
    assert "Sources:" in out
    assert "[1] about.md | About Insurellm, lines 1-5" in out
    assert '"Insurellm was founded in 2015."' in out
    assert pipeline.questions == ["When was it founded?"]


def test_a_refusal_is_a_success_with_no_sources(capsys):
    pipeline = FakePipeline(answer=Answer(text=REFUSAL_MESSAGE, citations=(), refused=True))
    assert _run(["ask", "Capital of Australia?"], pipeline) == 0
    out = capsys.readouterr().out
    assert REFUSAL_MESSAGE in out
    assert "Sources" not in out


def test_ask_json_output_is_machine_readable(capsys):
    assert _run(["ask", "--json", "When?"], FakePipeline(answer=ANSWER)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["answer" if "answer" in data else "text"] == "Founded in 2015 [1]."
    assert data["refused"] is False
    assert data["citations"][0]["section"] == "About Insurellm"
    assert (data["prompt_tokens"], data["completion_tokens"]) == (10, 5)


# ---------- errors and usage ----------


def test_a_pipeline_error_exits_one_with_the_message_on_stderr(capsys):
    pipeline = FakePipeline(error=InvalidQuestionError("the question is empty"))
    assert _run(["ask", "x"], pipeline) == 1
    assert "the question is empty" in capsys.readouterr().err


def test_a_factory_error_exits_one(capsys):
    def factory(_settings):
        raise ConfigError("OPENROUTER_API_KEY is not set")

    assert main(["ask", "x"], settings=SETTINGS, pipeline_factory=factory) == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_a_missing_command_is_a_usage_error():
    with pytest.raises(SystemExit) as info:
        main([], settings=SETTINGS)
    assert info.value.code == 2


def test_parse_args_reads_the_question():
    args = parse_args(["ask", "hello", "--json"])
    assert (args.command, args.question, args.as_json) == ("ask", "hello", True)


# ---------- stats ----------


def _stats_settings(tmp_path: Path):
    return load_settings({"DOCQA_STORE_DIR": str(tmp_path / "chroma")})


def _no_pipeline(_settings):
    raise AssertionError("stats must not build a pipeline")


def test_stats_with_no_records_says_so(tmp_path: Path, capsys):
    code = main(["stats"], settings=_stats_settings(tmp_path), pipeline_factory=_no_pipeline)
    assert code == 0
    assert "No questions recorded yet." in capsys.readouterr().out


def test_stats_summarises_the_log_without_building_a_pipeline(tmp_path: Path, capsys):
    settings = _stats_settings(tmp_path)
    log = TelemetryLog(tmp_path / "telemetry.jsonl")
    base = {"timestamp": "2026-11-19T10:00:00+00:00", "question_chars": 10, "retrieved": 10}
    log.record(
        QueryRecord(
            **base,
            refused=False,
            retrieve_seconds=1.0,
            generate_seconds=2.0,
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    log.record(
        QueryRecord(
            **base,
            refused=True,
            retrieve_seconds=0.5,
            generate_seconds=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )
    )
    assert main(["stats"], settings=settings, pipeline_factory=_no_pipeline) == 0
    out = capsys.readouterr().out
    assert "Questions: 2 (1 refused)" in out
    assert "retrieve 0.75" in out
    assert "total 1.75 (p95 3.00)" in out
    assert "100 prompt, 20 completion" in out


# ---------- chat ----------


def _inputs(*lines):
    remaining = iter(lines)

    def read(_prompt):
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    return read


def _chat(pipeline, *lines, turns=None):
    argv = ["chat"] + ([] if turns is None else ["--turns", str(turns)])
    return main(
        argv,
        settings=SETTINGS,
        pipeline_factory=lambda _settings: pipeline,
        input_fn=_inputs(*lines),
    )


def test_chat_answers_each_line_until_the_input_ends(capsys):
    pipeline = FakePipeline(answer=ANSWER)
    assert _chat(pipeline, "Who founded it?", "And when?") == 0
    assert capsys.readouterr().out.count("Founded in 2015 [1].") == 2
    assert pipeline.questions == ["Who founded it?", "And when?"]


def test_chat_ignores_blank_lines():
    pipeline = FakePipeline(answer=ANSWER)
    _chat(pipeline, "", "   ", "Who?")
    assert pipeline.questions == ["Who?"]


def test_chat_clear_forgets_the_earlier_turns(capsys):
    pipeline = FakePipeline(answer=ANSWER)
    _chat(pipeline, "one?", "two?", "/clear", "three?")
    assert pipeline.memory_sizes == [0, 1, 0]
    assert "Conversation cleared." in capsys.readouterr().out


def test_chat_exit_stops_without_reading_more():
    pipeline = FakePipeline(answer=ANSWER)
    assert _chat(pipeline, "one?", "/exit", "two?") == 0
    assert pipeline.questions == ["one?"]


def test_chat_an_error_is_reported_and_the_session_continues(capsys):
    pipeline = FakePipeline(answer=ANSWER)
    pipeline.fail_on = {"bad?"}
    assert _chat(pipeline, "bad?", "good?") == 0
    captured = capsys.readouterr()
    assert "boom" in captured.err
    assert "Founded in 2015 [1]." in captured.out


def test_chat_turns_option_sets_the_memory_size():
    pipeline = FakePipeline(answer=ANSWER)
    _chat(pipeline, "one?", turns=3)
    assert pipeline.memory.max_turns == 3


def test_chat_defaults_to_six_turns_and_rejects_a_bad_value():
    assert parse_args(["chat"]).turns == 6
    with pytest.raises(SystemExit) as info:
        parse_args(["chat", "--turns", "0"])
    assert info.value.code == 2
