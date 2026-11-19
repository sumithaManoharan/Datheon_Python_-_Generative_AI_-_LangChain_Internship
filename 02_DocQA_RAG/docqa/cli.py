"""Command-line interface: ingest a folder of documents, ask a question, show usage stats."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from dotenv import find_dotenv, load_dotenv

from docqa.answer import Answer
from docqa.config import Settings, load_settings
from docqa.errors import DocQAError
from docqa.logging_setup import configure_logging
from docqa.pipeline import IngestSummary, build_pipeline
from docqa.telemetry import Summary, TelemetryLog, summarize, telemetry_path

EXIT_OK = 0
EXIT_FAILURE = 1  # argparse itself exits with 2 on a usage error


class Pipeline(Protocol):
    """What the CLI needs from a pipeline."""

    def ingest(self, source_dir: Path) -> IngestSummary: ...

    def ask(self, question: str) -> Answer: ...


PipelineFactory = Callable[[Settings], Pipeline]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments (pass a list in tests)."""
    parser = argparse.ArgumentParser(prog="docqa", description="Document Q&A with citations.")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="embed the documents in a folder")
    ingest.add_argument("--source-dir", type=Path, default=None, help="default: DOCQA_SOURCE_DIR")
    ask = commands.add_parser("ask", help="answer a question from the stored documents")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", dest="as_json", help="machine-readable output")
    commands.add_parser("stats", help="show timing and token usage of past questions")
    return parser.parse_args(argv)


def format_summary(summary: IngestSummary) -> str:
    """Human-readable ingest result, listing every failed file."""
    lines = [
        f"Ingest complete: {summary.embedded} embedded, {summary.skipped} unchanged, "
        f"{len(summary.failures)} failed; {summary.total_chunks} chunks in the store."
    ]
    if summary.failures:
        lines.append("Failed:")
        lines.extend(f"  - {item}" for item in summary.failures)
    return "\n".join(lines) + "\n"


def format_answer(answer: Answer) -> str:
    """The answer text followed by its numbered sources."""
    lines = [answer.text]
    if answer.citations:
        lines.extend(["", "Sources:"])
        for citation in answer.citations:
            where = citation.location()
            suffix = f" | {where}" if where else ""
            lines.append(f"[{citation.number}] {citation.source}{suffix}")
            lines.append(f'    "{citation.snippet}"')
    return "\n".join(lines) + "\n"


def format_stats(summary: Summary | None) -> str:
    """Timing and token usage over all recorded questions."""
    if summary is None:
        return "No questions recorded yet.\n"
    return (
        f"Questions: {summary.questions} ({summary.refused} refused)\n"
        f"Latency (s): retrieve {summary.mean_retrieve:.2f}"
        f" | generate {summary.mean_generate:.2f}"
        f" | total {summary.mean_total:.2f} (p95 {summary.p95_total:.2f})\n"
        f"Tokens: {summary.prompt_tokens:,} prompt, {summary.completion_tokens:,} completion\n"
    )


def _run_ingest(pipeline: Pipeline, source_dir: Path) -> int:
    summary = pipeline.ingest(source_dir)
    sys.stdout.write(format_summary(summary))
    return EXIT_FAILURE if summary.failures else EXIT_OK


def _run_ask(pipeline: Pipeline, question: str, as_json: bool) -> int:
    answer = pipeline.ask(question)
    if as_json:
        sys.stdout.write(json.dumps(asdict(answer), indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(format_answer(answer))
    return EXIT_OK


def _run_stats(settings: Settings) -> int:
    log = TelemetryLog(telemetry_path(settings))
    sys.stdout.write(format_stats(summarize(log.read())))
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    pipeline_factory: PipelineFactory = build_pipeline,
) -> int:
    """Run one command and return the process exit code.

    Tests pass `settings` and a fake `pipeline_factory`, so no network or `.env` is used.
    """
    args = parse_args(argv)
    try:
        if settings is None:
            load_dotenv(find_dotenv(usecwd=True))
            settings = load_settings()
        configure_logging(settings.log_level)
        if args.command == "stats":
            return _run_stats(settings)
        pipeline = pipeline_factory(settings)
        if args.command == "ingest":
            return _run_ingest(pipeline, args.source_dir or settings.source_dir)
        return _run_ask(pipeline, args.question, args.as_json)
    except DocQAError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return EXIT_FAILURE
