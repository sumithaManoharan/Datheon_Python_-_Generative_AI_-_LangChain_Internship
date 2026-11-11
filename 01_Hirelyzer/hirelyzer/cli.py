"""Command-line entry point: read documents, extract resumes, write JSON, report a summary."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

from hirelyzer.config import Settings, load_settings
from hirelyzer.errors import ConfigError, HirelyzerError
from hirelyzer.extractor import ExtractionResult, ResumeExtractor
from hirelyzer.llm import LLMClient
from hirelyzer.prompts import PromptBuilder
from hirelyzer.readers import READERS, prepare_text, read_document

logger = logging.getLogger(__name__)
NOISY_LOGGERS = ("httpx", "httpx2", "httpcore")  # libraries that log every HTTP request

EXIT_OK = 0
EXIT_FAILED = 1  # at least one document failed
EXIT_USAGE = 2  # bad configuration or nothing to process
DEFAULT_OUTPUT_DIR = "output"
STDIN_MARKER = "-"
TEXT_SOURCE_NAME = "text-input"

# A source is a display name plus a function that loads its text when called.
Source = tuple[str, Callable[[], str]]


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="hirelyzer", description="Extract structured JSON from resumes."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Files or folders (PDF, DOCX, TXT). Use '-' to read text from stdin.",
    )
    parser.add_argument("--text", help="Resume text passed directly on the command line.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Folder for the JSON results (default: %(default)s).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the arguments; pass a list in tests, or None to read sys.argv."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.inputs and args.text is None:
        parser.error("give at least one file, folder, '-' for stdin, or --text")
    return args


def expand_path(path: Path) -> list[Path]:
    """A folder becomes its supported files (sorted); anything else is kept as given."""
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in READERS)
    return [path]  # a missing file is reported later, for that file only


def collect_sources(args: argparse.Namespace, settings: Settings) -> list[Source]:
    """Turn the command-line inputs into (name, load_text) pairs without reading anything yet."""
    sources: list[Source] = []
    if args.text is not None:
        sources.append(
            (
                TEXT_SOURCE_NAME,
                partial(prepare_text, args.text, TEXT_SOURCE_NAME, settings.max_text_chars),
            )
        )
    for item in args.inputs:
        if item == STDIN_MARKER:
            sources.append(
                ("stdin", lambda: prepare_text(sys.stdin.read(), "stdin", settings.max_text_chars))
            )
            continue
        for path in expand_path(Path(item)):
            loader = partial(read_document, path, settings.max_file_bytes, settings.max_text_chars)
            sources.append((path.name, loader))
    return sources


def build_extractor(settings: Settings) -> ResumeExtractor:
    """Wire the real client and the prompt files into an extractor."""
    return ResumeExtractor(
        LLMClient.from_settings(settings),
        PromptBuilder.from_files(),
        settings.max_repair_attempts,
    )


def extract_one(
    extractor: ResumeExtractor, name: str, load_text: Callable[[], str]
) -> ExtractionResult:
    """Run one document. Input errors become a failed result, so a batch keeps going."""
    try:
        text = load_text()
    except HirelyzerError as exc:
        return ExtractionResult(source=name, errors=[str(exc)])
    return extractor.extract(text, source=name)


def write_result(result: ExtractionResult, output_dir: Path) -> Path:
    """Write one successful result as <source name>.json and return its path."""
    if result.data is None:
        raise ValueError("Cannot write a result that has no data.")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{result.source}.json"  # full name avoids a.pdf / a.docx clashes
    target.write_text(result.data.model_dump_json(indent=2), encoding="utf-8")
    return target


def format_summary(results: Sequence[ExtractionResult]) -> str:
    """Build the end-of-run summary text."""
    succeeded = sum(1 for r in results if r.status == "success")
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)
    average_ms = sum(r.latency_ms for r in results) / len(results) if results else 0.0
    lines = [
        f"Processed {len(results)} document(s): {succeeded} succeeded, "
        f"{len(results) - succeeded} failed",
        f"Average latency: {average_ms:.0f} ms | Total tokens: {total_tokens}",
    ]
    for r in results:
        if r.status == "failed":
            lines.append(f"  FAILED {r.source}: {r.errors[-1] if r.errors else 'unknown error'}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline and return the process exit code."""
    args = parse_args(argv)  # first, so --help works without an API key
    load_dotenv()  # reads .env into the environment; real environment variables win
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in NOISY_LOGGERS:  # hide per-request HTTP lines
        logging.getLogger(noisy).setLevel(logging.WARNING)  # hide per-request HTTP lines

    try:
        settings = load_settings()
        extractor = build_extractor(settings)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    sources = collect_sources(args, settings)
    if not sources:
        print("No supported documents found in the given inputs.", file=sys.stderr)
        return EXIT_USAGE

    results: list[ExtractionResult] = []
    for name, load_text in sources:
        result = extract_one(extractor, name, load_text)
        if result.status == "success":
            logger.info("Wrote %s", write_result(result, args.output_dir))
        results.append(result)

    print(format_summary(results))
    return EXIT_OK if all(r.status == "success" for r in results) else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
