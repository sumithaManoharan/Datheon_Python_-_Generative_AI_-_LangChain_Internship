"""Command-line entry point: run the agent and save the recommendation."""

import argparse
import logging
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from langchain_core.language_models.chat_models import BaseChatModel

from cartsense.agent import (
    AgentError,
    AgentResult,
    Requirements,
    Step,
    build_model,
    build_requirements,
    build_tools,
    format_step,
    load_system_prompt,
    mode_note,
    run_agent,
)
from cartsense.config import PROJECT_ROOT, Config, ConfigError, load_config
from cartsense.models import CATEGORY_PATTERN, clean_category
from cartsense.tools import WISHLIST_DIR, UserPrompter

OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
MAX_DISPLAY_CHARS = 400
FALLBACK_CATEGORY = "recommendation"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2
EXIT_INTERRUPTED = 130

_REQUEST_PATTERN = re.compile(
    r"\b(?:buy|purchase|get|order)\s+(?:(?:a|an|the|some|new)\s+)*(.+)", re.IGNORECASE
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="cartsense", description="A purchase-decision agent for the console."
    )
    parser.add_argument("request", nargs="+", help='e.g. "I want to buy a phone"')
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="never ask questions; use the saved wishlist or stated defaults",
    )
    parser.add_argument("--category", help="category name for the output file (default: guessed)")
    args = parser.parse_args(argv)
    args.request = " ".join(args.request)
    return args


def is_interactive(no_interactive: bool, stdin: TextIO | None = None) -> bool:
    """Interactive only if not disabled by the flag and stdin is a terminal."""
    stream = stdin if stdin is not None else sys.stdin
    return not no_interactive and stream.isatty()


def guess_category(text: str) -> str:
    """Turn a request like 'I want to buy a phone' into a safe category name."""
    match = _REQUEST_PATTERN.search(text)
    raw = match.group(1) if match else text
    cleaned = re.sub(r"[^a-z0-9 _-]", "", clean_category(raw))
    cleaned = clean_category(cleaned)[:40].strip(" _-")
    return cleaned if re.fullmatch(CATEGORY_PATTERN, cleaned) else FALLBACK_CATEGORY


def save_recommendation(
    result: AgentResult, request: str, category: str, directory: Path = OUTPUT_DIR
) -> Path:
    """Write the final recommendation to <directory>/<category>.md and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{category.replace(' ', '_')}.md"
    path.write_text(
        f"# CartSense recommendation: {category}\n\nRequest: {request}\n\n{result.final_answer}\n",
        encoding="utf-8",
    )
    return path


def show_step(step: Step) -> None:
    """Print one step of the reasoning chain as it happens."""
    line = format_step(step)
    if step.kind == "observation" and len(line) > MAX_DISPLAY_CHARS:
        line = line[:MAX_DISPLAY_CHARS] + " ...[truncated]"
    print(line, flush=True)


def configure_logging(log_dir: Path = LOG_DIR) -> None:
    """Send INFO logs (telemetry) to a file and warnings to the console."""
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    handlers: list[logging.Handler] = [console]
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "cartsense.log", encoding="utf-8"))
    except OSError:
        pass  # logging to a file is optional
    logging.basicConfig(level=logging.INFO, handlers=handlers, format=LOG_FORMAT, force=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    model: BaseChatModel | None = None,
    config: Config | None = None,
    input_fn: Callable[[str], str] = input,
    stdin: TextIO | None = None,
    wishlist_dir: Path = WISHLIST_DIR,
    output_dir: Path = OUTPUT_DIR,
    requirements: Requirements | None = None,
) -> int:
    """Run one request end to end and return the process exit code."""
    args = parse_args(argv)
    interactive = is_interactive(args.no_interactive, stdin)
    try:
        settings = config if config is not None else load_config()
        chat_model = model if model is not None else build_model(settings)
        prompt = f"{load_system_prompt()}\n\n{mode_note(interactive)}"
        prompter = UserPrompter(input_fn) if interactive else None
        tools = build_tools(
            wishlist_dir=wishlist_dir, prompter=prompter, region=settings.search_region
        )
        result = run_agent(
            chat_model,
            settings,
            args.request,
            tools=tools,
            system_prompt=prompt,
            on_step=show_step,
            requirements=requirements or build_requirements(interactive),
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return EXIT_INTERRUPTED

    if result.status != "done":
        print(f"\n{result.final_answer}", file=sys.stderr)
        return EXIT_FAILED

    category = guess_category(args.category or args.request)
    try:
        path = save_recommendation(result, args.request, category, output_dir)
    except OSError as exc:
        print(f"Could not save the recommendation: {exc}", file=sys.stderr)
        return EXIT_FAILED
    print(f"\nSaved: {output_dir.name}/{path.name}")
    return EXIT_OK


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
