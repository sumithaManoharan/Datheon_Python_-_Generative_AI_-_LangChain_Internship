"""Command-line entry point: run the agent and save the recommendation."""

import argparse
import json
import logging
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

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
from cartsense.tools import WISHLIST_DIR, UserPrompter, is_product_url

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


def _paired(steps: list[Step]) -> list[tuple[str, str]]:
    """Match each action with its observation in order (tool calls can run in parallel)."""
    waiting: list[str] = []
    pairs: list[tuple[str, str]] = []
    for step in steps:
        if step.kind == "action":
            waiting.append(step.text)
        elif step.kind == "observation" and waiting:
            pairs.append((waiting.pop(0), step.text))
    return pairs


def calculation_log(steps: list[Step]) -> str:
    """List every successful calculator call and its result, verbatim, from the step trace."""
    lines = [
        f"- `{action}` -> `{result}`"
        for action, result in _paired(steps)
        if action.startswith("calculator(") and not result.startswith(("Error:", "Duplicate call"))
    ]
    return "\n".join(lines)


LINK_PATTERN = re.compile(r'"title": "((?:[^"\\]|\\.)*)", "url": "((?:[^"\\]|\\.)*)"')
MAX_LINKS = 10
MAX_LINK_TITLE = 80


def _unescape(value: str) -> str:
    """Decode JSON string escapes such as \\u20b9; return the text unchanged on failure."""
    try:
        return json.loads(f'"{value}"')
    except ValueError:
        return value


def search_links(steps: list[Step]) -> list[tuple[str, str]]:
    """Every title and URL from successful web_search results, without duplicates."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for action, result in _paired(steps):
        if not action.startswith("web_search("):
            continue
        for title, url in LINK_PATTERN.findall(result):
            address = _unescape(url)
            if address not in seen:
                seen.add(address)
                links.append((_unescape(title), address))
    return links


INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\ufeff]|â€.")


def _clean_title(title: str) -> str:
    """Remove invisible characters and the garbled left-to-right mark from a link title."""
    return re.sub(r"[\[\]]", "", INVISIBLE.sub("", title)).strip()


def links_section(steps: list[Step], answer: str | None = None) -> str:
    """A markdown 'Where to buy' section of product pages found in the search results."""
    links = [(t, u) for t, u in search_links(steps) if is_product_url(u)]
    if answer is not None:
        links = [(t, u) for t, u in links if u in answer]
    links = links[:MAX_LINKS]
    if not links:
        return "\n## Where to buy\n\nNo direct product link found in the search results.\n"
    lines = []
    for title, url in links:
        label = _clean_title(title)[:MAX_LINK_TITLE] or url
        lines.append(f"- [{label}]({url})")
    heading = (
        "## Where to buy (product pages from the search results; stock and price may have changed)"
    )
    return f"\n{heading}\n\n" + "\n".join(lines) + "\n"


def _json_or_none(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from text, or None if it is not one."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _row(cells: list[str]) -> str:
    """One markdown table row."""
    return "| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |"


def _nums(*values: float) -> list[str]:
    """Format amounts with thousands separators and two decimals."""
    return [f"{value:,.2f}" for value in values]


def _ownership_table(label: str, data: dict[str, Any]) -> str:
    """One candidate's upfront and running costs as markdown tables."""
    upfront = data["upfront"]
    lines = [f"### {label.replace('|', '/')}", "", _row(["Upfront", "Amount"]), "|---|---:|"]
    lines.append(_row(["Product", *_nums(upfront["product"])]))
    for item in upfront["accessories"]:
        lines.append(_row([f"Must-buy: {item['name']}", *_nums(item["price"])]))
    lines.append(_row(["**Upfront total**", f"**{upfront['total']:,.2f}**"]))
    lines += [
        "",
        _row(["Running cost", "Per use", "Per month", "Per year"]),
        "|---|---:|---:|---:|",
    ]
    for row in data["running"]:
        values = _nums(row["per_use"], row["per_month"], row["per_year"])
        lines.append(_row([row["name"], *values]))
    total = data["total_running"]
    bold = [
        f"**{value}**" for value in _nums(total["per_use"], total["per_month"], total["per_year"])
    ]
    lines.append(_row(["**Total**", *bold]))
    first_year = f"{data['first_year_total']:,.2f}"
    lines += ["", f"First-year cost (upfront plus 12 months of running cost): **{first_year}**"]
    return "\n".join(lines)


def ownership_breakdown(steps: list[Step]) -> str:
    """Readable cost tables for every successful ownership_cost call, built from the trace."""
    tables: list[str] = []
    for action, result in _paired(steps):
        if not action.startswith("calculator("):
            continue
        call = _json_or_none(action[len("calculator(") : -1])
        data = _json_or_none(result)
        if call is not None and data is not None and "upfront" in data:
            tables.append(_ownership_table(str(call.get("label") or "Candidate"), data))
    if not tables:
        return ""
    heading = "## Cost breakdown (computed by the calculator, in your currency)"
    return f"\n{heading}\n\n" + "\n\n".join(tables) + "\n"


def save_recommendation(
    result: AgentResult, request: str, category: str, directory: Path = OUTPUT_DIR
) -> Path:
    """Write the recommendation and the calculator log to <directory>/<category>.md."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{category.replace(' ', '_')}.md"
    log = calculation_log(result.steps)
    appendix = ""
    if log:
        appendix = f"\n## Figures computed by the calculator tool (verbatim)\n\n{log}\n"
    path.write_text(
        f"# CartSense recommendation: {category}\n\nRequest: {request}\n\n"
        f"{result.final_answer}\n{links_section(result.steps, result.final_answer)}"
        f"{ownership_breakdown(result.steps)}{appendix}",
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
            wishlist_dir=wishlist_dir,
            prompter=prompter,
            region=settings.search_region,
            require_price_evidence=True,
            provider=settings.search_provider,
            tavily_api_key=settings.tavily_api_key,
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
        print("No file written; any file in output/ is from an earlier run.", file=sys.stderr)
        print(links_section(result.steps))
        return EXIT_FAILED

    category = guess_category(args.category or args.request)
    try:
        path = save_recommendation(result, args.request, category, output_dir)
    except OSError as exc:
        print(f"Could not save the recommendation: {exc}", file=sys.stderr)
        return EXIT_FAILED
    print(f"\nSaved: {output_dir.name}/{path.name}")
    print(links_section(result.steps, result.final_answer))
    return EXIT_OK


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
