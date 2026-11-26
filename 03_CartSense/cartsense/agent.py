"""The LangChain tool-calling agent and its step trace."""

import functools
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from cartsense.config import PROJECT_ROOT, Config
from cartsense.scoring import running_cost, tco, value_score
from cartsense.tools import (
    WISHLIST_DIR,
    ToolError,
    UserPrompter,
    calculator,
    is_product_url,
    read_wishlist,
    save_wishlist,
    web_search,
)

logger = logging.getLogger(__name__)

PROMPT_PATH = PROJECT_ROOT / "prompts" / "system.md"
MAX_REQUEST_LENGTH = 500
MAX_SEARCHES = 12
STORE_SEARCHES = 4
GENERAL_LIMIT_NOTE = (
    "General search limit reached. Use the remaining store searches (store set to "
    "amazon or flipkart) to find each chosen model's product page."
)
SEARCH_LIMIT_NOTE = (
    "Search limit reached. Stop searching. Call calculator (running_cost and "
    "value_score) for each candidate using what you have, then write the final answer."
)
MAX_OBSERVATION_CHARS = 4000
MODEL_MAX_RETRIES = 3
MODEL_TIMEOUT_SECONDS = 60
# One model call plus one tool call is about two graph steps.
GRAPH_STEPS_PER_TURN = 2
MAX_STEPS_MESSAGE = "Stopped: the step limit was reached before a final answer."
FINAL_NUDGE = "Write your final answer now, using the results above. Do not call any tool."
STEP_LABELS = {
    "thought": "Thought",
    "action": "Action",
    "observation": "Observation",
    "final": "Final answer",
}
INTERACTIVE_NOTE = "Mode: interactive. ask_user and save_wishlist are available."
NON_INTERACTIVE_NOTE = (
    "Mode: non-interactive. ask_user and save_wishlist are NOT available. "
    "Do not ask the buyer anything. Use the saved wishlist if there is one; otherwise use "
    "sensible defaults and state each one as an assumption."
)

SCORING_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "tco": tco,
    "value_score": value_score,
    "running_cost": running_cost,
}


class AgentError(Exception):
    """The agent could not run or could not reach the model."""


@dataclass(frozen=True)
class Step:
    """One line of the reasoning chain."""

    kind: str  # thought, action, observation or final
    text: str


@dataclass(frozen=True)
class AgentResult:
    """Outcome of one agent run."""

    status: str  # done, max_steps or failed
    final_answer: str
    steps: list[Step]


def format_step(step: Step) -> str:
    """Render a step as one console line, e.g. 'Action: calculator({...})'."""
    return f"{STEP_LABELS.get(step.kind, step.kind)}: {step.text}"


def mode_note(interactive: bool) -> str:
    """The line appended to the system prompt that tells the agent which mode it is in."""
    return INTERACTIVE_NOTE if interactive else NON_INTERACTIVE_NOTE


# --- tools ------------------------------------------------------------------


class ReadWishlistArgs(BaseModel):
    """Arguments for read_wishlist."""

    category: str = Field(description="Product category, e.g. 'washing machine'.")


class AskUserArgs(BaseModel):
    """Arguments for ask_user."""

    question: str = Field(description="One short question for the buyer.")


class SaveWishlistArgs(BaseModel):
    """Arguments for save_wishlist."""

    data: dict[str, Any] = Field(
        description=(
            "Wishlist fields: category (text), budget (number), currency (e.g. INR), "
            "years_of_use (number), "
            "warranty_years (number), priorities (1 or 2 short texts), dealbreakers "
            "(list of short texts), specs (object of short texts)."
        )
    )


class WebSearchArgs(BaseModel):
    """Arguments for web_search."""

    query: str = Field(description="Search query for prices, reviews or specs.")
    max_results: int = Field(default=5, description="How many results to return (1 to 10).")
    store: str | None = Field(
        default=None, description="Optional: 'amazon' or 'flipkart' to search only that store."
    )


class CalculatorArgs(BaseModel):
    """Arguments for calculator: an expression, or a scoring function with arguments."""

    expression: str | None = Field(
        default=None, description="Arithmetic using numbers and + - * / // % ** ( ) only."
    )
    function: str | None = Field(
        default=None, description="'tco', 'value_score' or 'running_cost'."
    )
    arguments: dict[str, Any] | None = Field(
        default=None, description="Keyword arguments for the chosen function."
    )
    label: str | None = Field(
        default=None,
        description="Candidate model name this calculation is for.",
    )


def _guard(func: Callable[..., str]) -> Callable[..., str]:
    """Turn ToolError into an 'Error: ...' observation and cap the output length."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return func(*args, **kwargs)[:MAX_OBSERVATION_CHARS]
        except ToolError as exc:
            return f"Error: {exc}"

    return wrapper


DUPLICATE_NOTE = (
    "Duplicate call: you already have this result. "
    "Do not repeat it; use the earlier result or move on."
)


def _remember(func: Callable[..., str]) -> Callable[..., str]:
    """Wrap a tool so an identical repeat call returns a short note instead of running.

    Each call to _remember has its own cache, so every build_tools() starts fresh.
    Errors are not remembered, so a failed call can be retried.
    """
    cache: set[str] = set()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        key = json.dumps([args, kwargs], sort_keys=True, default=str)
        if key in cache:
            return DUPLICATE_NOTE
        result = func(*args, **kwargs)
        if not result.startswith("Error:"):
            cache.add(key)
        return result

    return wrapper


@_guard
def _calculator_tool(
    expression: str | None = None,
    function: str | None = None,
    arguments: dict[str, Any] | None = None,
    label: str | None = None,
) -> str:
    """Run a plain expression, or one of the pure scoring functions."""
    if function is None:
        return str(calculator(expression or ""))
    scorer = SCORING_FUNCTIONS.get(function)
    if scorer is None:
        raise ToolError(f"Unknown function {function!r}; use one of {sorted(SCORING_FUNCTIONS)}.")
    try:
        result = scorer(**(arguments or {}))
        return json.dumps(result) if isinstance(result, dict) else str(result)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"Bad arguments for {function}: {exc}") from None


def _make_web_search_tool(
    region: str | None,
    max_searches: int,
    provider: str = "ddgs",
    tavily_api_key: str = "",
) -> Callable[..., str]:
    """Build the web_search handler, bound to the region and the search provider."""
    used = {"general": 0, "store": 0}
    general_cap = max(max_searches - STORE_SEARCHES, 1)

    @_guard
    def web_search_tool(query: str, max_results: int = 5, store: str | None = None) -> str:
        """Return search hits as a JSON string."""
        kind = "store" if store else "general"
        cap = STORE_SEARCHES if store else general_cap
        if used[kind] >= cap:
            raise ToolError(SEARCH_LIMIT_NOTE if store else GENERAL_LIMIT_NOTE)
        used[kind] += 1
        extra: dict[str, Any] = {"region": region} if region else {}
        if provider == "tavily":
            extra.update(provider="tavily", tavily_api_key=tavily_api_key, store=store)
        hits = web_search(query, max_results, **extra)
        return json.dumps(hits, ensure_ascii=False)

    return web_search_tool


PRICE_NOT_FOUND = (
    "Error: price {price:g} was not found in any search result. Use only prices that appear "
    "in search results. If a candidate has no price, drop it and use fewer candidates."
)


def _indian_grouping(number: int) -> str:
    """Format a whole number with Indian digit grouping, e.g. 125000 -> 1,25,000."""
    digits = str(number)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join([*groups, tail])


def _price_in_text(price: float, text: str) -> bool:
    """True if the price appears in the text as a whole number, plain or with separators."""
    whole = int(round(price))
    forms = {str(whole), f"{whole:,}", _indian_grouping(whole)}
    return any(re.search(rf"(?<![\d,]){re.escape(form)}(?!\d)", text) for form in forms)


def _record_results(func: Callable[..., str], seen: list[str]) -> Callable[..., str]:
    """Wrap web_search so every successful result is remembered for price checks."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        result = func(*args, **kwargs)
        if not result.startswith(("Error:", "Duplicate call")):
            seen.append(result)
        return result

    return wrapper


def _price_checked(func: Callable[..., str], seen: list[str], enabled: bool) -> Callable[..., str]:
    """Wrap calculator so value_score only accepts prices seen in search results."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        if enabled and kwargs.get("function") == "value_score":
            inner = kwargs.get("arguments")
            price = inner.get("price") if isinstance(inner, dict) else None
            if isinstance(price, int | float) and not isinstance(price, bool):
                if not _price_in_text(price, "\n".join(seen)):
                    return PRICE_NOT_FOUND.format(price=price)
        return func(*args, **kwargs)

    return wrapper


MIN_CHECKED_PRICE = 1000
NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
STANDALONE_NUMBER = re.compile(r"(?<![\w,.])\d[\d,]*(?:\.\d+)?(?![\w])")
UNICODE_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}")
QUERY_NUMBER_NOTE = (
    "Error: the number {number} in your query is not in the wishlist or any search result. "
    "Do not guess prices. Search by product name and read prices from the results."
)


def _number(token: str) -> float | None:
    """Parse a number such as '2,999' or '87.5'; None if it is not one."""
    try:
        return round(float(token.replace(",", "")), 2)
    except ValueError:
        return None


def _numbers(text: str) -> set[float]:
    """Every number in a text, ignoring commas and JSON unicode escapes."""
    clean = UNICODE_ESCAPE.sub(" ", text)
    values = (_number(token) for token in NUMBER_PATTERN.findall(clean))
    return {value for value in values if value is not None}


def _is_year(value: float) -> bool:
    """True for whole numbers that look like a year, which are fine in a query."""
    return 1990 <= value <= 2100 and value == int(value)


def _query_checked(
    func: Callable[..., str], seen: list[str], known: list[str]
) -> Callable[..., str]:
    """Wrap web_search so a query cannot contain a large number the model made up."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        query = kwargs.get("query", args[0] if args else "")
        if isinstance(query, str):
            allowed = _numbers("\n".join([*seen, *known]))
            for token in STANDALONE_NUMBER.findall(query):
                value = _number(token)
                if (
                    value is not None
                    and value >= MIN_CHECKED_PRICE
                    and not _is_year(value)
                    and value not in allowed
                ):
                    return QUERY_NUMBER_NOTE.format(number=token.strip(","))
        return func(*args, **kwargs)

    return wrapper


CONFIRM_NOTE = (
    "Error: a saved wishlist exists. First ask the buyer with ask_user whether to reuse "
    "it or update it, then continue."
)


def _confirm_first(
    func: Callable[..., str], state: dict[str, bool], prompter: UserPrompter | None
) -> Callable[..., str]:
    """Block a tool until the buyer was asked about reusing a saved wishlist."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        if state["needed"] and prompter is not None and prompter.asked == 0:
            return CONFIRM_NOTE
        return func(*args, **kwargs)

    return wrapper


def build_tools(
    *,
    wishlist_dir: Path = WISHLIST_DIR,
    prompter: UserPrompter | None = None,
    region: str | None = None,
    max_searches: int = MAX_SEARCHES,
    require_price_evidence: bool = False,
    provider: str = "ddgs",
    tavily_api_key: str = "",
) -> list[BaseTool]:
    """Build the LangChain tools.

    ask_user and save_wishlist exist only when a prompter is given (interactive mode).
    """
    seen_text: list[str] = []  # successful web_search results
    known_text: list[str] = []  # wishlist contents and the buyer's answers
    confirm = {"needed": False}

    @_guard
    def read_wishlist_tool(category: str) -> str:
        """Return the saved wishlist as JSON, or a plain message when there is none."""
        wishlist = read_wishlist(category, wishlist_dir)
        if wishlist is None:
            return f"No saved wishlist for {category!r}."
        text = wishlist.model_dump_json()
        known_text.append(text)
        confirm["needed"] = True
        return text

    tools: list[BaseTool] = [
        StructuredTool.from_function(
            func=read_wishlist_tool,
            name="read_wishlist",
            description="Read the saved wishlist for a product category, if one exists.",
            args_schema=ReadWishlistArgs,
        )
    ]

    if prompter is not None:

        @_guard
        def ask_user_tool(question: str) -> str:
            """Ask the buyer one question and return the answer."""
            answer = prompter.ask(question)
            known_text.append(answer)
            return answer

        @_guard
        def save_wishlist_tool(data: dict[str, Any]) -> str:
            """Validate and save the wishlist; return a confirmation."""
            path = save_wishlist(data, wishlist_dir)
            prompter.close()
            known_text.append(json.dumps(data))
            return f"Saved wishlist to {path.name}."

        tools.append(
            StructuredTool.from_function(
                func=ask_user_tool,
                name="ask_user",
                description="Ask the buyer one short question. At most 10 questions per run.",
                args_schema=AskUserArgs,
            )
        )
        tools.append(
            StructuredTool.from_function(
                func=save_wishlist_tool,
                name="save_wishlist",
                description="Save the buyer's answers as their wishlist for this category.",
                args_schema=SaveWishlistArgs,
            )
        )

    search = _remember(_make_web_search_tool(region, max_searches, provider, tavily_api_key))
    search = _query_checked(_record_results(search, seen_text), seen_text, known_text)
    search = _confirm_first(search, confirm, prompter)
    tools.append(
        StructuredTool.from_function(
            func=search,
            name="web_search",
            description=(
                "Search the web for current prices, reviews and specs. Optional store "
                "('amazon' or 'flipkart') limits results to that store (Tavily only)."
            ),
            args_schema=WebSearchArgs,
        )
    )
    tools.append(
        StructuredTool.from_function(
            func=_remember(_price_checked(_calculator_tool, seen_text, require_price_evidence)),
            name="calculator",
            description=(
                "Do arithmetic. Pass 'expression', or pass 'function' ('tco', 'value_score' "
                "or 'running_cost') with its 'arguments' object."
            ),
            args_schema=CalculatorArgs,
        )
    )
    return tools


# --- model and prompt -------------------------------------------------------


def build_model(config: Config) -> ChatOpenAI:
    """Create the chat model; ChatOpenAI retries transient API errors itself."""
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_retries=MODEL_MAX_RETRIES,
        timeout=MODEL_TIMEOUT_SECONDS,
    )


def load_system_prompt(path: Path = PROMPT_PATH) -> str:
    """Read the system prompt from disk and make sure it is not empty."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AgentError(f"Cannot read the system prompt at {path.name}.") from exc
    if not text:
        raise AgentError(f"The system prompt {path.name} is empty.")
    return text


# --- the run ----------------------------------------------------------------


def _text(content: Any) -> str:
    """Flatten message content (a string or a list of blocks) into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str | dict)
        ]
        return "".join(parts).strip()
    return ""


def _request_final_answer(model: BaseChatModel, history: list[Any]) -> str:
    """Ask the model once more for a final answer after an empty reply (no tools bound)."""
    try:
        reply = model.invoke([*history, HumanMessage(FINAL_NUDGE)])
    except openai.OpenAIError as exc:
        raise AgentError(f"Model API error: {exc}") from exc
    return _text(reply.content)


URL_PATTERN = re.compile(r"https?://[^\s)\]>|\"']+")
PRICE_PREFIX = re.compile(r"(?:₹|\bRs\.?|\bINR)\s*(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
PRICE_SUFFIX = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:INR|rupees)", re.IGNORECASE)
SCORE_INLINE = re.compile(r"score\W{0,12}?(\d+(?:\.\d+)?)", re.IGNORECASE)
GATE_MAX_NUDGES = 2
MIN_QUESTIONS = 6


@dataclass(frozen=True)
class Requirements:
    """Steps the agent must complete before its final answer is accepted."""

    tools: frozenset[str] = frozenset()
    min_questions: int = 0
    save_new_wishlist: bool = False
    check_urls: bool = False
    require_link: bool = False
    check_figures: bool = False
    confirm_existing: bool = False
    interview_on_update: bool = False


def build_requirements(interactive: bool) -> Requirements:
    """The steps a real run must complete; the interview applies only when interactive."""
    core = frozenset({"web_search", "calculator"})
    checks = {"check_urls": True, "require_link": True, "check_figures": True}
    if interactive:
        return Requirements(
            core,
            MIN_QUESTIONS,
            save_new_wishlist=True,
            confirm_existing=True,
            interview_on_update=True,
            **checks,
        )
    return Requirements(core, **checks)


def _has_product_link(answer: str, evidence: str) -> bool:
    """True if the answer links a product page that appeared in a search result."""
    urls = (url.rstrip(".,;:") for url in URL_PATTERN.findall(answer))
    return any(is_product_url(url) and url in evidence for url in urls)


def _unsupported_urls(answer: str, evidence: str) -> list[str]:
    """URLs in the answer that never appeared in any tool result."""
    urls = (url.rstrip(".,;:") for url in URL_PATTERN.findall(answer))
    return sorted({url for url in urls if url not in evidence})


def _close_to_any(value: float, allowed: set[float]) -> bool:
    """True if the value is within 1 of an allowed figure (the model may round)."""
    return any(abs(value - other) < 1.0 for other in allowed)


def _unsupported_figures(answer: str, allowed: set[float], calc_scores: set[float]) -> list[str]:
    """Prices and scores in the answer that no tool result supports."""
    bad: list[str] = []

    def flag(label: str) -> None:
        if label not in bad:
            bad.append(label)

    for token in [*PRICE_PREFIX.findall(answer), *PRICE_SUFFIX.findall(answer)]:
        value = _number(token)
        if value is not None and value >= MIN_CHECKED_PRICE and not _close_to_any(value, allowed):
            flag(f"price {token.strip(',')}")

    score_column: int | None = None
    for line in answer.splitlines():
        stripped = line.strip()
        claimed: list[str] = []
        if stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            numeric = [bool(re.fullmatch(r"\d+(?:\.\d+)?", cell)) for cell in cells]
            header = [i for i, cell in enumerate(cells) if "score" in cell.lower()]
            if header and not any(numeric):
                score_column = header[0]
            elif score_column is not None and score_column < len(cells) and numeric[score_column]:
                claimed = [cells[score_column]]
        else:
            score_column = None
            claimed = SCORE_INLINE.findall(stripped)
        for token in claimed:
            value = _number(token)
            if value is not None and value not in calc_scores:
                flag(f"score {token}")
    return bad


REUSE_WORDS = ("reuse", "keep", "same")
UPDATE_WORDS = ("update", "change", "new")


def _wants_update(replies: list[str]) -> bool:
    """True if the buyer's first answer is anything other than a clear 'reuse'."""
    if not replies:
        return False
    text = replies[0].lower()
    reuse = any(word in text for word in REUSE_WORDS)
    update = any(word in text for word in UPDATE_WORDS)
    return not reuse or update


def _missing_steps(
    req: Requirements,
    answer: str,
    succeeded: dict[str, int],
    wishlist_found: bool,
    evidence: str,
    allowed: set[float],
    calc_scores: set[float],
    update_requested: bool = False,
) -> list[str]:
    """List the required steps that have not been done yet."""
    missing = [f"call {name} successfully" for name in sorted(req.tools) if not succeeded.get(name)]
    if not wishlist_found:
        asked = succeeded.get("ask_user", 0)
        if asked < req.min_questions:
            missing.append(
                f"ask the buyer at least {req.min_questions} questions with ask_user "
                f"(so far: {asked})"
            )
        if req.save_new_wishlist and not succeeded.get("save_wishlist"):
            missing.append("save the buyer's answers with save_wishlist")
    if wishlist_found and req.confirm_existing and not succeeded.get("ask_user"):
        missing.append("ask the buyer with ask_user whether to reuse or update the saved wishlist")
    if wishlist_found and req.interview_on_update and update_requested:
        asked = succeeded.get("ask_user", 0)
        if asked < req.min_questions + 1:
            missing.append(
                "interview the buyer field by field with ask_user (at least "
                f"{req.min_questions} questions after the reuse question, so far "
                f"{max(asked - 1, 0)}), then call save_wishlist"
            )
        elif not succeeded.get("save_wishlist"):
            missing.append("save the updated wishlist with save_wishlist")
    if req.check_urls:
        bad_urls = _unsupported_urls(answer, evidence)
        if bad_urls:
            missing.append(
                "remove these URLs, which never appeared in a search result: "
                + ", ".join(bad_urls[:3])
            )
    if req.require_link and not _has_product_link(answer, evidence):
        missing.append(
            "include a product-page link (not a category or search page) copied from a "
            "web_search result"
        )
    if req.check_figures:
        bad_figures = _unsupported_figures(answer, allowed, calc_scores)
        if bad_figures:
            missing.append(
                "remove or correct these figures, which no tool result supports: "
                + ", ".join(bad_figures[:4])
            )
    return missing


def run_agent(
    model: BaseChatModel,
    config: Config,
    user_request: str,
    *,
    tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    on_step: Callable[[Step], None] | None = None,
    requirements: Requirements | None = None,
) -> AgentResult:
    """Run the LangChain agent until an accepted final answer or the step limit."""
    request = user_request.strip() if isinstance(user_request, str) else ""
    if not request:
        raise AgentError("The request is empty.")
    if len(request) > MAX_REQUEST_LENGTH:
        raise AgentError(f"The request is longer than {MAX_REQUEST_LENGTH} characters.")

    req = requirements if requirements is not None else Requirements()
    prompt = system_prompt if system_prompt is not None else load_system_prompt()
    agent = create_agent(
        model, tools=tools if tools is not None else build_tools(), system_prompt=prompt
    )
    steps: list[Step] = []
    history: list[Any] = [SystemMessage(prompt), HumanMessage(request)]
    usage = {"prompt": 0, "completion": 0}
    succeeded: dict[str, int] = {}
    evidence: list[str] = []
    calc_scores: set[float] = set()
    calc_inputs: set[float] = set()
    replies: list[str] = []
    found = {"wishlist": False}

    def record(kind: str, text: str) -> None:
        """Store a step and show it right away if a listener is attached."""
        step = Step(kind, text)
        steps.append(step)
        if on_step is not None:
            on_step(step)

    def run_pass(messages_in: list[Any]) -> tuple[bool, str]:
        """Stream one pass of the agent; return whether it ended with a reply, and its text."""
        final_seen, final_text = False, ""
        stream = agent.stream(
            {"messages": messages_in},
            config={"recursion_limit": config.max_steps * GRAPH_STEPS_PER_TURN + 1},
            stream_mode="updates",
        )
        for update in stream:
            for node_update in update.values():
                if not isinstance(node_update, dict):
                    continue
                for message in node_update.get("messages", []):
                    history.append(message)
                    if isinstance(message, ToolMessage):
                        text = _text(message.content)
                        evidence.append(text)
                        name = getattr(message, "name", None) or ""
                        if not text.startswith("Error:"):
                            succeeded[name] = succeeded.get(name, 0) + 1
                            if name == "read_wishlist" and not text.startswith("No saved"):
                                found["wishlist"] = True
                            if name == "ask_user":
                                replies.append(text)
                            if name == "calculator" and not text.startswith("Duplicate call"):
                                calc_scores.update(_numbers(text))
                        record("observation", text)
                    elif isinstance(message, AIMessage):
                        meta = message.usage_metadata or {}
                        usage["prompt"] += meta.get("input_tokens", 0)
                        usage["completion"] += meta.get("output_tokens", 0)
                        text = _text(message.content)
                        if message.tool_calls:
                            if text:
                                record("thought", text)
                            for call in message.tool_calls:
                                if call["name"] == "calculator":
                                    calc_inputs.update(_numbers(json.dumps(call["args"])))
                                record("action", f"{call['name']}({json.dumps(call['args'])})")
                        else:
                            final_seen, final_text = True, text
        return final_seen, final_text

    started = time.perf_counter()
    messages_in: list[Any] = [{"role": "user", "content": request}]
    try:
        for attempt in range(GATE_MAX_NUDGES + 1):
            final_seen, final_text = run_pass(messages_in)
            if not final_seen:
                return AgentResult("failed", "The agent ended without a final answer.", steps)
            evidence_text = "\n".join(evidence)
            missing = _missing_steps(
                req,
                final_text,
                succeeded,
                found["wishlist"],
                evidence_text,
                _numbers(evidence_text) | calc_inputs,
                calc_scores,
                _wants_update(replies),
            )
            if not missing:
                break
            listing = "; ".join(missing)
            record("observation", f"CartSense rejected the final answer: {listing}")
            if attempt == GATE_MAX_NUDGES:
                reason = f"The agent skipped required steps: {listing}."
                return AgentResult("failed", reason, steps)
            nudge = (
                f"You cannot finish yet. Fix this first: {listing}. Do not invent results. "
                "Reply with only the corrected final answer, no preamble."
            )
            history.append(HumanMessage(nudge))
            messages_in = history[1:]

        if not final_text:
            final_text = _request_final_answer(model, history)
            if not final_text:
                return AgentResult("failed", "The model returned an empty answer.", steps)
        record("final", final_text)
        return AgentResult("done", final_text, steps)
    except GraphRecursionError:
        return AgentResult("max_steps", MAX_STEPS_MESSAGE, steps)
    except openai.OpenAIError as exc:
        raise AgentError(f"Model API error: {exc}") from exc
    finally:
        logger.info(
            "agent_run latency_ms=%d steps=%d prompt_tokens=%d completion_tokens=%d",
            (time.perf_counter() - started) * 1000,
            len(steps),
            usage["prompt"],
            usage["completion"],
        )
