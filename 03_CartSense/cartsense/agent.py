"""The LangChain tool-calling agent and its step trace."""

import functools
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from cartsense.config import PROJECT_ROOT, Config
from cartsense.scoring import tco, value_score
from cartsense.tools import (
    WISHLIST_DIR,
    ToolError,
    UserPrompter,
    calculator,
    read_wishlist,
    save_wishlist,
    web_search,
)

logger = logging.getLogger(__name__)

PROMPT_PATH = PROJECT_ROOT / "prompts" / "system.md"
MAX_REQUEST_LENGTH = 500
MAX_OBSERVATION_CHARS = 4000
MODEL_MAX_RETRIES = 3
MODEL_TIMEOUT_SECONDS = 60
# One model call plus one tool call is about two graph steps.
GRAPH_STEPS_PER_TURN = 2
MAX_STEPS_MESSAGE = "Stopped: the step limit was reached before a final answer."
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

SCORING_FUNCTIONS: dict[str, Callable[..., float]] = {"tco": tco, "value_score": value_score}


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
            "Wishlist fields: category (text), budget (number), years_of_use (number), "
            "warranty_years (number), priorities (1 or 2 short texts), dealbreakers "
            "(list of short texts), specs (object of short texts)."
        )
    )


class WebSearchArgs(BaseModel):
    """Arguments for web_search."""

    query: str = Field(description="Search query for prices, reviews or specs.")
    max_results: int = Field(default=5, description="How many results to return (1 to 10).")


class CalculatorArgs(BaseModel):
    """Arguments for calculator: an expression, or a scoring function with arguments."""

    expression: str | None = Field(
        default=None, description="Arithmetic using numbers and + - * / // % ** ( ) only."
    )
    function: str | None = Field(default=None, description="'tco' or 'value_score'.")
    arguments: dict[str, Any] | None = Field(
        default=None, description="Keyword arguments for the chosen function."
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


@_guard
def _calculator_tool(
    expression: str | None = None,
    function: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Run a plain expression, or one of the pure scoring functions."""
    if function is None:
        return str(calculator(expression or ""))
    scorer = SCORING_FUNCTIONS.get(function)
    if scorer is None:
        raise ToolError(f"Unknown function {function!r}; use one of {sorted(SCORING_FUNCTIONS)}.")
    try:
        return str(scorer(**(arguments or {})))
    except (TypeError, ValueError) as exc:
        raise ToolError(f"Bad arguments for {function}: {exc}") from None


@_guard
def _web_search_tool(query: str, max_results: int = 5) -> str:
    """Return search hits as a JSON string."""
    return json.dumps(web_search(query, max_results))


def build_tools(
    *, wishlist_dir: Path = WISHLIST_DIR, prompter: UserPrompter | None = None
) -> list[BaseTool]:
    """Build the LangChain tools.

    ask_user and save_wishlist exist only when a prompter is given (interactive mode).
    """

    @_guard
    def read_wishlist_tool(category: str) -> str:
        """Return the saved wishlist as JSON, or a plain message when there is none."""
        wishlist = read_wishlist(category, wishlist_dir)
        if wishlist is None:
            return f"No saved wishlist for {category!r}."
        return wishlist.model_dump_json()

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
            return prompter.ask(question)

        @_guard
        def save_wishlist_tool(data: dict[str, Any]) -> str:
            """Validate and save the wishlist; return a confirmation."""
            path = save_wishlist(data, wishlist_dir)
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

    tools.append(
        StructuredTool.from_function(
            func=_web_search_tool,
            name="web_search",
            description="Search the web for current prices, reviews and specs.",
            args_schema=WebSearchArgs,
        )
    )
    tools.append(
        StructuredTool.from_function(
            func=_calculator_tool,
            name="calculator",
            description=(
                "Do arithmetic. Pass 'expression', or pass 'function' ('tco' or "
                "'value_score') with its 'arguments' object."
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


def run_agent(
    model: BaseChatModel,
    config: Config,
    user_request: str,
    *,
    tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> AgentResult:
    """Run the LangChain agent until a final answer or the step limit."""
    request = user_request.strip() if isinstance(user_request, str) else ""
    if not request:
        raise AgentError("The request is empty.")
    if len(request) > MAX_REQUEST_LENGTH:
        raise AgentError(f"The request is longer than {MAX_REQUEST_LENGTH} characters.")

    prompt = system_prompt if system_prompt is not None else load_system_prompt()
    agent = create_agent(
        model, tools=tools if tools is not None else build_tools(), system_prompt=prompt
    )
    steps: list[Step] = []
    usage = {"prompt": 0, "completion": 0}
    final_seen = False
    final_text = ""

    def record(kind: str, text: str) -> None:
        """Store a step and show it right away if a listener is attached."""
        step = Step(kind, text)
        steps.append(step)
        if on_step is not None:
            on_step(step)

    started = time.perf_counter()
    try:
        stream = agent.stream(
            {"messages": [{"role": "user", "content": request}]},
            config={"recursion_limit": config.max_steps * GRAPH_STEPS_PER_TURN + 1},
            stream_mode="updates",
        )
        for update in stream:
            for node_update in update.values():
                if not isinstance(node_update, dict):
                    continue
                for message in node_update.get("messages", []):
                    if isinstance(message, ToolMessage):
                        record("observation", _text(message.content))
                    elif isinstance(message, AIMessage):
                        meta = message.usage_metadata or {}
                        usage["prompt"] += meta.get("input_tokens", 0)
                        usage["completion"] += meta.get("output_tokens", 0)
                        text = _text(message.content)
                        if message.tool_calls:
                            if text:
                                record("thought", text)
                            for call in message.tool_calls:
                                record("action", f"{call['name']}({json.dumps(call['args'])})")
                        else:
                            final_seen, final_text = True, text
                            if text:
                                record("final", text)
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

    if not final_seen:
        return AgentResult("failed", "The agent ended without a final answer.", steps)
    if not final_text:
        return AgentResult("failed", "The model returned an empty answer.", steps)
    return AgentResult("done", final_text, steps)
