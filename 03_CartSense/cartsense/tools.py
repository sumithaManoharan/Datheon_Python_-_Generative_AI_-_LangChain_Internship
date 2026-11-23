"""Tools the agent can call: calculator, wishlist reader and web search."""

import ast
import logging
import operator
import random
import re
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from cartsense.config import PROJECT_ROOT
from cartsense.models import CATEGORY_PATTERN, Wishlist, clean_category

logger = logging.getLogger(__name__)

WISHLIST_DIR = PROJECT_ROOT / "wishlists"
MAX_WISHLIST_BYTES = 10_000
MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100
MAX_ABS_RESULT = 1e15
MAX_QUERY_LENGTH = 200
MAX_SNIPPET_LENGTH = 300
MAX_RESULTS_LIMIT = 10
SEARCH_MAX_ATTEMPTS = 3
SEARCH_BASE_DELAY = 1.0

SearchFn = Callable[[str, int], list[dict[str, str]]]


class ToolError(Exception):
    """A tool failed in a way the agent can read and recover from."""


# --- calculator -------------------------------------------------------------

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.expr) -> float:
    """Evaluate one whitelisted AST node; anything else is rejected."""
    # type() is used instead of isinstance so that True/False are not treated as numbers.
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"Exponent too large (limit {MAX_EXPONENT}).")
        result = _BINARY_OPS[type(node.op)](left, right)
        if isinstance(result, complex) or abs(result) > MAX_ABS_RESULT:
            raise ToolError("Result is out of the supported range.")
        return result
    raise ToolError("Only numbers and + - * / // % ** ( ) are allowed.")


def calculator(expression: str) -> float:
    """Safely evaluate an arithmetic expression and return the result."""
    text = expression.strip() if isinstance(expression, str) else ""
    if not text:
        raise ToolError("Expression is empty.")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"Expression is longer than {MAX_EXPRESSION_LENGTH} characters.")
    try:
        tree = ast.parse(text, mode="eval")
        value = _eval_node(tree.body)
    except (SyntaxError, ValueError, RecursionError):
        raise ToolError("Not a valid arithmetic expression.") from None
    except ZeroDivisionError:
        raise ToolError("Division by zero.") from None
    except OverflowError:
        raise ToolError("Result is out of the supported range.") from None
    return round(value, 6)


# --- wishlist reader --------------------------------------------------------


def wishlist_path(category: str, directory: Path = WISHLIST_DIR) -> Path:
    """Build the wishlist file path from a cleaned category name, never from raw input."""
    cleaned = clean_category(category)
    if not re.fullmatch(CATEGORY_PATTERN, cleaned):
        raise ToolError(
            "Invalid category: use letters, digits, spaces, '-' or '_' (max 40 characters)."
        )
    return directory / f"{cleaned.replace(' ', '_')}.json"


def read_wishlist(category: str, directory: Path = WISHLIST_DIR) -> Wishlist | None:
    """Return the saved wishlist for a category, or None if there is none."""
    path = wishlist_path(category, directory)
    if not path.is_file():
        return None
    if path.stat().st_size > MAX_WISHLIST_BYTES:
        raise ToolError(f"Saved wishlist is larger than {MAX_WISHLIST_BYTES} bytes.")
    try:
        return Wishlist.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, UnicodeDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        raise ToolError("Saved wishlist is corrupt or invalid; ask the user to redo it.") from None


# --- web search -------------------------------------------------------------


def _ddgs_text(query: str, max_results: int) -> list[dict[str, str]]:
    """Run a DuckDuckGo text search. Imported lazily so tests never need the network."""
    from ddgs import DDGS

    return DDGS().text(query, max_results=max_results)


def web_search(
    query: str,
    max_results: int = 5,
    *,
    search_fn: SearchFn = _ddgs_text,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = SEARCH_MAX_ATTEMPTS,
) -> list[dict[str, str]]:
    """Search the web with retry and backoff; return title, url and snippet per hit."""
    text = query.strip() if isinstance(query, str) else ""
    if not text:
        raise ToolError("Search query is empty.")
    if len(text) > MAX_QUERY_LENGTH:
        raise ToolError(f"Search query is longer than {MAX_QUERY_LENGTH} characters.")
    limit = max(1, min(max_results, MAX_RESULTS_LIMIT))

    raw: list[dict[str, str]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            raw = search_fn(text, limit)
            break
        except Exception as exc:  # backend errors vary; narrow once real ones are seen
            if attempt == max_attempts:
                logger.error("Search failed after %d attempts: %s", attempt, exc)
                raise ToolError("Web search is unavailable right now.") from exc
            delay = SEARCH_BASE_DELAY * 2 ** (attempt - 1) + random.uniform(0, 0.5)
            logger.warning("Search attempt %d failed (%s); retrying in %.1fs", attempt, exc, delay)
            sleep(delay)

    return [
        {
            "title": str(hit.get("title", "")),
            "url": str(hit.get("href", "")),
            "snippet": str(hit.get("body", ""))[:MAX_SNIPPET_LENGTH],
        }
        for hit in raw
    ]
