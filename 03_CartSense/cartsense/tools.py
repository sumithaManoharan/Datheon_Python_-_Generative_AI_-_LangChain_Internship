"""Tools the agent can call: calculator, wishlist reader and web search."""

import ast
import logging
import operator
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cartsense.config import DEFAULT_SEARCH_REGION, PROJECT_ROOT
from cartsense.models import CATEGORY_PATTERN, Wishlist, clean_category

logger = logging.getLogger(__name__)

WISHLIST_DIR = PROJECT_ROOT / "wishlists"
MAX_WISHLIST_BYTES = 10_000
MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100
MAX_ABS_RESULT = 1e15
MAX_QUERY_LENGTH = 200
MAX_SNIPPET_LENGTH = 600
MAX_RESULTS_LIMIT = 10
MIN_RESULTS_LIMIT = 3
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


AD_URL_MARKERS = ("bing.com/aclick", "duckduckgo.com/y.js")


def _is_ad(url: str) -> bool:
    """True for sponsored-ad redirect links, which are not product pages."""
    return any(marker in url for marker in AD_URL_MARKERS)


STORE_DOMAINS: dict[str, list[str] | None] = {
    "any": None,
    "amazon": ["amazon.in"],
    "flipkart": ["flipkart.com"],
}


def make_tavily_search(api_key: str, store: str | None = None) -> SearchFn:
    """Build a Tavily search function, optionally limited to one store's domain."""
    name = (store or "any").lower()
    if name not in STORE_DOMAINS:
        raise ToolError(f"Unknown store {store!r}; use any, amazon or flipkart.")
    domains = STORE_DOMAINS[name]

    def search(query: str, max_results: int) -> list[dict[str, str]]:
        """Run a basic-depth Tavily search; imported lazily so tests never need it."""
        from tavily import TavilyClient

        options: dict[str, object] = {"max_results": max_results, "search_depth": "basic"}
        if domains:
            options["include_domains"] = domains
        response = TavilyClient(api_key=api_key).search(query, **options)
        return [
            {
                "title": hit.get("title", ""),
                "href": hit.get("url", ""),
                "body": hit.get("content", ""),
            }
            for hit in response.get("results", [])
        ]

    return search


def make_ddgs_search(region: str) -> SearchFn:
    """Build a DuckDuckGo search function bound to a region such as 'in-en'."""

    def search(query: str, max_results: int) -> list[dict[str, str]]:
        """Run a DuckDuckGo text search. Imported lazily so tests never need the network."""
        from ddgs import DDGS

        return DDGS().text(query, region=region, max_results=max_results)

    return search


def web_search(
    query: str,
    max_results: int = 5,
    *,
    region: str = DEFAULT_SEARCH_REGION,
    search_fn: SearchFn | None = None,
    provider: str = "ddgs",
    store: str | None = None,
    tavily_api_key: str = "",
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = SEARCH_MAX_ATTEMPTS,
) -> list[dict[str, str]]:
    """Search the web with retry and backoff; return title, url and snippet per hit."""
    text = query.strip() if isinstance(query, str) else ""
    if not text:
        raise ToolError("Search query is empty.")
    if len(text) > MAX_QUERY_LENGTH:
        raise ToolError(f"Search query is longer than {MAX_QUERY_LENGTH} characters.")
    limit = max(MIN_RESULTS_LIMIT, min(max_results, MAX_RESULTS_LIMIT))

    backend = search_fn
    if backend is None:
        if provider == "tavily":
            backend = make_tavily_search(tavily_api_key, store)
        else:
            backend = make_ddgs_search(region)
    raw: list[dict[str, str]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            raw = backend(text, limit)
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
        if not _is_ad(str(hit.get("href", "")))
    ]


# --- interview and wishlist saving -----------------------------------------

MAX_QUESTIONS = 10
MAX_QUESTION_LENGTH = 300
MAX_ANSWER_LENGTH = 500
NO_ANSWER = "(no answer given)"


class UserPrompter:
    """Asks the buyer questions on the console, with a hard cap on how many."""

    def __init__(
        self, input_fn: Callable[[str], str] = input, max_questions: int = MAX_QUESTIONS
    ) -> None:
        self._input_fn = input_fn
        self._max_questions = max_questions
        self.asked = 0
        self._closed = False

    def close(self) -> None:
        """Stop further questions once the wishlist is saved."""
        self._closed = True

    def ask(self, question: str) -> str:
        """Ask one question and return the trimmed answer."""
        text = question.strip() if isinstance(question, str) else ""
        if not text:
            raise ToolError("Question is empty.")
        if len(text) > MAX_QUESTION_LENGTH:
            raise ToolError(f"Question is longer than {MAX_QUESTION_LENGTH} characters.")
        if self._closed:
            raise ToolError(
                "The interview is over because the wishlist is saved. "
                "Do not ask more questions; continue with the research."
            )
        if self.asked >= self._max_questions:
            raise ToolError(
                f"Question limit reached ({self._max_questions}). "
                "Continue with the answers you have and state assumptions."
            )
        self.asked += 1
        try:
            answer = self._input_fn(f"\n? {text}\n> ")
        except EOFError:
            raise ToolError("No input is available; continue with defaults.") from None
        return answer.strip()[:MAX_ANSWER_LENGTH] or NO_ANSWER


def save_wishlist(data: dict[str, Any], directory: Path = WISHLIST_DIR) -> Path:
    """Validate the data and write it to <directory>/<category>.json; return the path.

    The filename comes only from the cleaned, pattern-checked category.
    """
    if not isinstance(data, dict):
        raise ToolError("Wishlist data must be an object.")
    try:
        wishlist = Wishlist.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ToolError(f"Invalid wishlist: {problems}") from None

    payload = wishlist.model_dump_json(indent=2)
    if len(payload.encode("utf-8")) > MAX_WISHLIST_BYTES:
        raise ToolError(f"Wishlist is larger than {MAX_WISHLIST_BYTES} bytes.")
    path = wishlist_path(wishlist.category, directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        logger.error("Could not write %s: %s", path.name, exc)
        raise ToolError("Could not write the wishlist file.") from exc
    return path


# --- product page detection ---------------------------------------------------

PRODUCT_URL_PATTERNS = (
    re.compile(r"amazon\.[a-z.]+/(?:[^/?#]+/)?dp/[A-Z0-9]{10}"),
    re.compile(r"amazon\.[a-z.]+/gp/product/[A-Z0-9]{10}"),
    re.compile(r"flipkart\.com/(?:[^/?#]+/)?[^/?#]+/p/itm"),
    re.compile(r"/(?:product|proddetail)/[^/?#]+"),
)
LISTING_MARKERS = ("/s?k=", "/pr?sid", "-brand/")


def is_product_url(url: str) -> bool:
    """True for a URL that looks like one product's page, not a search or listing page."""
    if any(marker in url for marker in LISTING_MARKERS):
        return False
    return any(pattern.search(url) for pattern in PRODUCT_URL_PATTERNS)
