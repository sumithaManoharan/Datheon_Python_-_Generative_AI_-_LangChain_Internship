"""Grounded answer generation: prompt building, LLM call with retry, citations."""

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI

from docqa.config import PROJECT_ROOT
from docqa.errors import ConfigError, InvalidQuestionError, LLMCallError
from docqa.retriever import RetrievedChunk
from docqa.retry import call_with_retry, is_transient_api_error

logger = logging.getLogger(__name__)

PROMPT_PATH = PROJECT_ROOT / "prompts" / "grounding.txt"
REFUSAL_MESSAGE = "I could not find this in the loaded documents."
SNIPPET_CHARS = 200
CITATION_RE = re.compile(r"\[(\d+)\]")
# Models sometimes write fullwidth brackets (【1】) or lists ([1, 2]); rewrite to [1][2].
MARKER_RE = re.compile(r"[\[【［]\s*(\d+(?:\s*[,，]\s*\d+)*)\s*[\]】］]")


@dataclass(frozen=True)
class Citation:
    """Where an answer's evidence came from. PDFs set `page`; text sets section and lines."""

    number: int
    source: str
    snippet: str
    page: int | None = None
    section: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    def location(self) -> str:
        """Human-readable location: 'page 3' or 'Section, lines 4-9'."""
        if self.page is not None:
            return f"page {self.page}"
        parts: list[str] = []
        if self.section:
            parts.append(self.section)
        if self.start_line is not None:
            if self.start_line == self.end_line:
                parts.append(f"line {self.start_line}")
            else:
                parts.append(f"lines {self.start_line}-{self.end_line}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Answer:
    """The final result for one question."""

    text: str
    citations: tuple[Citation, ...]
    refused: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def validate_question(question: str, max_chars: int) -> str:
    """Return the stripped question, or raise InvalidQuestionError if unusable."""
    cleaned = question.strip()
    if not cleaned:
        raise InvalidQuestionError("the question is empty")
    if len(cleaned) > max_chars:
        raise InvalidQuestionError(
            f"the question has {len(cleaned)} characters; the limit is {max_chars}"
        )
    return cleaned


def load_system_prompt(path: Path = PROMPT_PATH) -> str:
    """Read the grounding prompt and fill in the fixed refusal message."""
    try:
        template = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read prompt file {path}: {exc}") from exc
    if not template.strip():
        raise ConfigError(f"prompt file {path} is empty")
    # replace(), not format(): the prompt may contain other curly braces.
    return template.replace("{refusal}", REFUSAL_MESSAGE).strip()


def _snippet(text: str) -> str:
    """Collapse whitespace and cut long text for display."""
    flat = " ".join(text.split())
    if len(flat) <= SNIPPET_CHARS:
        return flat
    return flat[:SNIPPET_CHARS].rstrip() + "..."


def _build_citations(retrieved: Sequence[RetrievedChunk]) -> list[Citation]:
    """Number the retrieved chunks from 1 and turn each into a citation."""
    return [
        Citation(
            number=number,
            source=Path(item.hit.source).name,
            snippet=_snippet(item.hit.text),
            page=item.hit.page,
            section=item.hit.section,
            start_line=item.hit.start_line,
            end_line=item.hit.end_line,
        )
        for number, item in enumerate(retrieved, start=1)
    ]


def _user_message(
    question: str, citations: Sequence[Citation], retrieved: Sequence[RetrievedChunk]
) -> str:
    """Numbered context passages followed by the question."""
    passages = []
    for citation, item in zip(citations, retrieved, strict=True):
        label = f"[{citation.number}] {citation.source}"
        if citation.location():
            label += f" ({citation.location()})"
        passages.append(f"{label}\n{item.hit.text}")
    return "Context:\n\n" + "\n\n".join(passages) + f"\n\nQuestion: {question}"


def _is_refusal(text: str) -> bool:
    """True if the model replied with the fixed refusal (ignoring a trailing period)."""
    return text.strip().rstrip(".") == REFUSAL_MESSAGE.rstrip(".")


def _normalize_markers(text: str) -> str:
    """Rewrite fullwidth brackets and comma lists as plain [n] markers."""
    return MARKER_RE.sub(
        lambda match: "".join(f"[{number}]" for number in re.findall(r"\d+", match.group(1))),
        text,
    )


def _cited(text: str, citations: Sequence[Citation]) -> tuple[Citation, ...]:
    """Citations the answer refers to; all of them if it refers to none that exist."""
    valid = {citation.number for citation in citations}
    used = {int(m.group(1)) for m in CITATION_RE.finditer(text)} & valid
    if not used:
        return tuple(citations)
    return tuple(citation for citation in citations if citation.number in used)


def _extract_text(response: Any) -> str:
    """The answer text, or LLMCallError if the model sent nothing usable."""
    choices = getattr(response, "choices", None)
    content = choices[0].message.content if choices else None
    if not content or not content.strip():
        raise LLMCallError("the model returned an empty answer")
    return content.strip()


HISTORY_RULE = (
    "\n7. Earlier messages are the conversation so far. Use them only to understand what the "
    "question refers to (for example 'he' or 'that product'). Every fact in your answer must "
    "still come from the numbered context in the last message, and you must not reuse passage "
    "numbers from earlier answers."
)
CONDENSE_PROMPT = (
    "Rewrite the follow-up question as one standalone question that can be understood without "
    "the conversation. Replace pronouns and references such as 'he', 'it' or 'that product' "
    "with the names they stand for. Do not answer the question. If it is already standalone, "
    "return it unchanged. Reply with the question only."
)
OLD_MARKER_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


@dataclass(frozen=True)
class Turn:
    """One earlier question and the answer it received."""

    question: str
    answer: str


def strip_citations(text: str) -> str:
    """Remove [n] markers so old passage numbers cannot be mistaken for the current ones."""
    return OLD_MARKER_RE.sub("", text).strip()


def _history_messages(history: Sequence[Turn]) -> list[dict[str, str]]:
    """Earlier turns as chat messages: the question, then the answer without its markers."""
    messages: list[dict[str, str]] = []
    for turn in history:
        messages.append({"role": "user", "content": turn.question})
        messages.append({"role": "assistant", "content": strip_citations(turn.answer)})
    return messages


class AnswerGenerator:
    """Turns a question and retrieved chunks into a grounded, cited answer.

    The client is injected (any OpenAI-compatible client), so tests pass a fake.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_question_chars: int = 500,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_question_chars = max_question_chars
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._sleep = sleep

    def answer(
        self,
        question: str,
        retrieved: Sequence[RetrievedChunk],
        history: Sequence[Turn] = (),
    ) -> Answer:
        """Answer from the retrieved chunks only; refuse when there is no evidence."""
        question = validate_question(question, self._max_question_chars)
        if not retrieved:
            logger.info("no relevant chunks; refusing without an LLM call")
            return Answer(text=REFUSAL_MESSAGE, citations=(), refused=True)
        citations = _build_citations(retrieved)
        system_prompt = self._system_prompt + HISTORY_RULE if history else self._system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            *_history_messages(history),
            {"role": "user", "content": _user_message(question, citations, retrieved)},
        ]
        response = self._complete(messages)
        text = _normalize_markers(_extract_text(response))
        usage = getattr(response, "usage", None)
        tokens = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
        if _is_refusal(text):
            logger.info("model found no answer in the context; refusing")
            return Answer(text=REFUSAL_MESSAGE, citations=(), refused=True, **tokens)
        return Answer(text=text, citations=_cited(text, citations), refused=False, **tokens)

    def condense(self, question: str, history: Sequence[Turn]) -> str:
        """Rewrite a follow-up as a standalone question; no history means no LLM call."""
        if not history:
            return question
        transcript = "\n".join(
            f"Q: {turn.question}\nA: {strip_citations(turn.answer)}" for turn in history
        )
        messages = [
            {"role": "system", "content": CONDENSE_PROMPT},
            {
                "role": "user",
                "content": f"Conversation:\n{transcript}\n\nFollow-up question: {question}",
            },
        ]
        rewritten = " ".join(_extract_text(self._complete(messages)).split())
        logger.debug("follow-up rewritten as: %s", rewritten)
        if len(rewritten) > self._max_question_chars:
            logger.warning("rewritten question is too long; searching with the original")
            return question
        return rewritten

    def _complete(self, messages: list[dict[str, str]]) -> Any:
        """One chat completion with retry on transient errors."""
        try:
            return call_with_retry(
                lambda: self._client.chat.completions.create(
                    model=self._model, messages=messages, temperature=self._temperature
                ),
                is_transient=is_transient_api_error,
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
                sleep=self._sleep,
            )
        except openai.OpenAIError as exc:
            raise LLMCallError(f"LLM request failed: {exc}") from exc
