"""Orchestrate one extraction: build prompt, call the model, parse, validate, repair."""

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import ValidationError

from hirelyzer.errors import LLMCallError, OutputValidationError
from hirelyzer.llm import LLMClient
from hirelyzer.parser import parse_model_output
from hirelyzer.prompts import PromptBuilder
from hirelyzer.schema import Resume

logger = logging.getLogger(__name__)

MAX_ERROR_CHARS = 1500  # keep the error text sent back to the model short

REPAIR_TEMPLATE = (
    "Your previous reply was rejected:\n{error}\n\n"
    "Return the corrected JSON object only, following the same schema and rules."
)


@dataclass
class ExtractionResult:
    """Outcome of one extraction. A failed extraction is a result, not an exception."""

    source: str
    status: Literal["success", "failed"] = "failed"
    data: Resume | None = None
    errors: list[str] = field(default_factory=list)
    attempts: int = 0  # model calls made: the first try plus any repair rounds
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class ResumeExtractor:
    """Runs the extraction stages. The LLM client and prompt builder are injected."""

    def __init__(
        self, llm: LLMClient, prompts: PromptBuilder, max_repair_attempts: int = 2
    ) -> None:
        """Store the collaborators; tests pass a fake LLM here."""
        self._llm = llm
        self._prompts = prompts
        self._max_repair_attempts = max_repair_attempts

    def extract(self, document_text: str, source: str = "<text>") -> ExtractionResult:
        """Extract a validated Resume from text, repairing invalid output when possible."""
        base_messages = self._prompts.build_messages(document_text)
        messages = base_messages
        result = ExtractionResult(source=source)
        total_attempts = self._max_repair_attempts + 1  # first try plus repair rounds
        start = time.perf_counter()

        for attempt in range(1, total_attempts + 1):
            result.attempts = attempt
            try:
                response = self._llm.complete(messages)
            except LLMCallError as exc:
                # Retries were already used up inside the client, so repairing cannot help.
                result.errors.append(str(exc))
                break

            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens

            try:
                result.data = Resume.model_validate(parse_model_output(response.content))
            except (OutputValidationError, ValidationError) as exc:
                error_text = str(exc)[:MAX_ERROR_CHARS]
                result.errors.append(f"Attempt {attempt}: {error_text}")
                logger.warning(
                    "Invalid model output for %s (attempt %d/%d)", source, attempt, total_attempts
                )
                # Rebuild from the original messages so the conversation never grows past
                # one rejected reply plus its error.
                messages = [
                    *base_messages,
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": REPAIR_TEMPLATE.format(error=error_text)},
                ]
                continue

            result.status = "success"
            break

        result.latency_ms = (time.perf_counter() - start) * 1000
        return result
