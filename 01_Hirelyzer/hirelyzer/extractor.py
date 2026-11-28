"""Orchestrate one extraction: build prompt, call the model, parse, validate, repair."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

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
    """Runs the extraction stages. The LLM clients and prompt builder are injected."""

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptBuilder,
        max_repair_attempts: int = 2,
        vision_llm: LLMClient | None = None,
    ) -> None:
        """Store the collaborators; tests pass fake LLMs here. Without a vision client,
        image extraction returns a failed result instead of raising."""
        self._llm = llm
        self._vision_llm = vision_llm
        self._prompts = prompts
        self._max_repair_attempts = max_repair_attempts

    def extract(self, document_text: str, source: str = "<text>") -> ExtractionResult:
        """Extract a validated Resume from text, repairing invalid output when possible."""
        messages = self._prompts.build_messages(document_text)
        return self._run(self._llm, messages, source, "text")

    def extract_image(self, image_data_url: str, source: str = "<image>") -> ExtractionResult:
        """Extract a validated Resume from an image (a data URL) with the vision model."""
        if self._vision_llm is None:
            return ExtractionResult(source=source, errors=["No vision model is configured."])
        messages = self._prompts.build_vision_messages(image_data_url)
        return self._run(self._vision_llm, messages, source, "vision")

    def _run(
        self, llm: LLMClient, base_messages: list[dict[str, Any]], source: str, pipeline: str
    ) -> ExtractionResult:
        """Call the model, validate the reply, and repair it up to the allowed number of times."""
        messages = base_messages
        result = ExtractionResult(source=source)
        total_attempts = self._max_repair_attempts + 1  # first try plus repair rounds
        start = time.perf_counter()

        for attempt in range(1, total_attempts + 1):
            result.attempts = attempt
            try:
                response = llm.complete(messages, source=source, pipeline=pipeline)
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
