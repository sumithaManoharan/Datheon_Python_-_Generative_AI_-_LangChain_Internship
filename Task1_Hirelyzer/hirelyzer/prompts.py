"""Build the LLM messages: a parameterized system prompt plus the document text.
The template and the few-shot examples live in the prompts/ folder, not in code.
They are loaded and validated once at startup, so a broken prompt file fails fast.
"""

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from pydantic import ValidationError

from hirelyzer.errors import ConfigError
from hirelyzer.schema import Resume

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
TEMPLATE_FILE = "system.txt"
EXAMPLES_FILE = "examples.json"
PLACEHOLDERS = ("schema", "examples", "extra_rules")
_PLACEHOLDER_RE = re.compile(r"\{(" + "|".join(PLACEHOLDERS) + r")\}")


@dataclass(frozen=True)
class FewShotExample:
    """One worked example: a document and the JSON the model should return for it."""

    document: str
    output: dict[str, Any]


def resume_schema_json() -> str:
    """Return the Resume JSON schema as pretty-printed text for the prompt."""
    return json.dumps(Resume.model_json_schema(), indent=2)


def load_template(prompts_dir: Path = PROMPTS_DIR) -> str:
    """Read the system prompt template and check every placeholder appears exactly once."""
    path = prompts_dir / TEMPLATE_FILE
    try:
        template = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read prompt template {path}: {exc}") from exc
    for name in PLACEHOLDERS:
        if template.count("{" + name + "}") != 1:
            raise ConfigError(
                f"{TEMPLATE_FILE} must contain the placeholder {{{name}}} exactly once"
            )
    return template


def load_examples(prompts_dir: Path = PROMPTS_DIR) -> list[FewShotExample]:
    """Read the few-shot examples and check each output is valid against the Resume schema."""
    path = prompts_dir / EXAMPLES_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read few-shot examples from {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{EXAMPLES_FILE} must contain a non-empty JSON list")

    examples: list[FewShotExample] = []
    for number, item in enumerate(raw, start=1):
        try:
            document, output = item["document"], item["output"]
            if not isinstance(document, str) or not document.strip():
                raise ValueError("document must be a non-empty string")
            Resume.model_validate(output)
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ConfigError(f"Example {number} in {EXAMPLES_FILE} is invalid: {exc}") from exc
        examples.append(FewShotExample(document=document, output=output))
    return examples


def format_examples(examples: Sequence[FewShotExample]) -> str:
    """Render the examples as text blocks for the {examples} placeholder."""
    blocks = []
    for number, example in enumerate(examples, start=1):
        output_text = json.dumps(example.output, indent=2, ensure_ascii=False)
        blocks.append(f"EXAMPLE {number}\nDOCUMENT:\n{example.document}\n\nJSON:\n{output_text}")
    return "\n\n".join(blocks)


def render_template(template: str, values: Mapping[str, str]) -> str:
    """Fill the placeholders in a single pass, so inserted text is never re-scanned.

    str.format cannot be used here because the schema and the examples contain braces.
    """
    rendered = _PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], template)
    return re.sub(r"\n{3,}", "\n\n", rendered).strip()


class PromptBuilder:
    """Builds the chat messages for one extraction request."""

    def __init__(
        self,
        template: str,
        examples: Sequence[FewShotExample],
        schema_json: str,
        extra_rules: str = "",
    ) -> None:
        """Render the system prompt once from the given template, examples and schema."""
        self._system_prompt = render_template(
            template,
            {
                "schema": schema_json,
                "examples": format_examples(examples),
                "extra_rules": extra_rules.strip(),
            },
        )

    @classmethod
    def from_files(cls, prompts_dir: Path = PROMPTS_DIR, extra_rules: str = "") -> Self:
        """Load and validate the prompt files, then build a ready-to-use builder."""
        examples = load_examples(prompts_dir)
        logger.debug("Loaded prompt template and %d few-shot examples", len(examples))
        return cls(load_template(prompts_dir), examples, resume_schema_json(), extra_rules)

    @property
    def system_prompt(self) -> str:
        """The fully rendered system prompt."""
        return self._system_prompt

    def build_messages(self, document_text: str) -> list[dict[str, str]]:
        """Return the system and user messages for one document's extracted text."""
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Document:\n\n{document_text}"},
        ]

    def build_vision_messages(self, image_data_url: str) -> list[dict[str, Any]]:
        """Return the system and user messages for one image (the same system prompt)."""
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the JSON for the document in this image."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ]
