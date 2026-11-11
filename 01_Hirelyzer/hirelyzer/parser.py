"""Turn raw model output into a JSON object, tolerating code fences and extra prose."""

import json
import re
from typing import Any

from hirelyzer.errors import OutputValidationError

# Matches ```json ... ``` or ``` ... ``` and captures what is inside the fence.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_code_fence(text: str) -> str:
    """Return the text inside the first markdown code fence, or the text unchanged."""
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def parse_model_output(raw: str) -> dict[str, Any]:
    """Parse the model's reply into a dict; raise OutputValidationError if it is not valid JSON."""
    text = strip_code_fence(raw)

    # Skip any prose before the JSON. An empty or prose-only reply has no "{" at all.
    start = text.find("{")
    if start == -1:
        raise OutputValidationError("No JSON object found in the model output.")

    try:
        # raw_decode reads one JSON value from `start` and ignores trailing prose.
        data, _end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        # Keep the parser's message: the repair loop sends it back to the model.
        raise OutputValidationError(f"Invalid JSON: {exc}") from exc

    # We started decoding at "{", so a successful result is always a dict.
    return data
