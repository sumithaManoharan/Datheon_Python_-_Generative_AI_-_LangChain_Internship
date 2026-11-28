"""Tests for turning raw model output into a JSON object."""

from typing import Any

import pytest
from hirelyzer.errors import OutputValidationError
from hirelyzer.parser import parse_model_output, strip_code_fence


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('Here you go:\n{"a": 1}', {"a": 1}),
        ('{"a": 1}\nHope this helps!', {"a": 1}),
        ('{"name": "Zoë Müller"}', {"name": "Zoë Müller"}),
        ('{"a": null, "b": []}', {"a": None, "b": []}),
        ('{"outer": {"inner": [1, 2]}}', {"outer": {"inner": [1, 2]}}),
    ],
    ids=[
        "plain",
        "fenced",
        "fenced_no_tag",
        "leading_prose",
        "trailing_prose",
        "unicode",
        "nulls_and_empty_list",
        "nested",
    ],
)
def test_parse_valid_outputs(raw: str, expected: dict[str, Any]) -> None:
    assert parse_model_output(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ['{"a": [1, 2', "", "   ", "Sorry, I cannot help.", '{"a": 1,}', "[1, 2, 3]"],
    ids=["truncated", "empty", "blank", "prose_only", "trailing_comma", "array_not_object"],
)
def test_parse_invalid_outputs_raise(raw: str) -> None:
    with pytest.raises(OutputValidationError):
        parse_model_output(raw)


def test_truncated_json_error_mentions_decoder_detail() -> None:
    # The repair loop sends this message back to the model, so it must be informative.
    with pytest.raises(OutputValidationError, match="Invalid JSON"):
        parse_model_output('{"a": [1, 2')


def test_strip_code_fence_leaves_plain_text_alone() -> None:
    assert strip_code_fence('  {"a": 1}  ') == '{"a": 1}'
