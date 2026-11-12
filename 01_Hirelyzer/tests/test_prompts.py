"""Tests for prompt loading, validation and rendering."""

import json
from pathlib import Path

import pytest
from hirelyzer.errors import ConfigError
from hirelyzer.prompts import (
    FewShotExample,
    PromptBuilder,
    format_examples,
    load_examples,
    load_template,
    render_template,
    resume_schema_json,
)

VALID_TEMPLATE = "A {schema} B {examples} C {extra_rules}"


def test_system_prompt_contains_schema_examples_and_extra_rules() -> None:
    prompt = PromptBuilder.from_files(extra_rules="Always answer politely.").system_prompt

    assert '"is_resume"' in prompt  # from the schema
    assert "Jordan Avery" in prompt  # from the few-shot examples
    assert "Always answer politely." in prompt
    for placeholder in ("{schema}", "{examples}", "{extra_rules}"):
        assert placeholder not in prompt


def test_build_messages_returns_system_then_user(prompt_builder: PromptBuilder) -> None:
    messages = prompt_builder.build_messages("resume text here")

    assert [m["role"] for m in messages] == ["system", "user"]
    assert "resume text here" in messages[1]["content"]


def test_inserted_text_is_never_rescanned_for_placeholders() -> None:
    values = {"schema": "{examples}", "examples": "x", "extra_rules": ""}

    assert render_template(VALID_TEMPLATE, values) == "A {examples} B x C"


def test_schema_json_is_valid_json_with_the_resume_fields() -> None:
    assert "is_resume" in json.loads(resume_schema_json())["properties"]


def test_real_examples_cover_a_resume_and_a_non_resume() -> None:
    flags = {example.output["is_resume"] for example in load_examples()}

    assert flags == {True, False}


def test_format_examples_numbers_each_example() -> None:
    examples = [FewShotExample("doc one", {"is_resume": False})] * 2

    text = format_examples(examples)

    assert "EXAMPLE 1" in text
    assert "EXAMPLE 2" in text


def test_missing_template_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Cannot read"):
        load_template(tmp_path)


@pytest.mark.parametrize(
    "template",
    ["no placeholders", "{schema} {examples}", "{schema} {schema} {examples} {extra_rules}"],
    ids=["none", "missing_one", "duplicated"],
)
def test_template_placeholders_must_appear_exactly_once(tmp_path: Path, template: str) -> None:
    (tmp_path / "system.txt").write_text(template, encoding="utf-8")

    with pytest.raises(ConfigError, match="placeholder"):
        load_template(tmp_path)


def test_valid_template_loads(tmp_path: Path) -> None:
    (tmp_path / "system.txt").write_text(VALID_TEMPLATE, encoding="utf-8")

    assert load_template(tmp_path) == VALID_TEMPLATE


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "Cannot read"),
        ("[]", "non-empty"),
        ('{"a": 1}', "non-empty"),
        ('[{"output": {"is_resume": false}}]', "invalid"),
        ('[{"document": "x", "output": {"is_resume": true, "bogus": 1}}]', "invalid"),
        ('[{"document": "  ", "output": {"is_resume": false}}]', "invalid"),
    ],
    ids=["empty_file", "empty_list", "not_a_list", "no_document", "bad_output", "blank_document"],
)
def test_broken_examples_files_fail_fast(tmp_path: Path, content: str, message: str) -> None:
    (tmp_path / "examples.json").write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_examples(tmp_path)


def test_valid_examples_file_loads(tmp_path: Path) -> None:
    content = '[{"document": "an invoice", "output": {"is_resume": false}}]'
    (tmp_path / "examples.json").write_text(content, encoding="utf-8")

    (example,) = load_examples(tmp_path)

    assert example.document == "an invoice"
