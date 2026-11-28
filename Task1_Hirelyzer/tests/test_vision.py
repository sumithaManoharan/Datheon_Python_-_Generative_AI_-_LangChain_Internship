"""Tests for image loading and the vision branch of the extractor."""

import base64
from pathlib import Path
from typing import Any

import pytest
from hirelyzer.errors import EmptyDocumentError, InputTooLargeError, UnsupportedFileError
from hirelyzer.extractor import ResumeExtractor
from hirelyzer.prompts import PromptBuilder
from hirelyzer.readers import SUPPORTED_EXTENSIONS, load_document
from hirelyzer.vision import image_to_data_url

NON_RESUME = '{"is_resume": false}'
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake image body"
MAX_BYTES = 1_000_000


def write_image(directory: Path, name: str, content: bytes = PNG_BYTES) -> Path:
    """Create a file with the given bytes."""
    path = directory / name
    path.write_bytes(content)
    return path


def test_png_becomes_a_base64_data_url(tmp_path: Path) -> None:
    url = image_to_data_url(write_image(tmp_path, "a.png"), MAX_BYTES)

    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    assert base64.b64decode(url[len(prefix) :]) == PNG_BYTES


@pytest.mark.parametrize("name", ["a.jpg", "a.JPEG", "A.Jpg"])
def test_jpeg_extensions_use_the_jpeg_mime_type(tmp_path: Path, name: str) -> None:
    url = image_to_data_url(write_image(tmp_path, name), MAX_BYTES)

    assert url.startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize(
    ("name", "content", "error"),
    [
        ("a.gif", b"x", UnsupportedFileError),
        ("empty.png", b"", EmptyDocumentError),
    ],
)
def test_bad_images_raise_typed_errors(
    tmp_path: Path, name: str, content: bytes, error: type[Exception]
) -> None:
    with pytest.raises(error):
        image_to_data_url(write_image(tmp_path, name, content), MAX_BYTES)


def test_oversized_image_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InputTooLargeError):
        image_to_data_url(write_image(tmp_path, "big.png"), 5)


def test_load_document_marks_images(tmp_path: Path) -> None:
    image = load_document(write_image(tmp_path, "a.png"), MAX_BYTES, 1000)
    text_file = tmp_path / "a.txt"
    text_file.write_text("hello", encoding="utf-8")
    text = load_document(text_file, MAX_BYTES, 1000)

    assert image.is_image is True
    assert (text.is_image, text.content) == (False, "hello")


def test_load_document_reports_a_missing_image(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileError, match="not found"):
        load_document(tmp_path / "nope.png", MAX_BYTES, 1000)


def test_supported_extensions_include_images_and_documents() -> None:
    assert {".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png"} <= SUPPORTED_EXTENSIONS


def test_vision_messages_carry_the_image(prompt_builder: PromptBuilder) -> None:
    system, user = prompt_builder.build_vision_messages("data:image/png;base64,AAA")

    assert system["role"] == "system"
    assert user["content"][1]["image_url"]["url"] == "data:image/png;base64,AAA"


def test_image_extraction_uses_the_vision_client(
    prompt_builder: PromptBuilder, make_llm: Any
) -> None:
    text_llm = make_llm([])
    vision_llm = make_llm([NON_RESUME])
    extractor = ResumeExtractor(text_llm, prompt_builder, 2, vision_llm=vision_llm)

    result = extractor.extract_image("data:image/png;base64,AAA", source="scan.png")

    assert result.status == "success"
    assert len(vision_llm.calls) == 1
    assert text_llm.calls == []


def test_image_extraction_without_a_vision_client_fails_cleanly(
    prompt_builder: PromptBuilder, make_llm: Any
) -> None:
    extractor = ResumeExtractor(make_llm([]), prompt_builder, 2)

    result = extractor.extract_image("data:image/png;base64,AAA", source="scan.png")

    assert result.status == "failed"
    assert "vision" in result.errors[0]
