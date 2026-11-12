"""Tests for file reading and the input guards."""

import logging
from pathlib import Path

import docx
import pymupdf
import pytest
from hirelyzer.errors import EmptyDocumentError, InputTooLargeError, UnsupportedFileError
from hirelyzer.readers import READERS, prepare_text, read_document

MAX_BYTES = 1_000_000
MAX_CHARS = 60_000


def read(path: Path, max_bytes: int = MAX_BYTES, max_chars: int = MAX_CHARS) -> str:
    """Read a document with generous default limits."""
    return read_document(path, max_bytes, max_chars)


def write_pdf(path: Path, text: str) -> Path:
    """Create a one-page PDF containing the text."""
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), text)
    pdf.save(str(path))
    return path


def write_docx(path: Path, text: str, table_row: tuple[str, ...] = ()) -> Path:
    """Create a DOCX with one paragraph and, optionally, a one-row table."""
    document = docx.Document()
    document.add_paragraph(text)
    if table_row:
        table = document.add_table(rows=1, cols=len(table_row))
        for cell, value in zip(table.rows[0].cells, table_row, strict=True):
            cell.text = value
    document.save(str(path))
    return path


def test_supported_extensions() -> None:
    assert set(READERS) == {".pdf", ".docx", ".txt"}


def test_reads_txt(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello world", encoding="utf-8")

    assert read(path) == "hello world"


def test_extension_check_ignores_case(tmp_path: Path) -> None:
    path = tmp_path / "A.TXT"
    path.write_text("hello", encoding="utf-8")

    assert read(path) == "hello"


def test_reads_unicode_txt(tmp_path: Path) -> None:
    path = tmp_path / "u.txt"
    path.write_text("Zoë Müller", encoding="utf-8")

    assert read(path) == "Zoë Müller"


def test_undecodable_bytes_do_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "odd.txt"
    path.write_bytes(b"abc\xff\xfe")

    assert read(path).startswith("abc")


def test_reads_pdf(tmp_path: Path) -> None:
    assert "Hello resume" in read(write_pdf(tmp_path / "ok.pdf", "Hello resume"))


def test_reads_docx_paragraphs_and_tables(tmp_path: Path) -> None:
    path = write_docx(tmp_path / "ok.docx", "Hello resume", table_row=("Skills", "Python"))

    text = read(path)

    assert "Hello resume" in text
    assert "Skills | Python" in text


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileError, match="not found"):
        read(tmp_path / "nope.txt")


def test_directory_is_not_a_file(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileError):
        read(tmp_path)


@pytest.mark.parametrize(
    ("name", "content", "error"),
    [
        ("empty.txt", b"", EmptyDocumentError),
        ("blank.txt", b"  \n  ", EmptyDocumentError),
        ("bad.xyz", b"x", UnsupportedFileError),
        ("photo.jpg", b"x", UnsupportedFileError),
        ("corrupt.pdf", b"not a pdf", UnsupportedFileError),
        ("corrupt.docx", b"not a docx", UnsupportedFileError),
    ],
)
def test_bad_files_raise_typed_errors(
    tmp_path: Path, name: str, content: bytes, error: type[Exception]
) -> None:
    path = tmp_path / name
    path.write_bytes(content)

    with pytest.raises(error):
        read(path)


def test_oversized_file_is_rejected_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "big.txt"
    path.write_text("x" * 5000)

    with pytest.raises(InputTooLargeError, match="limit"):
        read(path, max_bytes=1000)


def test_long_text_is_truncated_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "long.txt"
    path.write_text("x" * 50)

    with caplog.at_level(logging.WARNING):
        text = read(path, max_chars=10)

    assert len(text) == 10
    assert "truncating" in caplog.text


def test_prepare_text_strips_whitespace() -> None:
    assert prepare_text("  hi  \n", "demo", 100) == "hi"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_prepare_text_rejects_blank_input(text: str) -> None:
    with pytest.raises(EmptyDocumentError):
        prepare_text(text, "demo", 100)
