"""Tests for docqa.loader."""

from pathlib import Path

import pytest
from docqa.errors import (
    DocQAError,
    DocumentReadError,
    EmptyDocumentError,
    InputTooLargeError,
    UnsupportedFileError,
)
from docqa.loader import load_directory, load_file, normalize_text

# ---------- happy paths ----------


def test_pdf_pages_are_one_indexed(make_pdf):
    docs = load_file(make_pdf(["alpha", "beta"]))
    assert [d.page for d in docs] == [1, 2]
    assert "alpha" in docs[0].text
    assert "beta" in docs[1].text


def test_blank_pdf_page_is_skipped_but_numbering_kept(make_pdf):
    docs = load_file(make_pdf(["alpha", "", "gamma"]))
    assert [d.page for d in docs] == [1, 3]


def test_markdown_has_no_page(tmp_path: Path):
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nSome text.\n", encoding="utf-8")
    docs = load_file(path)
    assert len(docs) == 1
    assert docs[0].page is None
    assert docs[0].source.name == "note.md"


def test_txt_file_is_loaded(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("plain text", encoding="utf-8")
    assert load_file(path)[0].text.strip() == "plain text"


def test_crlf_line_endings_are_normalized(tmp_path: Path):
    path = tmp_path / "crlf.md"
    path.write_bytes(b"line one\r\nline two\r\n")
    assert "\r" not in load_file(path)[0].text


def test_utf8_bom_is_stripped(tmp_path: Path):
    path = tmp_path / "bom.md"
    path.write_bytes(b"\xef\xbb\xbfHello")
    assert load_file(path)[0].text.startswith("Hello")


def test_normalize_text_expands_ligatures():
    assert normalize_text("E\ufb00ective") == "Effective"


# ---------- error cases ----------


def test_zero_byte_pdf_raises(tmp_path: Path):
    path = tmp_path / "x.pdf"
    path.write_bytes(b"")
    with pytest.raises(DocQAError):
        load_file(path)


def test_corrupt_pdf_raises_read_error(tmp_path: Path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"this is not a pdf at all")
    with pytest.raises(DocumentReadError):
        load_file(path)


def test_encrypted_pdf_raises_read_error(make_pdf):
    path = make_pdf(["secret"], password="pw")
    with pytest.raises(DocumentReadError):
        load_file(path)


def test_pdf_without_text_layer_raises_empty(make_pdf):
    path = make_pdf(["", ""])
    with pytest.raises(EmptyDocumentError):
        load_file(path)


def test_unsupported_extension_raises(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(UnsupportedFileError):
        load_file(path)


def test_blank_markdown_raises_empty(tmp_path: Path):
    path = tmp_path / "blank.md"
    path.write_text("  \n\n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_file(path)


def test_missing_file_raises_read_error(tmp_path: Path):
    with pytest.raises(DocumentReadError):
        load_file(tmp_path / "nope.md")


def test_non_utf8_text_raises_read_error(tmp_path: Path):
    path = tmp_path / "latin.txt"
    path.write_bytes(b"caf\xe9 \xff\xfe")
    with pytest.raises(DocumentReadError):
        load_file(path)


def test_oversized_file_raises(tmp_path: Path):
    path = tmp_path / "big.md"
    path.write_text("x" * 100, encoding="utf-8")
    with pytest.raises(InputTooLargeError):
        load_file(path, max_bytes=10)


# ---------- load_directory ----------


def test_directory_isolates_failures(tmp_path: Path):
    (tmp_path / "good.md").write_text("# Good\n\ntext", encoding="utf-8")
    (tmp_path / "blank.md").write_text("   ", encoding="utf-8")
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    report = load_directory(tmp_path)
    assert [d.source.name for d in report.documents] == ["good.md"]
    assert len(report.failed) == 2


def test_directory_ignores_unsupported_files(tmp_path: Path):
    (tmp_path / "good.md").write_text("text", encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b", encoding="utf-8")
    report = load_directory(tmp_path)
    assert len(report.documents) == 1


def test_missing_directory_raises(tmp_path: Path):
    with pytest.raises(DocQAError):
        load_directory(tmp_path / "nope")


def test_directory_with_no_supported_files_raises(tmp_path: Path):
    (tmp_path / "data.csv").write_text("a,b", encoding="utf-8")
    with pytest.raises(DocQAError):
        load_directory(tmp_path)
