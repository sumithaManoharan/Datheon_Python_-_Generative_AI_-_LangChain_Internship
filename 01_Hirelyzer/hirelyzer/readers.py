"""Read a document (PDF, DOCX, TXT) or raw text into clean text, with input guards."""

import logging
from collections.abc import Callable
from pathlib import Path

from hirelyzer.errors import EmptyDocumentError, InputTooLargeError, UnsupportedFileError

logger = logging.getLogger(__name__)


def _read_txt(path: Path) -> str:
    """Read a plain text file; undecodable bytes become replacement characters."""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    """Extract markdown text from a PDF with pymupdf4llm."""
    # Imported here so the text-only path works without loading the PDF stack.
    import pymupdf4llm

    try:
        return pymupdf4llm.to_markdown(str(path))
    except Exception as exc:  # third-party parsers raise many different error types
        raise UnsupportedFileError(f"Cannot read PDF {path.name}: {exc}") from exc


def _read_docx(path: Path) -> str:
    """Extract paragraph and table text from a DOCX file."""
    import docx

    try:
        document = docx.Document(str(path))
    except Exception as exc:  # corrupt zip, missing parts, wrong format, and so on
        raise UnsupportedFileError(f"Cannot read DOCX {path.name}: {exc}") from exc

    lines = [paragraph.text for paragraph in document.paragraphs]
    # Many resumes put content in tables; each row becomes one "a | b | c" line.
    for table in document.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(lines)


# One reader per supported extension. Adding a format means adding one entry here.
READERS: dict[str, Callable[[Path], str]] = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".txt": _read_txt,
}


def prepare_text(text: str, source: str, max_chars: int) -> str:
    """Strip the text, reject it if empty, and truncate it (with a warning) if too long."""
    cleaned = text.strip()
    if not cleaned:
        raise EmptyDocumentError(f"No text found in {source}.")
    if len(cleaned) > max_chars:
        logger.warning(
            "%s has %d characters; truncating to the first %d", source, len(cleaned), max_chars
        )
        cleaned = cleaned[:max_chars]
    return cleaned


def read_document(path: Path, max_file_bytes: int, max_text_chars: int) -> str:
    """Return clean text from a file, or raise a typed error explaining why not."""
    if not path.is_file():
        raise UnsupportedFileError(f"File not found: {path}")

    reader = READERS.get(path.suffix.lower())
    if reader is None:
        supported = ", ".join(sorted(READERS))
        raise UnsupportedFileError(
            f"Unsupported file type {path.suffix!r} for {path.name}. Supported: {supported}"
        )

    size = path.stat().st_size
    if size == 0:
        raise EmptyDocumentError(f"{path.name} is empty.")
    if size > max_file_bytes:
        raise InputTooLargeError(
            f"{path.name} is {size / 1024 / 1024:.1f} MB; the limit is "
            f"{max_file_bytes / 1024 / 1024:.1f} MB."
        )

    return prepare_text(reader(path), path.name, max_text_chars)
