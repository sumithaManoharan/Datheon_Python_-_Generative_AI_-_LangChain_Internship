"""Read .pdf, .md and .txt files into text documents that remember their origin."""

import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from docqa.errors import (
    DocQAError,
    DocumentReadError,
    EmptyDocumentError,
    InputTooLargeError,
    UnsupportedFileError,
)

logger = logging.getLogger(__name__)

PDF_SUFFIX = ".pdf"
TEXT_SUFFIXES = frozenset({".md", ".txt"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {PDF_SUFFIX}
DEFAULT_MAX_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class LoadedDocument:
    """Text from one text file, or from one page of a PDF."""

    source: Path
    text: str
    page: int | None = None  # 1-indexed for PDFs, None for text files


@dataclass
class LoadReport:
    """Outcome of loading a directory: the documents read and the files skipped."""

    documents: list[LoadedDocument] = field(default_factory=list)
    failed: dict[Path, str] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Apply NFKC and unify line endings so matching and line numbers are stable.

    NFKC turns PDF ligatures such as U+FB00 back into plain letters ("ff").
    """
    folded = unicodedata.normalize("NFKC", text)
    return folded.replace("\r\n", "\n").replace("\r", "\n")


def load_file(path: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> list[LoadedDocument]:
    """Read one file into non-empty documents (one per PDF page, or one per text file).

    Raises:
        DocumentReadError: missing, unreadable, corrupt or encrypted file.
        UnsupportedFileError: the extension is not .pdf, .md or .txt.
        InputTooLargeError: the file is bigger than ``max_bytes``.
        EmptyDocumentError: the file has no extractable text.
    """
    if not path.is_file():
        raise DocumentReadError(f"{path} is not an existing file")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileError(f"{path.name}: unsupported file type '{suffix}'")
    size = path.stat().st_size
    if size > max_bytes:
        raise InputTooLargeError(f"{path.name} is {size} bytes; limit is {max_bytes}")

    documents = _read_pdf(path) if suffix == PDF_SUFFIX else _read_text(path)
    if not documents:
        raise EmptyDocumentError(f"{path.name} has no extractable text")
    return documents


def _read_text(path: Path) -> list[LoadedDocument]:
    """Read a UTF-8 text file; returns an empty list if it is only whitespace."""
    try:
        # utf-8-sig silently drops a leading byte-order mark if there is one.
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise DocumentReadError(f"cannot read {path.name}: {exc}") from exc
    text = normalize_text(raw)
    return [LoadedDocument(path, text)] if text.strip() else []


def _read_pdf(path: Path) -> list[LoadedDocument]:
    """Read a PDF page by page, skipping pages with no text but keeping numbering."""
    documents: list[LoadedDocument] = []
    try:
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise DocumentReadError(f"{path.name} is encrypted")
            for page in pdf:
                text = normalize_text(page.get_text())
                if text.strip():
                    documents.append(LoadedDocument(path, text, page.number + 1))
    except RuntimeError as exc:  # pymupdf raises RuntimeError subclasses for bad files
        raise DocumentReadError(f"cannot open {path.name}: {exc}") from exc
    return documents


def load_directory(directory: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> LoadReport:
    """Load every supported file under ``directory``; one bad file never stops the rest.

    Raises:
        DocumentReadError: ``directory`` is missing or holds no supported files.
    """
    if not directory.is_dir():
        raise DocumentReadError(f"directory not found: {directory}")
    files = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise DocumentReadError(f"no .pdf, .md or .txt files under {directory}")

    report = LoadReport()
    for path in files:
        try:
            report.documents.extend(load_file(path, max_bytes))
        except DocQAError as exc:
            logger.warning("skipped %s: %s", path.name, exc)
            report.failed[path] = str(exc)
    logger.info(
        "loaded %d documents from %d files, skipped %d",
        len(report.documents),
        len(files) - len(report.failed),
        len(report.failed),
    )
    return report
