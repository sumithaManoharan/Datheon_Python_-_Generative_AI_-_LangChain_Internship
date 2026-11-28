"""Split loaded documents into chunks that carry citation metadata.

Markdown files are split on headings first, so every chunk knows its section and
line span. Long sections are then split by LangChain's RecursiveCharacterTextSplitter.
PDFs are split page by page, so every chunk keeps its page number.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from docqa.errors import ConfigError
from docqa.loader import LoadedDocument

logger = logging.getLogger(__name__)

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FENCE_RE = re.compile(r"^\s*(`{3}|~{3})")
BREADCRUMB_SEP = " > "


@dataclass(frozen=True)
class Section:
    """A run of text under one heading; lines are 1-indexed and inclusive."""

    path: tuple[str, ...]
    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of text plus everything needed to cite it.

    `text` is what the user sees; `embed_text` adds a location header and is what
    gets embedded. PDFs set `page`; Markdown and text set `section` and line span.
    """

    source: Path
    text: str
    embed_text: str
    page: int | None = None
    section: str | None = None
    start_line: int | None = None
    end_line: int | None = None


def _has_content(text: str) -> bool:
    """True if the text has at least one letter or digit (not just '---' or '#')."""
    return any(ch.isalnum() for ch in text)


def _build_section(
    path: tuple[str, ...],
    lines: list[tuple[int, str]],
    has_heading: bool,
) -> Section | None:
    """Build a section from numbered lines; None when there is no body text."""
    start, end = 0, len(lines)
    while start < end and not lines[start][1].strip():
        start += 1
    while end > start and not lines[end - 1][1].strip():
        end -= 1
    kept = lines[start:end]
    # A heading with nothing under it carries no information, so it makes no chunk.
    body = kept[1:] if has_heading else kept
    if not _has_content("".join(text for _, text in body)):
        return None
    return Section(
        path=path,
        text="\n".join(text for _, text in kept),
        start_line=kept[0][0],
        end_line=kept[-1][0],
    )


def split_sections(text: str) -> list[Section]:
    """Split Markdown into sections on headings, tracking the heading path.

    A new heading replaces any heading of the same or deeper level in the path, so
    two consecutive `#` lines replace each other. `#` lines inside code fences are
    not headings.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    path: tuple[str, ...] = ()
    current: list[tuple[int, str]] = []
    has_heading = False
    in_fence = False

    # Split on "\n" only, so line numbers match what an editor shows.
    for line_no, line in enumerate(text.split("\n"), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING_RE.match(line)
        if match is None:
            current.append((line_no, line))
            continue
        section = _build_section(path, current, has_heading)
        if section is not None:
            sections.append(section)
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2)))
        path = tuple(title for _, title in stack)
        current = [(line_no, line)]
        has_heading = True

    section = _build_section(path, current, has_heading)
    if section is not None:
        sections.append(section)
    return sections


def _section_chunks(
    source: Path,
    section: Section,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Chunk]:
    """Split one section and compute each piece's line span."""
    breadcrumb = BREADCRUMB_SEP.join(section.path) or None
    header = source.name if breadcrumb is None else f"{source.name}{BREADCRUMB_SEP}{breadcrumb}"
    chunks: list[Chunk] = []
    cursor = 0
    for piece in splitter.split_text(section.text):
        if not _has_content(piece):
            continue
        if section.path and _is_heading_only(piece):
            continue
        # Pieces are substrings of the section, so find them to recover line numbers.
        # Searching forward from the previous start keeps repeated lines in order.
        offset = section.text.find(piece, cursor)
        if offset == -1:
            start_line, end_line = section.start_line, section.end_line
        else:
            start_line = section.start_line + section.text.count("\n", 0, offset)
            end_line = section.start_line + section.text.count("\n", 0, offset + len(piece))
            cursor = offset + 1
        chunks.append(
            Chunk(
                source=source,
                text=piece,
                embed_text=f"{header}\n\n{piece}",
                section=breadcrumb,
                start_line=start_line,
                end_line=end_line,
            )
        )
    return chunks


def _text_chunks(doc: LoadedDocument, splitter: RecursiveCharacterTextSplitter) -> list[Chunk]:
    """Chunk a Markdown or plain-text document."""
    if doc.source.suffix.lower() in MARKDOWN_SUFFIXES:
        sections = split_sections(doc.text)
    else:
        numbered = list(enumerate(doc.text.split("\n"), start=1))
        plain = _build_section((), numbered, has_heading=False)
        sections = [plain] if plain is not None else []
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(_section_chunks(doc.source, section, splitter))
    return chunks


def _page_chunks(doc: LoadedDocument, splitter: RecursiveCharacterTextSplitter) -> list[Chunk]:
    """Chunk one PDF page; every chunk keeps the page number."""
    header = f"{doc.source.name} (page {doc.page})"
    return [
        Chunk(
            source=doc.source,
            text=piece,
            embed_text=f"{header}\n\n{piece}",
            page=doc.page,
        )
        for piece in splitter.split_text(doc.text)
    ]


def chunk_documents(
    documents: Sequence[LoadedDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split loaded documents into chunks with citation metadata."""
    if chunk_size < 1:
        raise ConfigError(f"chunk_size must be at least 1, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ConfigError(f"chunk_overlap must be from 0 to chunk_size - 1, got {chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for doc in documents:
        if doc.page is None:
            chunks.extend(_text_chunks(doc, splitter))
        else:
            chunks.extend(_page_chunks(doc, splitter))
    logger.info("split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def _is_heading_only(piece: str) -> bool:
    """True if the piece is a single Markdown heading line and nothing else."""
    stripped = piece.strip()
    return "\n" not in stripped and HEADING_RE.match(stripped) is not None
