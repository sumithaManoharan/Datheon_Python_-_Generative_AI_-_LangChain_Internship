"""Tests for docqa.chunking."""

from pathlib import Path

import pytest
from docqa.chunking import chunk_documents, split_sections
from docqa.errors import ConfigError
from docqa.loader import load_file


def _load(tmp_path: Path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return load_file(path)


# ---------- split_sections ----------


def test_headings_split_sections_with_line_spans():
    text = "# Title\n\nIntro text.\n\n## Features\n\n- one\n- two\n\n## Pricing\n\nCosts money.\n"
    spans = [(s.path, s.start_line, s.end_line) for s in split_sections(text)]
    assert spans == [
        (("Title",), 1, 3),
        (("Title", "Features"), 5, 8),
        (("Title", "Pricing"), 10, 12),
    ]


def test_heading_without_body_yields_no_section():
    sections = split_sections("# A\n\n## B\n\nbody")
    assert len(sections) == 1
    assert sections[0].path == ("A", "B")
    assert (sections[0].start_line, sections[0].end_line) == (3, 5)


def test_consecutive_top_level_headings_replace_each_other():
    sections = split_sections("# HR Record\n\n# Maxine Thompson\n\n- Role: engineer")
    assert [s.path for s in sections] == [("Maxine Thompson",)]


def test_text_before_first_heading_has_empty_path():
    sections = split_sections("Preamble line\n\n# T\n\nbody")
    assert [s.path for s in sections] == [(), ("T",)]
    assert (sections[0].start_line, sections[0].end_line) == (1, 1)


def test_hash_inside_code_fence_is_not_a_heading():
    sections = split_sections("# T\n\n```\n# not a heading\n```\n")
    assert len(sections) == 1
    assert sections[0].end_line == 5


def test_hash_without_space_is_not_a_heading():
    sections = split_sections("#hashtag text\n")
    assert [s.path for s in sections] == [()]


# ---------- chunk_documents ----------


def test_long_section_is_split_with_correct_line_spans(tmp_path: Path):
    rows = [f"row {i:02d} " + "x" * 20 for i in range(1, 21)]
    text = "# Big\n\n" + "\n".join(rows) + "\n"
    chunks = chunk_documents(_load(tmp_path, "big.md", text), chunk_size=100, chunk_overlap=0)
    lines = text.split("\n")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 100
        assert chunk.text.split("\n")[0] == lines[chunk.start_line - 1]
        assert chunk.text.split("\n")[-1] == lines[chunk.end_line - 1]
    assert chunks[0].start_line == 3
    assert chunks[-1].end_line == 22


def test_overlap_makes_neighbouring_spans_share_lines(tmp_path: Path):
    rows = [f"row {i:02d} " + "x" * 20 for i in range(1, 21)]
    text = "# Big\n\n" + "\n".join(rows) + "\n"
    chunks = chunk_documents(_load(tmp_path, "big.md", text), chunk_size=100, chunk_overlap=30)
    assert all(len(c.text) <= 100 for c in chunks)
    assert any(b.start_line <= a.end_line for a, b in zip(chunks, chunks[1:], strict=False))


def test_embed_text_starts_with_file_and_section_path(tmp_path: Path):
    text = "# Acme Contract\n\n## Features\n\n- fast\n"
    chunks = chunk_documents(_load(tmp_path, "acme.md", text), chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.section == "Acme Contract > Features"
    assert chunk.embed_text.startswith("acme.md > Acme Contract > Features\n\n")
    assert chunk.text in chunk.embed_text


def test_pdf_chunks_carry_page_and_no_line_span(make_pdf):
    docs = load_file(make_pdf(["alpha", "", "gamma"]))
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert [c.page for c in chunks] == [1, 3]
    assert all(c.section is None and c.start_line is None for c in chunks)
    assert "(page 3)" in chunks[1].embed_text


def test_plain_text_is_one_section_with_line_span(tmp_path: Path):
    chunks = chunk_documents(
        _load(tmp_path, "a.txt", "one\ntwo\nthree"), chunk_size=500, chunk_overlap=0
    )
    assert len(chunks) == 1
    assert chunks[0].section is None
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 3)


def test_no_documents_gives_no_chunks():
    assert chunk_documents([], chunk_size=100, chunk_overlap=10) == []


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_size_or_overlap_raises(size: int, overlap: int):
    with pytest.raises(ConfigError):
        chunk_documents([], chunk_size=size, chunk_overlap=overlap)


def test_heading_followed_only_by_separator_yields_no_section():
    assert split_sections("# A\n\n---\n") == []


def test_separator_only_document_yields_no_chunks(tmp_path: Path):
    docs = _load(tmp_path, "a.md", "# A\n\n---\n")
    assert chunk_documents(docs, chunk_size=500, chunk_overlap=50) == []


def test_oversized_body_does_not_leave_a_heading_only_chunk(tmp_path: Path):
    rows = [f"row {i:02d} " + "x" * 20 for i in range(1, 11)]
    text = "# T\n\n## Terms\n\n" + "\n".join(rows) + "\n"
    chunks = chunk_documents(_load(tmp_path, "t.md", text), chunk_size=100, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(c.text.strip() != "## Terms" for c in chunks)
    assert all(c.section == "T > Terms" for c in chunks)
