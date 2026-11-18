"""Tests for docqa.store."""

import copy
from pathlib import Path

import pytest
from docqa.chunking import Chunk
from docqa.errors import DocumentReadError, EmbeddingError, StoreError
from docqa.store import VectorStore, file_sha256

A = Path("docs/a.md")
B = Path("docs/b.md")


def _chunk(source: Path, text: str, **kwargs) -> Chunk:
    return Chunk(source=source, text=text, embed_text=text, **kwargs)


@pytest.fixture
def store(tmp_path: Path, embedder) -> VectorStore:
    return VectorStore(tmp_path / "chroma", embedder)


def test_ingest_stores_chunks(store):
    assert store.ingest_file(A, "h1", [_chunk(A, "alpha beta"), _chunk(A, "gamma delta")])
    assert store.count() == 2
    assert store.known_hash(A) == "h1"


def test_unknown_file_has_no_hash(store):
    assert store.known_hash(A) is None


def test_unchanged_hash_is_skipped_without_embedding(store, embedder):
    chunks = [_chunk(A, "alpha beta"), _chunk(A, "gamma delta")]
    store.ingest_file(A, "h1", chunks)
    calls = len(embedder.document_calls)
    assert store.ingest_file(A, "h1", chunks) is False
    assert len(embedder.document_calls) == calls
    assert store.count() == 2


def test_changed_hash_replaces_old_chunks(store):
    store.ingest_file(A, "h1", [_chunk(A, "alpha beta"), _chunk(A, "gamma delta")])
    store.ingest_file(A, "h2", [_chunk(A, "omega")])
    assert store.count() == 1
    assert store.known_hash(A) == "h2"
    assert [hit.text for hit in store.search("alpha", k=5)] == ["omega"]


def test_failed_embedding_keeps_old_data(store, embedder):
    store.ingest_file(A, "h1", [_chunk(A, "alpha beta")])
    embedder.fail_with = EmbeddingError("boom")
    with pytest.raises(EmbeddingError):
        store.ingest_file(A, "h2", [_chunk(A, "omega")])
    embedder.fail_with = None
    assert store.count() == 1
    assert store.known_hash(A) == "h1"


def test_empty_chunk_list_removes_old_chunks(store):
    store.ingest_file(A, "h1", [_chunk(A, "alpha beta")])
    assert store.ingest_file(A, "h2", []) is False
    assert store.count() == 0


def test_files_do_not_overwrite_each_other(store):
    store.ingest_file(A, "h1", [_chunk(A, "alpha")])
    store.ingest_file(B, "h9", [_chunk(B, "beta")])
    store.ingest_file(A, "h2", [_chunk(A, "gamma")])
    assert store.count() == 2
    assert store.known_hash(B) == "h9"


def test_persisted_store_reloads(tmp_path: Path, embedder):
    path = tmp_path / "chroma"
    first = VectorStore(path, embedder)
    first.ingest_file(A, "h1", [_chunk(A, "alpha beta"), _chunk(A, "gamma delta")])
    del first
    reopened = VectorStore(path, embedder)
    assert reopened.count() == 2
    assert reopened.known_hash(A) == "h1"
    assert "alpha" in reopened.search("alpha", k=1)[0].text


def test_store_built_with_another_embedder_is_refused(tmp_path: Path, embedder):
    path = tmp_path / "chroma"
    VectorStore(path, embedder)
    other = copy.copy(embedder)
    other.name = "other-model"
    with pytest.raises(StoreError):
        VectorStore(path, other)


def test_search_returns_nearest_first_with_citation_fields(store):
    store.ingest_file(
        A,
        "h1",
        [
            _chunk(A, "alpha beta", section="Doc > One", start_line=3, end_line=5),
            _chunk(A, "gamma delta", section="Doc > Two", start_line=7, end_line=9),
        ],
    )
    hits = store.search("alpha", k=2)
    assert hits[0].text == "alpha beta"
    assert (hits[0].section, hits[0].start_line, hits[0].end_line) == ("Doc > One", 3, 5)
    assert hits[0].page is None
    assert hits[0].source == str(A)
    assert hits[0].distance <= hits[1].distance


def test_pdf_chunk_keeps_page_and_has_no_section(store):
    pdf = Path("docs/report.pdf")
    store.ingest_file(pdf, "h1", [_chunk(pdf, "epsilon zeta", page=4)])
    hit = store.search("epsilon", k=1)[0]
    assert hit.page == 4
    assert hit.section is None
    assert hit.start_line is None


def test_search_distance_is_cosine(store, embedder):
    chunk = _chunk(A, "alpha beta")
    store.ingest_file(A, "h1", [chunk])
    query = embedder.embed_query("alpha gamma")
    doc = embedder.embed_documents([chunk.embed_text])[0]
    expected = 1 - sum(x * y for x, y in zip(query, doc, strict=True))
    assert store.search("alpha gamma", k=1)[0].distance == pytest.approx(expected, abs=1e-4)


def test_search_k_larger_than_count_returns_everything(store):
    store.ingest_file(A, "h1", [_chunk(A, "alpha"), _chunk(A, "beta")])
    assert len(store.search("alpha", k=10)) == 2


def test_search_empty_store_returns_nothing(store):
    assert store.search("anything", k=3) == []


def test_search_rejects_non_positive_k(store):
    with pytest.raises(ValueError):
        store.search("alpha", k=0)


def test_file_sha256_changes_with_content(tmp_path: Path):
    path = tmp_path / "a.md"
    path.write_text("one", encoding="utf-8")
    first = file_sha256(path)
    assert file_sha256(path) == first
    path.write_text("two", encoding="utf-8")
    assert file_sha256(path) != first


def test_file_sha256_missing_file_raises(tmp_path: Path):
    with pytest.raises(DocumentReadError):
        file_sha256(tmp_path / "nope.md")
