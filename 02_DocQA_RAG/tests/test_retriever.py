"""Tests for docqa.retriever."""

from pathlib import Path

import pytest
from docqa.chunking import Chunk
from docqa.errors import ConfigError
from docqa.retriever import Retriever
from docqa.store import VectorStore

A = Path("docs/a.md")


def _chunk(text: str, line: int) -> Chunk:
    return Chunk(
        source=A, text=text, embed_text=text, section="Doc", start_line=line, end_line=line
    )


@pytest.fixture
def store(tmp_path: Path, embedder) -> VectorStore:
    store = VectorStore(tmp_path / "chroma", embedder)
    store.ingest_file(
        A,
        "h1",
        [
            _chunk("alpha beta gamma", 1),
            _chunk("alpha delta", 2),
            _chunk("omega sigma tau", 3),
            _chunk("kappa lambda mu", 4),
        ],
    )
    return store


def test_returns_top_k_most_relevant_first(store):
    results = Retriever(store, top_k=2, relevance_threshold=0.0).retrieve("alpha beta gamma")
    assert len(results) == 2
    assert results[0].hit.text == "alpha beta gamma"
    assert results[0].relevance >= results[1].relevance


def test_relevance_is_one_minus_distance(store):
    result = Retriever(store, top_k=1, relevance_threshold=0.0).retrieve("alpha beta gamma")[0]
    assert result.relevance == pytest.approx(1.0 - result.hit.distance)
    assert result.relevance == pytest.approx(1.0, abs=1e-4)


def test_threshold_zero_keeps_every_hit(store):
    assert len(Retriever(store, top_k=4, relevance_threshold=0.0).retrieve("alpha")) == 4


def test_threshold_drops_weak_hits(store):
    results = Retriever(store, top_k=4, relevance_threshold=0.5).retrieve("alpha beta gamma")
    assert 1 <= len(results) < 4
    assert all(r.relevance >= 0.5 for r in results)


def test_unrelated_question_returns_nothing_above_a_strict_threshold(store):
    assert Retriever(store, top_k=4, relevance_threshold=0.9).retrieve("zzz qqq www") == []


def test_threshold_one_keeps_only_an_exact_match(store):
    results = Retriever(store, top_k=4, relevance_threshold=1.0).retrieve("alpha beta gamma")
    assert all(r.hit.text == "alpha beta gamma" for r in results)


def test_empty_store_returns_nothing(tmp_path: Path, embedder):
    empty = VectorStore(tmp_path / "chroma", embedder)
    assert Retriever(empty, top_k=3, relevance_threshold=0.0).retrieve("anything") == []


def test_top_k_larger_than_store_returns_everything(store):
    assert len(Retriever(store, top_k=20, relevance_threshold=0.0).retrieve("alpha")) == 4


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises(store, top_k: int):
    with pytest.raises(ConfigError):
        Retriever(store, top_k=top_k, relevance_threshold=0.0)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold_raises(store, threshold: float):
    with pytest.raises(ConfigError):
        Retriever(store, top_k=3, relevance_threshold=threshold)
