"""Tests for docqa.pipeline."""

from pathlib import Path

import pytest
from docqa.answer import AnswerGenerator
from docqa.config import load_settings
from docqa.errors import ConfigError, DocQAError, EmbeddingError, InvalidQuestionError
from docqa.pipeline import DocQAPipeline, build_pipeline
from docqa.retriever import Retriever
from docqa.store import VectorStore


@pytest.fixture
def store(tmp_path: Path, embedder) -> VectorStore:
    return VectorStore(tmp_path / "chroma", embedder)


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    alpha = "# Alpha\n\nalpha beta gamma facts about alpha.\n"
    beta = "# Beta\n\ndelta epsilon zeta facts about delta.\n"
    (folder / "a.md").write_text(alpha, encoding="utf-8")
    (folder / "b.md").write_text(beta, encoding="utf-8")
    return folder


def _make(store, llm, *, chunk_size=800, chunk_overlap=100, threshold=0.0) -> DocQAPipeline:
    retriever = Retriever(store, top_k=3, relevance_threshold=threshold)
    generator = AnswerGenerator(
        llm, "test-model", "SYSTEM", max_question_chars=100, sleep=lambda _seconds: None
    )
    return DocQAPipeline(
        store,
        retriever,
        generator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_file_bytes=1_000_000,
        max_question_chars=100,
    )


# ---------- ingest ----------


def test_ingest_embeds_every_file(store, llm, docs):
    summary = _make(store, llm).ingest(docs)
    assert (summary.embedded, summary.skipped) == (2, 0)
    assert summary.failures == ()
    assert summary.total_chunks == store.count() >= 2


def test_second_ingest_skips_unchanged_files_without_embedding(store, llm, embedder, docs):
    pipeline = _make(store, llm)
    pipeline.ingest(docs)
    calls = len(embedder.document_calls)
    summary = pipeline.ingest(docs)
    assert (summary.embedded, summary.skipped) == (0, 2)
    assert len(embedder.document_calls) == calls


def test_only_the_changed_file_is_re_embedded(store, llm, docs):
    pipeline = _make(store, llm)
    pipeline.ingest(docs)
    (docs / "a.md").write_text("# Alpha\n\nnew text about omega.\n", encoding="utf-8")
    summary = pipeline.ingest(docs)
    assert (summary.embedded, summary.skipped) == (1, 1)


def test_changed_chunk_settings_trigger_re_ingestion(store, llm, docs):
    _make(store, llm, chunk_size=800).ingest(docs)
    summary = _make(store, llm, chunk_size=300).ingest(docs)
    assert (summary.embedded, summary.skipped) == (2, 0)


def test_one_unreadable_file_does_not_stop_the_batch(store, llm, docs):
    (docs / "blank.md").write_text("  \n", encoding="utf-8")
    summary = _make(store, llm).ingest(docs)
    assert summary.embedded == 2
    assert len(summary.failures) == 1


def test_embedding_failure_is_recorded_not_raised_and_recovers(store, llm, embedder, docs):
    pipeline = _make(store, llm)
    embedder.fail_with = EmbeddingError("boom")
    summary = pipeline.ingest(docs)
    assert summary.embedded == 0
    assert len(summary.failures) == 2
    assert all("boom" in item for item in summary.failures)
    assert store.count() == 0
    embedder.fail_with = None
    assert pipeline.ingest(docs).embedded == 2


def test_missing_folder_raises(store, llm, tmp_path: Path):
    with pytest.raises(DocQAError):
        _make(store, llm).ingest(tmp_path / "nope")


# ---------- ask ----------


def test_ask_returns_a_cited_answer_from_the_stored_documents(store, llm, docs):
    pipeline = _make(store, llm)
    pipeline.ingest(docs)
    llm.reply = "Alpha facts [1]."
    answer = pipeline.ask("alpha beta gamma")
    assert not answer.refused
    assert answer.citations[0].source == "a.md"
    assert len(llm.calls) == 1
    assert "alpha beta gamma" in llm.calls[0]["messages"][1]["content"]


def test_ask_on_an_empty_store_refuses_without_calling_the_llm(store, llm):
    answer = _make(store, llm).ask("anything at all?")
    assert answer.refused
    assert llm.calls == []


def test_unrelated_question_is_refused_by_a_strict_threshold(store, llm, docs):
    pipeline = _make(store, llm, threshold=0.99)
    pipeline.ingest(docs)
    answer = pipeline.ask("completely unrelated words here")
    assert answer.refused
    assert llm.calls == []


@pytest.mark.parametrize("question", ["", "   ", "x" * 101])
def test_invalid_question_raises_before_any_call(store, llm, docs, question: str):
    pipeline = _make(store, llm)
    pipeline.ingest(docs)
    with pytest.raises(InvalidQuestionError):
        pipeline.ask(question)
    assert llm.calls == []


# ---------- build_pipeline ----------


def test_build_pipeline_without_an_api_key_raises_config_error():
    with pytest.raises(ConfigError):
        build_pipeline(load_settings({}))


def test_build_pipeline_with_an_injected_client_needs_no_key(tmp_path: Path, llm):
    settings = load_settings({"DOCQA_STORE_DIR": str(tmp_path / "chroma")})
    answer = build_pipeline(settings, client=llm).ask("anything?")
    assert answer.refused
    assert llm.calls == []
