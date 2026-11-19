"""Orchestration: ingest a folder of documents and answer questions about them."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from docqa.answer import Answer, AnswerGenerator, load_system_prompt, validate_question
from docqa.chunking import Chunk, chunk_documents
from docqa.config import Settings
from docqa.embeddings import OpenAIEmbedder
from docqa.errors import DocQAError
from docqa.loader import load_directory
from docqa.retriever import Retriever
from docqa.store import VectorStore, content_key

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class IngestSummary:
    """Outcome of one ingest run. Any entry in `failures` makes the CLI exit non-zero."""

    embedded: int
    skipped: int
    failures: tuple[str, ...]
    total_chunks: int


class DocQAPipeline:
    """Ingest documents and answer questions. All parts are injected."""

    def __init__(
        self,
        store: VectorStore,
        retriever: Retriever,
        generator: AnswerGenerator,
        *,
        chunk_size: int,
        chunk_overlap: int,
        max_file_bytes: int,
        max_question_chars: int,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._generator = generator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_file_bytes = max_file_bytes
        self._max_question_chars = max_question_chars

    def ingest(self, source_dir: Path) -> IngestSummary:
        """Load, chunk and store every supported file in a folder.

        A file that cannot be read, or whose embedding or storage fails, is recorded
        and the rest continue. A missing folder, or one with no supported files, raises.
        """
        report = load_directory(source_dir, self._max_file_bytes)
        failures = [str(item) for item in report.failed]
        chunks = chunk_documents(report.documents, self._chunk_size, self._chunk_overlap)
        by_source: dict[Path, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            by_source[chunk.source].append(chunk)

        embedded = skipped = 0
        for source, group in by_source.items():
            try:
                key = content_key(source, self._chunk_size, self._chunk_overlap)
                stored = self._store.ingest_file(source, key, group)
            except DocQAError as exc:
                logger.error("could not ingest %s: %s", source, exc)
                failures.append(f"{source.name}: {exc}")
                continue
            if stored:
                embedded += 1
            else:
                skipped += 1
        logger.info(
            "ingest finished: %d embedded, %d unchanged, %d failed",
            embedded,
            skipped,
            len(failures),
        )
        return IngestSummary(embedded, skipped, tuple(failures), self._store.count())

    def ask(self, question: str) -> Answer:
        """Answer from the stored documents; refuses when nothing relevant is found."""
        question = validate_question(question, self._max_question_chars)
        return self._generator.answer(question, self._retriever.retrieve(question))


def build_pipeline(settings: Settings, client: OpenAI | None = None) -> DocQAPipeline:
    """Create the real pipeline from settings. Pass `client` to avoid needing an API key."""
    api_client = client or OpenAI(
        base_url=settings.base_url, api_key=settings.require_api_key(), max_retries=0
    )
    embedder = OpenAIEmbedder(
        api_client,
        settings.embedding_model,
        batch_size=settings.embed_batch_size,
        max_attempts=settings.max_attempts,
        base_delay=settings.retry_base_delay,
    )
    store = VectorStore(settings.store_dir, embedder)
    retriever = Retriever(store, settings.top_k, settings.relevance_threshold)
    generator = AnswerGenerator(
        api_client,
        settings.llm_model,
        load_system_prompt(),
        temperature=settings.temperature,
        max_question_chars=settings.max_question_chars,
        max_attempts=settings.max_attempts,
        base_delay=settings.retry_base_delay,
    )
    return DocQAPipeline(
        store,
        retriever,
        generator,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_file_bytes=settings.max_file_mb * BYTES_PER_MB,
        max_question_chars=settings.max_question_chars,
    )
