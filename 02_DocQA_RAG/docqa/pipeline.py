"""Orchestration: ingest a folder of documents and answer questions about them."""

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from docqa.answer import (
    Answer,
    AnswerGenerator,
    AnswerStream,
    Turn,
    load_system_prompt,
    validate_question,
)
from docqa.chunking import Chunk, chunk_documents
from docqa.config import Settings
from docqa.conversation import ConversationMemory
from docqa.embeddings import OpenAIEmbedder
from docqa.errors import DocQAError
from docqa.loader import load_directory
from docqa.retriever import RetrievedChunk, Retriever
from docqa.store import VectorStore, content_key
from docqa.telemetry import QueryRecord, TelemetryLog, telemetry_path, utc_timestamp

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024
TIMING_DECIMALS = 4


@dataclass(frozen=True)
class IngestSummary:
    """Outcome of one ingest run. Any entry in `failures` makes the CLI exit non-zero."""

    embedded: int
    skipped: int
    failures: tuple[str, ...]
    total_chunks: int


class ChatStream:
    """A chat answer arriving in pieces; memory and telemetry are updated when it ends.

    Iterate once. If the reader stops early or the stream fails, nothing is recorded.
    """

    def __init__(self, stream: AnswerStream, on_complete: Callable[[Answer], None]) -> None:
        self._stream = stream
        self._on_complete = on_complete

    def __iter__(self) -> Iterator[str]:
        yield from self._stream
        self._on_complete(self._stream.answer)

    @property
    def answer(self) -> Answer:
        """The finished answer; only available after the stream has been read to the end."""
        return self._stream.answer


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
        telemetry: TelemetryLog | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._generator = generator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_file_bytes = max_file_bytes
        self._max_question_chars = max_question_chars
        self._telemetry = telemetry
        self._clock = clock

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
        """Answer one question on its own; no conversation memory is used."""
        return self._answer_question(validate_question(question, self._max_question_chars), ())

    def chat(self, question: str, memory: ConversationMemory) -> Answer:
        """Answer inside a conversation: earlier turns resolve references like 'he' or 'it'.

        The exchange is added to `memory` once answered; an invalid or failed question is not.
        """
        question = validate_question(question, self._max_question_chars)
        answer = self._answer_question(question, memory.turns)
        memory.add(question, answer.text)
        return answer

    def chat_stream(self, question: str, memory: ConversationMemory) -> ChatStream:
        """Like `chat`, but the answer streams in pieces.

        The follow-up is rewritten and the passages are retrieved before this returns, so
        those errors surface here. The exchange is added to `memory` and the telemetry record
        is written only once the stream has been read to the end.
        """
        question = validate_question(question, self._max_question_chars)
        history = memory.turns
        started = self._clock()
        search_text, retrieved = self._search(question, history)
        retrieved_at = self._clock()
        stream = self._generator.answer_stream(search_text, retrieved, history)

        def on_complete(answer: Answer) -> None:
            self._record(question, len(retrieved), answer, started, retrieved_at)
            memory.add(question, answer.text)

        return ChatStream(stream, on_complete)

    def _answer_question(self, question: str, history: Sequence[Turn]) -> Answer:
        """Retrieve and answer; refuses when nothing relevant is found.

        With history, the follow-up is first rewritten as a standalone question for retrieval.
        When telemetry is configured, one record per answered question is appended.
        """
        started = self._clock()
        search_text, retrieved = self._search(question, history)
        retrieved_at = self._clock()
        answer = self._generator.answer(search_text, retrieved, history)
        self._record(question, len(retrieved), answer, started, retrieved_at)
        return answer

    def _search(
        self, question: str, history: Sequence[Turn]
    ) -> tuple[str, Sequence[RetrievedChunk]]:
        """The text searched for (a standalone rewrite when there is history) and its hits."""
        search_text = self._generator.condense(question, history) if history else question
        return search_text, self._retriever.retrieve(search_text)

    def _record(
        self, question: str, retrieved: int, answer: Answer, started: float, retrieved_at: float
    ) -> None:
        """Append one telemetry record, when telemetry is configured."""
        finished = self._clock()
        if self._telemetry is not None:
            self._telemetry.record(
                QueryRecord(
                    timestamp=utc_timestamp(),
                    question_chars=len(question),
                    retrieved=retrieved,
                    refused=answer.refused,
                    retrieve_seconds=round(retrieved_at - started, TIMING_DECIMALS),
                    generate_seconds=round(finished - retrieved_at, TIMING_DECIMALS),
                    prompt_tokens=answer.prompt_tokens or 0,
                    completion_tokens=answer.completion_tokens or 0,
                )
            )


def build_pipeline(
    settings: Settings, client: OpenAI | None = None, *, record_telemetry: bool = True
) -> DocQAPipeline:
    """Create the real pipeline from settings. Pass `client` to avoid needing an API key.

    Evaluation runs pass `record_telemetry=False` so they do not fill the usage log.
    """
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
        telemetry=TelemetryLog(telemetry_path(settings)) if record_telemetry else None,
    )
