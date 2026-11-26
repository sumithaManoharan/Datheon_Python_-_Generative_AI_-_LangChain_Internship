"""Tests for DocQAPipeline.chat_stream: what is recorded, and when."""

from types import SimpleNamespace

import pytest
from docqa.answer import REFUSAL_MESSAGE, Answer, AnswerStream, Turn
from docqa.conversation import ConversationMemory
from docqa.errors import InvalidQuestionError, LLMCallError
from docqa.pipeline import DocQAPipeline
from docqa.retriever import RetrievedChunk
from docqa.store import SearchHit


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(hit=SearchHit(text=text, source="docs/a.md", distance=0.2), relevance=0.8)


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, text: str) -> list[RetrievedChunk]:
        self.queries.append(text)
        return self._chunks


class FakeGenerator:
    """Streams scripted pieces; records what the pipeline asked it to do."""

    def __init__(self, pieces=("Hello ", "world [1]."), *, refused=False, fail_after=None):
        self._pieces = list(pieces)
        self._refused = refused
        self._fail_after = fail_after
        self.condensed: list[str] = []
        self.stream_calls: list[tuple] = []

    def condense(self, question: str, history) -> str:
        self.condensed.append(question)
        return f"standalone: {question}"

    def answer_stream(self, search_text, retrieved, history) -> AnswerStream:
        self.stream_calls.append((search_text, tuple(retrieved), tuple(history)))
        if self._refused:
            refusal = Answer(text=REFUSAL_MESSAGE, citations=(), refused=True)
            return AnswerStream([REFUSAL_MESSAGE], lambda _text: refusal)

        def pieces():
            for number, piece in enumerate(self._pieces):
                if self._fail_after is not None and number == self._fail_after:
                    raise LLMCallError("connection lost")
                yield piece

        def finish(text: str) -> Answer:
            return Answer(
                text=text, citations=(), refused=False, prompt_tokens=11, completion_tokens=4
            )

        return AnswerStream(pieces(), finish)


class FakeTelemetry:
    def __init__(self) -> None:
        self.records: list = []

    def record(self, record) -> None:
        self.records.append(record)


def _pipeline(generator, retriever=None, telemetry=None, clock=None) -> DocQAPipeline:
    extra = {} if clock is None else {"clock": clock}
    return DocQAPipeline(
        SimpleNamespace(),
        retriever or FakeRetriever([_chunk("alpha")]),
        generator,
        chunk_size=800,
        chunk_overlap=100,
        max_file_bytes=1_000_000,
        max_question_chars=100,
        telemetry=telemetry,
        **extra,
    )


def test_pieces_stream_and_memory_is_updated_only_when_the_stream_ends():
    memory = ConversationMemory()
    stream = _pipeline(FakeGenerator()).chat_stream("What is alpha?", memory)

    seen = []
    for piece in stream:
        seen.append(piece)
        if len(seen) == 1:
            assert len(memory) == 0  # half an answer is never remembered

    assert seen == ["Hello ", "world [1]."]
    assert memory.turns == (Turn(question="What is alpha?", answer="Hello world [1]."),)
    assert stream.answer.text == "Hello world [1]."


def test_the_first_question_is_searched_as_asked_without_a_rewrite():
    generator, retriever = FakeGenerator(), FakeRetriever([_chunk("alpha")])
    stream = _pipeline(generator, retriever).chat_stream("What is alpha?", ConversationMemory())
    list(stream)

    assert generator.condensed == []
    assert retriever.queries == ["What is alpha?"]
    assert generator.stream_calls[0][0] == "What is alpha?"
    assert generator.stream_calls[0][2] == ()


def test_a_follow_up_is_rewritten_and_the_history_is_passed_on():
    memory = ConversationMemory()
    memory.add("Who founded it?", "Ann.")
    generator, retriever = FakeGenerator(), FakeRetriever([_chunk("alpha")])
    list(_pipeline(generator, retriever).chat_stream("When?", memory))

    assert generator.condensed == ["When?"]
    assert retriever.queries == ["standalone: When?"]
    search_text, _retrieved, history = generator.stream_calls[0]
    assert search_text == "standalone: When?"
    assert history == (Turn(question="Who founded it?", answer="Ann."),)
    # memory keeps the question as the user typed it, not the rewrite
    assert memory.turns[-1].question == "When?"
    assert len(memory) == 2


def test_stopping_the_stream_early_records_nothing():
    memory, telemetry = ConversationMemory(), FakeTelemetry()
    stream = _pipeline(FakeGenerator(), telemetry=telemetry).chat_stream("Q?", memory)

    iterator = iter(stream)
    next(iterator)
    iterator.close()

    assert len(memory) == 0
    assert telemetry.records == []
    with pytest.raises(RuntimeError):
        _ = stream.answer


def test_an_error_mid_stream_records_nothing_and_propagates():
    memory, telemetry = ConversationMemory(), FakeTelemetry()
    generator = FakeGenerator(fail_after=1)
    stream = _pipeline(generator, telemetry=telemetry).chat_stream("Q?", memory)

    with pytest.raises(LLMCallError):
        list(stream)

    assert len(memory) == 0
    assert telemetry.records == []


def test_one_telemetry_record_is_written_when_the_stream_ends():
    ticks = iter([10.0, 11.0, 13.5])
    telemetry = FakeTelemetry()
    stream = _pipeline(FakeGenerator(), telemetry=telemetry, clock=lambda: next(ticks)).chat_stream(
        "What is alpha?", ConversationMemory()
    )

    assert telemetry.records == []  # nothing yet: retrieval is done, the answer is not
    list(stream)

    (record,) = telemetry.records
    assert record.question_chars == len("What is alpha?")
    assert record.retrieved == 1
    assert record.refused is False
    assert record.retrieve_seconds == 1.0
    assert record.generate_seconds == 2.5
    assert (record.prompt_tokens, record.completion_tokens) == (11, 4)


def test_a_refusal_is_streamed_remembered_and_logged_as_refused():
    memory, telemetry = ConversationMemory(), FakeTelemetry()
    stream = _pipeline(FakeGenerator(refused=True), telemetry=telemetry).chat_stream("Q?", memory)

    assert list(stream) == [REFUSAL_MESSAGE]
    assert stream.answer.refused is True
    assert memory.turns[-1].answer == REFUSAL_MESSAGE
    assert telemetry.records[0].refused is True


def test_an_invalid_question_touches_nothing():
    memory, generator = ConversationMemory(), FakeGenerator()
    retriever = FakeRetriever([_chunk("alpha")])

    with pytest.raises(InvalidQuestionError):
        _pipeline(generator, retriever).chat_stream("   ", memory)

    assert (len(memory), retriever.queries, generator.stream_calls) == (0, [], [])


def test_works_without_telemetry():
    memory = ConversationMemory()
    list(_pipeline(FakeGenerator(), telemetry=None).chat_stream("Q?", memory))

    assert len(memory) == 1
