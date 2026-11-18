"""Tests for docqa.embeddings."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from docqa.embeddings import OpenAIEmbedder
from docqa.errors import ConfigError, EmbeddingError

REQUEST = httpx.Request("POST", "https://example.test/v1")


def _status_error(cls, status: int):
    return cls("boom", response=httpx.Response(status, request=REQUEST), body=None)


class FakeEmbeddingsAPI:
    """Stands in for `client.embeddings`; the vector's first value is the text length."""

    def __init__(self, failures=(), reverse=False, drop_last=False):
        self.failures = list(failures)
        self.reverse = reverse
        self.drop_last = drop_last
        self.calls: list[list[str]] = []

    def create(self, model, input):
        self.calls.append(list(input))
        if self.failures:
            raise self.failures.pop(0)
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text)), 1.0])
            for i, text in enumerate(input)
        ]
        if self.reverse:
            data.reverse()
        if self.drop_last:
            data = data[:-1]
        return SimpleNamespace(data=data)


def _embedder(api, sleeps=None, **kwargs):
    sleeps = [] if sleeps is None else sleeps
    client = SimpleNamespace(embeddings=api)
    return OpenAIEmbedder(client, "test-model", sleep=sleeps.append, **kwargs)


def test_texts_are_sent_in_batches_and_order_is_kept():
    api = FakeEmbeddingsAPI()
    vectors = _embedder(api, batch_size=2).embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [len(call) for call in api.calls] == [2, 2, 1]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_out_of_order_response_is_sorted_by_index():
    api = FakeEmbeddingsAPI(reverse=True)
    vectors = _embedder(api).embed_documents(["a", "bb", "ccc"])
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


def test_rate_limit_is_retried_then_succeeds():
    sleeps: list[float] = []
    api = FakeEmbeddingsAPI(failures=[_status_error(openai.RateLimitError, 429)])
    vectors = _embedder(api, sleeps).embed_documents(["a"])
    assert vectors == [[1.0, 1.0]]
    assert len(api.calls) == 2
    assert len(sleeps) == 1


def test_auth_error_is_not_retried():
    api = FakeEmbeddingsAPI(failures=[_status_error(openai.AuthenticationError, 401)])
    with pytest.raises(EmbeddingError):
        _embedder(api).embed_documents(["a"])
    assert len(api.calls) == 1


def test_attempts_exhausted_raises_embedding_error():
    errors = [_status_error(openai.RateLimitError, 429)] * 3
    api = FakeEmbeddingsAPI(failures=errors)
    with pytest.raises(EmbeddingError) as info:
        _embedder(api, max_attempts=3).embed_documents(["a"])
    assert len(api.calls) == 3
    assert isinstance(info.value.__cause__, openai.RateLimitError)


def test_wrong_number_of_vectors_raises():
    api = FakeEmbeddingsAPI(drop_last=True)
    with pytest.raises(EmbeddingError):
        _embedder(api).embed_documents(["a", "bb"])


def test_empty_input_makes_no_request():
    api = FakeEmbeddingsAPI()
    assert _embedder(api).embed_documents([]) == []
    assert api.calls == []


def test_embed_query_returns_one_vector():
    assert _embedder(FakeEmbeddingsAPI()).embed_query("abc") == [3.0, 1.0]


def test_name_is_the_model():
    assert _embedder(FakeEmbeddingsAPI()).name == "test-model"


def test_invalid_batch_size_raises():
    with pytest.raises(ConfigError):
        _embedder(FakeEmbeddingsAPI(), batch_size=0)
