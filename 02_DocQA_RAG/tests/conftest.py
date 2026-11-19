"""Shared pytest fixtures."""

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

PdfFactory = Callable[..., Path]


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors.

    Records every `embed_documents` call, and raises `fail_with` when it is set.
    """

    name = "fake-embedder"

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.document_calls: list[list[str]] = []
        self.fail_with: Exception | None = None

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        if not any(vec):
            vec[0] = 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail_with is not None:
            raise self.fail_with
        self.document_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.fail_with is not None:
            raise self.fail_with
        return self._vector(text)


@pytest.fixture
def embedder() -> FakeEmbedder:
    """A fresh deterministic fake embedder."""
    return FakeEmbedder()


@pytest.fixture
def make_pdf(tmp_path: Path) -> PdfFactory:
    """Return a function that writes a small PDF into tmp_path.

    Each string in `pages` becomes one page; an empty string gives a blank page.
    """

    def _make(
        pages: list[str],
        name: str = "sample.pdf",
        password: str | None = None,
    ) -> Path:
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        path = tmp_path / name
        if password:
            doc.save(
                path,
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                user_pw=password,
                owner_pw=password,
            )
        else:
            doc.save(path)
        doc.close()
        return path

    return _make


class FakeLLM:
    """Stands in for an OpenAI client: returns a canned reply or raises scripted errors."""

    def __init__(self, reply: str | None = "Answer [1].") -> None:
        self.reply = reply
        self.failures: list[Exception] = []
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, temperature):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        if self.failures:
            raise self.failures.pop(0)
        message = SimpleNamespace(content=self.reply)
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture
def llm() -> FakeLLM:
    """A fresh fake LLM client."""
    return FakeLLM()
