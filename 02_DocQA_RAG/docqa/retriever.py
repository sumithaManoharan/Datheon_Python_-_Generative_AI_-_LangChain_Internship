"""Retrieve the chunks most relevant to a question, with a relevance guard."""

import logging
from dataclasses import dataclass

from docqa.errors import ConfigError
from docqa.store import SearchHit, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A search hit plus its relevance score (1 - cosine distance; higher is better)."""

    hit: SearchHit
    relevance: float


class Retriever:
    """Top-k search over a vector store that drops weakly related chunks.

    A `relevance_threshold` of 0.0 keeps every hit (guard off). An empty result is
    not an error: it means nothing in the documents answers the question, and the
    answer step turns it into a fixed refusal message.
    """

    def __init__(self, store: VectorStore, top_k: int, relevance_threshold: float) -> None:
        if top_k < 1:
            raise ConfigError(f"top_k must be at least 1, got {top_k}")
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ConfigError(f"relevance_threshold must be from 0 to 1, got {relevance_threshold}")
        self._store = store
        self._top_k = top_k
        self._threshold = relevance_threshold

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return up to top_k chunks at or above the threshold, most relevant first."""
        hits = self._store.search(question, self._top_k)
        kept = [
            RetrievedChunk(hit=hit, relevance=1.0 - hit.distance)
            for hit in hits
            if 1.0 - hit.distance >= self._threshold
        ]
        logger.info(
            "retrieved %d of %d chunks (threshold %.2f)", len(kept), len(hits), self._threshold
        )
        return kept
