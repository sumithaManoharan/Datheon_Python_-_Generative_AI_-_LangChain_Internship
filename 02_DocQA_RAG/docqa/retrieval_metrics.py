"""Retrieval metrics with keyword-based relevance.

The question file has no relevance labels, so a chunk's relevance is the share of the
question's keywords found in what a reader sees in a citation: file name, section path
and chunk text. These are indicators for comparing changes, not ground truth: a chunk
can contain a keyword without answering the question.
"""

import math
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from docqa.store import SearchHit

# A chunk counts as relevant when it holds at least this share of the keywords.
RELEVANT_SHARE = 0.5


def _normalize(text: str) -> str:
    """NFKC plus case folding, so ligatures and capitals do not hide a match."""
    return unicodedata.normalize("NFKC", text).casefold()


def hit_text(hit: SearchHit) -> str:
    """The text a reader sees in a citation: file name, section path and the chunk."""
    return " ".join(part for part in (Path(hit.source).stem, hit.section, hit.text) if part)


def keyword_share(text: str, keywords: Sequence[str]) -> float:
    """Share of keywords found in the text (0.0 when there are no keywords)."""
    if not keywords:
        return 0.0
    haystack = _normalize(text)
    found = sum(1 for keyword in keywords if _normalize(keyword) in haystack)
    return found / len(keywords)


def is_relevant(share: float) -> bool:
    """Binary relevance from a keyword share."""
    return share >= RELEVANT_SHARE


def keyword_recall(texts: Sequence[str], keywords: Sequence[str]) -> float:
    """Share of keywords found anywhere in the retrieved texts."""
    return keyword_share("\n".join(texts), keywords)


def precision_at_k(shares: Sequence[float], k: int) -> float:
    """Relevant chunks among the top k, divided by k (missing results count as misses)."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return sum(is_relevant(share) for share in shares[:k]) / k


def reciprocal_rank(shares: Sequence[float], k: int) -> float:
    """1 / rank of the first relevant chunk within the top k, else 0."""
    for rank, share in enumerate(shares[:k], start=1):
        if is_relevant(share):
            return 1.0 / rank
    return 0.0


def _dcg(grades: Sequence[float]) -> float:
    """Discounted cumulative gain of graded relevance in ranked order."""
    return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def ndcg_at_k(shares: Sequence[float], k: int) -> float:
    """nDCG of the top k, with the ideal order taken from the same retrieved chunks.

    It scores the ordering of what was retrieved; recall covers whether the facts
    were retrieved at all.
    """
    top = list(shares[:k])
    ideal = _dcg(sorted(top, reverse=True))
    return _dcg(top) / ideal if ideal > 0 else 0.0
