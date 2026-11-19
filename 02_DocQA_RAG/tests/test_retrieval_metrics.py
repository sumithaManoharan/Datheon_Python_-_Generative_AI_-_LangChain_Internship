"""Tests for docqa.retrieval_metrics."""

import math

import pytest
from docqa.retrieval_metrics import (
    hit_text,
    is_relevant,
    keyword_recall,
    keyword_share,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)
from docqa.store import SearchHit


def test_keyword_share_ignores_case_and_ligatures():
    assert keyword_share("The E\ufb00ective date", ["effective", "DATE"]) == 1.0


def test_keyword_share_counts_the_found_fraction():
    assert keyword_share("alpha beta", ["alpha", "gamma", "delta", "beta"]) == 0.5


def test_no_keywords_gives_zero():
    assert keyword_share("anything", []) == 0.0


@pytest.mark.parametrize(
    ("share", "expected"), [(0.0, False), (0.4, False), (0.5, True), (1.0, True)]
)
def test_is_relevant_boundary(share: float, expected: bool):
    assert is_relevant(share) is expected


def test_keyword_recall_pools_all_retrieved_texts():
    assert keyword_recall(["alpha beta", "gamma"], ["alpha", "gamma", "delta"]) == pytest.approx(
        2 / 3
    )


def test_precision_counts_relevant_over_k():
    assert precision_at_k([1.0, 0.0, 0.5, 0.2], 4) == 0.5
    assert precision_at_k([1.0, 0.0, 0.5, 0.2], 2) == 0.5


def test_precision_treats_missing_results_as_misses():
    assert precision_at_k([1.0], 4) == 0.25


def test_precision_rejects_k_below_one():
    with pytest.raises(ValueError):
        precision_at_k([1.0], 0)


def test_reciprocal_rank_uses_the_first_relevant_rank():
    assert reciprocal_rank([0.0, 0.0, 1.0], 3) == pytest.approx(1 / 3)
    assert reciprocal_rank([1.0, 1.0], 2) == 1.0


def test_reciprocal_rank_is_zero_without_a_relevant_chunk_or_beyond_k():
    assert reciprocal_rank([0.0, 0.0], 2) == 0.0
    assert reciprocal_rank([0.0, 0.0, 1.0], 2) == 0.0


def test_ndcg_is_one_for_the_ideal_order():
    assert ndcg_at_k([1.0, 0.5, 0.0], 3) == pytest.approx(1.0)


def test_ndcg_penalises_a_relevant_chunk_ranked_low():
    assert ndcg_at_k([0.0, 1.0], 2) == pytest.approx(1 / math.log2(3))


def test_ndcg_is_zero_when_nothing_is_relevant():
    assert ndcg_at_k([0.0, 0.0], 2) == 0.0
    assert ndcg_at_k([0.0, 0.0, 1.0], 2) == 0.0


def test_hit_text_includes_file_name_section_and_chunk():
    hit = SearchHit(text="body", source="docs/Maxine Thompson.md", distance=0.1, section="Career")
    assert hit_text(hit) == "Maxine Thompson Career body"
