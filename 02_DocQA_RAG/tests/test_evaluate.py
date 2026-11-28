"""Tests for docqa.evaluate."""

import json
import math
from pathlib import Path

import pytest
from docqa.errors import EvaluationError
from docqa.evaluate import (
    OVERALL,
    Question,
    evaluate_retrieval,
    format_report,
    load_questions,
    parse_args,
    summarize,
)
from docqa.store import SearchHit


def _hit(text: str, name: str = "a.md", section: str = "Doc") -> SearchHit:
    return SearchHit(text=text, source=f"docs/{name}", distance=0.2, section=section)


class FakeSearch:
    """Returns canned hits per question and records every call."""

    def __init__(self, hits_by_question):
        self.hits_by_question = hits_by_question
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, k: int):
        self.calls.append((query, k))
        return self.hits_by_question[query][:k]


QUESTION = Question("Who founded it?", ("avery", "lancaster"), "direct_fact")


# ---------- loading ----------


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "q.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_questions_parses_rows_and_skips_blank_lines(tmp_path: Path):
    row = {"question": "Q?", "keywords": ["a", "b"], "category": "temporal"}
    questions = load_questions(_write(tmp_path, json.dumps(row) + "\n\n"))
    assert questions == [Question("Q?", ("a", "b"), "temporal")]


def test_missing_category_defaults(tmp_path: Path):
    path = _write(tmp_path, json.dumps({"question": "Q?", "keywords": ["a"]}))
    assert load_questions(path)[0].category == "uncategorized"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps({"keywords": ["a"]}),
        json.dumps({"question": "Q?"}),
        json.dumps({"question": "Q?", "keywords": []}),
        json.dumps({"question": "Q?", "keywords": [1]}),
        json.dumps(["not", "an", "object"]),
        "",
    ],
)
def test_bad_question_files_raise(tmp_path: Path, text: str):
    with pytest.raises(EvaluationError):
        load_questions(_write(tmp_path, text))


def test_missing_question_file_raises(tmp_path: Path):
    with pytest.raises(EvaluationError):
        load_questions(tmp_path / "nope.jsonl")


# ---------- scoring ----------


def test_scores_for_a_relevant_chunk_ranked_second():
    search = FakeSearch(
        {QUESTION.question: [_hit("unrelated text"), _hit("Avery Lancaster founded it")]}
    )
    score = evaluate_retrieval(search, [QUESTION], [2])[0]
    assert score.k == 2
    assert score.recall == 1.0
    assert score.precision == 0.5
    assert score.mrr == 0.5
    assert score.ndcg == pytest.approx(1 / math.log2(3))
    assert score.top_hits[0] == "a.md | Doc"


def test_search_runs_once_per_question_at_the_deepest_k():
    search = FakeSearch({QUESTION.question: [_hit("x")] * 10})
    evaluate_retrieval(search, [QUESTION], [4, 10])
    assert search.calls == [(QUESTION.question, 10)]


def test_smaller_k_ignores_deeper_hits():
    search = FakeSearch({QUESTION.question: [_hit("nothing"), _hit("Avery Lancaster")]})
    shallow, deep = evaluate_retrieval(search, [QUESTION], [1, 2])
    assert (shallow.mrr, shallow.recall) == (0.0, 0.0)
    assert (deep.mrr, deep.recall) == (0.5, 1.0)


@pytest.mark.parametrize("ks", [[], [0], [4, -1]])
def test_invalid_ks_raise(ks):
    with pytest.raises(EvaluationError):
        evaluate_retrieval(FakeSearch({}), [QUESTION], ks)


def test_file_name_counts_toward_relevance():
    question = Question("Who won?", ("maxine", "thompson"), "direct_fact")
    search = FakeSearch({question.question: [_hit("award text", name="Maxine Thompson.md")]})
    assert evaluate_retrieval(search, [question], [1])[0].mrr == 1.0


# ---------- summaries and report ----------


def _scores():
    hit = _hit("Avery Lancaster founded it")
    miss = _hit("unrelated")
    search = FakeSearch({"good?": [hit], "bad?": [miss], "other?": [hit]})
    questions = [
        Question("good?", ("avery", "lancaster"), "direct_fact"),
        Question("bad?", ("avery", "lancaster"), "direct_fact"),
        Question("other?", ("avery", "lancaster"), "temporal"),
    ]
    return evaluate_retrieval(search, questions, [1])


def test_summarize_groups_by_category_with_an_overall_row():
    rows = {row.category: row for row in summarize(_scores(), 1)}
    assert set(rows) == {"direct_fact", "temporal", OVERALL}
    assert rows["direct_fact"].n == 2
    assert rows["direct_fact"].mrr == 0.5
    assert rows["temporal"].mrr == 1.0
    assert rows[OVERALL].n == 3
    assert rows[OVERALL].mrr == pytest.approx(2 / 3)


def test_report_has_a_table_and_lists_the_failing_question():
    report = format_report(_scores(), [1], worst=5)
    assert "### Retrieval at k=1" in report
    assert "| direct_fact | 2 |" in report
    assert "| ALL | 3 |" in report
    assert "No relevant chunk in the top 1: 1 of 3 questions" in report
    assert "bad?" in report
    assert "a.md | Doc" in report


def test_report_says_none_when_every_question_has_a_relevant_chunk():
    search = FakeSearch({QUESTION.question: [_hit("Avery Lancaster")]})
    report = format_report(evaluate_retrieval(search, [QUESTION], [1]), [1])
    assert "0 of 1 questions" in report
    assert "none" in report


# ---------- command line ----------


def test_parse_args_defaults_and_ks():
    args = parse_args([])
    assert args.ks == (4, 10)
    assert parse_args(["--ks", "3,5"]).ks == (3, 5)


@pytest.mark.parametrize("value", ["4,x", "0", "-1"])
def test_parse_args_rejects_bad_ks(value: str):
    with pytest.raises(SystemExit):
        parse_args(["--ks", value])
