"""Tests for docqa.evaluate_answers."""

import json
from pathlib import Path

import pytest
from docqa.answer import Answer, Citation
from docqa.config import load_settings
from docqa.errors import EvaluationError, LLMCallError
from docqa.evaluate import OVERALL, Question
from docqa.evaluate_answers import (
    RefusalRecord,
    biggest_differences,
    check_refusals,
    coverage_by_category,
    format_report,
    load_refusal_questions,
    main,
    parse_args,
    run_questions,
    sample,
    summarize_answers,
)

QUESTIONS = [
    Question("Who founded it?", ("avery", "lancaster"), "easy"),
    Question("When?", ("2015",), "easy"),
    Question("Spans?", ("alpha", "beta"), "hard"),
]
FULL = {
    "Who founded it?": "Avery Lancaster [1].",
    "When?": "Founded in 2015 [1].",
    "Spans?": "alpha and beta [1].",
}


def _answer(text, *, refused=False, cited=True, prompt=100, completion=10) -> Answer:
    citations = ()
    if cited:
        citations = (
            Citation(number=1, source="a.md", snippet="s", section="S", start_line=1, end_line=2),
        )
    return Answer(
        text=text,
        citations=citations,
        refused=refused,
        prompt_tokens=prompt,
        completion_tokens=completion,
    )


def _clock():
    ticks = iter(range(10_000))
    return lambda: float(next(ticks))


def _run(replies, k=6, questions=QUESTIONS):
    return run_questions(lambda q: replies[q], questions, k, clock=_clock())


# ---------- run_questions ----------


def test_run_questions_records_one_scored_row_per_question():
    texts = {**FULL, "Spans?": "alpha only [1]."}
    partial = {question: _answer(text) for question, text in texts.items()}
    records = _run(partial)
    assert [r.question for r in records] == [q.question for q in QUESTIONS]
    assert all(r.k == 6 and r.seconds == 1.0 and r.cited for r in records)
    assert [r.coverage for r in records] == [1.0, 1.0, 0.5]


def test_a_failed_call_is_recorded_and_the_run_continues():
    def ask(question):
        if question == "When?":
            raise LLMCallError("boom")
        return _answer("Avery Lancaster [1].")

    records = run_questions(ask, QUESTIONS, 6, clock=_clock())
    assert [r.error for r in records] == [None, "boom", None]


def test_progress_is_reported_after_each_question():
    seen = []
    run_questions(
        lambda q: _answer(FULL[q]),
        QUESTIONS,
        6,
        clock=_clock(),
        progress=lambda d, t: seen.append((d, t)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_sample_spans_the_file_and_respects_the_limit():
    items = [Question(f"q{i}?", ("x",), "c") for i in range(10)]
    chosen = sample(items, 4)
    assert len(chosen) == 4
    assert chosen[0] == items[0]
    assert sample(items, None) == items
    assert sample(items, 50) == items
    with pytest.raises(EvaluationError):
        sample(items, 0)


# ---------- summaries ----------


def test_summary_averages_coverage_refusals_citations_and_tokens():
    questions = [*QUESTIONS, Question("Fails?", ("x",), "hard")]
    replies = {
        "Who founded it?": _answer("Avery Lancaster [1]."),
        "When?": _answer("I could not find this.", refused=True, cited=False, prompt=0),
        "Spans?": _answer("alpha only [1]."),
    }

    def ask(question):
        if question == "Fails?":
            raise LLMCallError("boom")
        return replies[question]

    summary = summarize_answers(run_questions(ask, questions, 6, clock=_clock()), 6)
    assert (summary.n, summary.errors) == (3, 1)
    assert summary.coverage == pytest.approx(0.5)
    assert summary.full_rate == pytest.approx(1 / 3)
    assert summary.hit_rate == pytest.approx(2 / 3)
    assert summary.refusal_rate == pytest.approx(1 / 3)
    assert summary.cited_rate == 1.0
    assert summary.mean_prompt_tokens == pytest.approx(200 / 3)


def test_summary_of_no_records_is_all_zero():
    summary = summarize_answers([], 6)
    assert (summary.n, summary.coverage, summary.cited_rate) == (0, 0.0, 0.0)


def test_coverage_by_category_has_one_column_per_top_k_and_an_overall_row_last():
    nothing = {q.question: _answer("nothing [1].") for q in QUESTIONS}
    full = {q.question: _answer(FULL[q.question]) for q in QUESTIONS}
    table = coverage_by_category({6: _run(nothing), 10: _run(full, 10)})
    assert table["easy"] == {6: 0.0, 10: 1.0}
    assert table["hard"] == {6: 0.0, 10: 1.0}
    assert list(table)[-1] == OVERALL


def test_biggest_differences_lists_only_changed_questions():
    nothing = {q.question: _answer("nothing [1].") for q in QUESTIONS}
    full = {q.question: _answer(FULL[q.question]) for q in QUESTIONS}
    changed = biggest_differences({6: _run(nothing), 10: _run(full, 10)}, 6, 10, 10)
    assert len(changed) == 3
    assert all((before, after) == (0.0, 1.0) for _, before, after in changed)
    same = biggest_differences({6: _run(full), 10: _run(full, 10)}, 6, 10, 10)
    assert same == []


# ---------- refusal questions ----------


def test_load_refusal_questions_accepts_objects_and_plain_strings(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"question": "A?"}\n\n"B?"\n', encoding="utf-8")
    assert load_refusal_questions(path) == ["A?", "B?"]


@pytest.mark.parametrize("content", ["not json\n", '{"other": 1}\n', "\n"])
def test_load_refusal_questions_rejects_bad_files(tmp_path: Path, content: str):
    path = tmp_path / "r.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(EvaluationError):
        load_refusal_questions(path)


def test_load_refusal_questions_rejects_a_missing_file(tmp_path: Path):
    with pytest.raises(EvaluationError):
        load_refusal_questions(tmp_path / "none.jsonl")


def test_check_refusals_tells_retrieval_refusals_from_model_refusals():
    replies = {
        "gone?": _answer("I could not find this.", refused=True, cited=False, prompt=0),
        "model?": _answer("I could not find this.", refused=True, cited=False, prompt=50),
        "leaked?": _answer("It is 42 [1].", prompt=50),
    }
    records = check_refusals(lambda q: replies[q], list(replies), "near_topic", 6)
    assert [(r.refused, r.llm_called) for r in records] == [
        (True, False),
        (True, True),
        (False, True),
    ]


def test_check_refusals_records_a_failed_call():
    def ask(_question):
        raise LLMCallError("boom")

    (record,) = check_refusals(ask, ["x?"], "near_topic", 6)
    assert record.error == "boom"
    assert record.refused is False


# ---------- report and CLI ----------


def test_report_has_every_section_and_shows_unrefused_answers():
    nothing = {q.question: _answer("nothing [1].") for q in QUESTIONS}
    full = {q.question: _answer(FULL[q.question]) for q in QUESTIONS}
    refusals = [
        RefusalRecord("near_topic", "Ticker?", 6, False, True, "It is ACME [1]."),
        RefusalRecord("near_topic", "Ticker?", 10, True, False, "I could not find this."),
    ]
    report = format_report({6: _run(nothing), 10: _run(full, 10)}, refusals)
    assert "### Answer quality by top_k" in report
    assert "### Keyword coverage by category" in report
    assert "### Answers that changed most between k=6 and k=10" in report
    assert "### Answerable questions refused at k=10: 0" in report
    assert "### Unanswerable questions" in report
    assert "Not refused: 1" in report
    assert "It is ACME [1]." in report


def test_parse_args_defaults():
    args = parse_args([])
    assert args.ks == (6, 10)
    assert args.limit is None


def _write_inputs(tmp_path: Path):
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        "\n".join(
            json.dumps(
                {"question": q.question, "keywords": list(q.keywords), "category": q.category}
            )
            for q in QUESTIONS[:2]
        ),
        encoding="utf-8",
    )
    refusal = tmp_path / "r.jsonl"
    refusal.write_text('{"question": "Ticker?"}\n', encoding="utf-8")
    return questions, refusal


def test_main_runs_every_top_k_and_writes_the_json_records(tmp_path: Path, capsys):
    questions, refusal = _write_inputs(tmp_path)
    out = tmp_path / "out" / "answers.json"

    def factory(_settings, _k):
        return lambda q: _answer(FULL.get(q, "I could not find this."), refused=q not in FULL)

    code = main(
        [
            "--questions",
            str(questions),
            "--refusal-files",
            str(refusal),
            "--ks",
            "4,6",
            "--out",
            str(out),
        ],
        settings=load_settings({}),
        ask_factory=factory,
    )
    assert code == 0
    assert "| 4 |" in capsys.readouterr().out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["answers"]) == 4
    assert len(data["refusals"]) == 2


def test_main_with_a_missing_questions_file_exits_one(tmp_path: Path, capsys):
    code = main(["--questions", str(tmp_path / "none.jsonl")], settings=load_settings({}))
    assert code == 1
    assert "error:" in capsys.readouterr().err
