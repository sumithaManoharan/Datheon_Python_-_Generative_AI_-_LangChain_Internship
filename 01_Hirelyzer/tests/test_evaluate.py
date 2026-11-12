"""Tests for the evaluation metrics, matching rules and run history."""

import copy
from pathlib import Path
from typing import Any

import pytest
from hirelyzer.evaluate import (
    METRICS,
    RunRecord,
    append_history,
    completeness_report,
    document_key,
    evaluate,
    normalize_field,
    precision_recall_f1,
    score_document,
    unrecognized_keys,
)
from hirelyzer.extractor import ExtractionResult
from hirelyzer.schema import Resume

TRUTH: dict[str, Any] = {
    "is_resume": True,
    "language": "English",
    "name": "Jordan Avery",
    "headline": "Backend Developer",
    "email": "jordan.avery@example.com",
    "phone": "+1 555 010 0199",
    "location": "Springfield",
    "skills": ["Python", "SQL"],
    "certifications": ["Cloud Practitioner"],
    "languages": ["English"],
    "experience": [
        {
            "company": "Northwind Labs",
            "job_title": "Developer",
            "job_location": "Springfield",
            "start_date": "2022-03",
            "end_date": None,
            "is_current": True,
            "highlights": ["Built APIs"],
        }
    ],
    "education": [
        {
            "institution": "Springfield University",
            "degree": "B.Sc.",
            "field_of_study": "Computer Science",
            "start_year": "2015",
            "end_year": "2019",
            "grade": "3.6",
        }
    ],
    "projects": [
        {"project_title": "Task Tracker", "project_description": "CLI", "skills_used": ["Python"]}
    ],
}


def predict(**changes: Any) -> dict[str, Any]:
    """A copy of the ground truth with some fields changed."""
    return {**copy.deepcopy(TRUTH), **changes}


@pytest.mark.parametrize(
    ("predicted", "expected", "result"),
    [
        ({"a", "b"}, {"a", "b"}, (1.0, 1.0, 1.0)),
        (set(), set(), (1.0, 1.0, 1.0)),
        ({"a"}, {"a", "b"}, (1.0, 0.5, 2 / 3)),
        ({"x"}, {"a"}, (0.0, 0.0, 0.0)),
        (set(), {"a"}, (0.0, 0.0, 0.0)),
    ],
)
def test_precision_recall_f1(
    predicted: set[str], expected: set[str], result: tuple[float, float, float]
) -> None:
    assert precision_recall_f1(predicted, expected) == pytest.approx(result)


@pytest.mark.parametrize("name", ["01_x.pdf.json", "01_x.json", "01_x.pdf", "01_x.JPG", "01_x"])
def test_document_key_matches_outputs_inputs_and_truth(name: str) -> None:
    assert document_key(name) == "01_x"


def test_phone_numbers_compare_by_digits() -> None:
    assert normalize_field("phone", "+91 90000-31245") == normalize_field("phone", "919000031245")


def test_perfect_prediction_scores_one_on_every_metric() -> None:
    scores = score_document(predict(), TRUTH)

    assert set(scores) == set(METRICS)
    assert all(value == pytest.approx(1.0) for value in scores.values())


def test_case_and_spacing_do_not_matter() -> None:
    assert score_document(predict(name="  jordan   AVERY "), TRUTH)["name"] == 1.0


def test_wrong_email_scores_zero_for_that_field_only() -> None:
    scores = score_document(predict(email="other@example.com"), TRUTH)

    assert scores["email"] == 0.0
    assert scores["name"] == 1.0


def test_extra_skill_lowers_precision_but_not_recall() -> None:
    scores = score_document(predict(skills=["Python", "SQL", "Go"]), TRUTH)

    assert scores["skills_precision"] == pytest.approx(2 / 3)
    assert scores["skills_recall"] == 1.0
    assert scores["skills_f1"] == pytest.approx(0.8)


def test_wrong_detail_inside_a_paired_job_lowers_the_field_score() -> None:
    wrong_job = {**TRUTH["experience"][0], "is_current": False}

    scores = score_document(predict(experience=[wrong_job]), TRUTH)

    assert scores["experience_f1"] == 1.0  # the job was found
    assert scores["experience_fields"] == pytest.approx(0.75)  # one of four fields is wrong


def test_clipped_highlight_counts_as_a_miss() -> None:
    clipped = {**TRUTH["experience"][0], "highlights": ["Built AP"]}

    assert score_document(predict(experience=[clipped]), TRUTH)["experience_highlights_f1"] == 0.0


def test_missing_job_scores_zero_for_experience() -> None:
    scores = score_document(predict(experience=[]), TRUTH)

    assert scores["experience_f1"] == 0.0
    assert scores["experience_fields"] == 0.0


def test_failed_pipeline_scores_zero_everywhere() -> None:
    assert score_document(None, TRUTH) == dict.fromkeys(METRICS, 0.0)


def test_non_resume_only_scores_is_resume() -> None:
    truth = {"is_resume": False}

    assert score_document({"is_resume": False, "name": None}, truth) == {"is_resume": 1.0}
    assert score_document({"is_resume": True, "name": "X"}, truth) == {"is_resume": 0.0}


def test_evaluate_averages_over_documents() -> None:
    truth = {"a": TRUTH, "b": {"is_resume": False}}

    _, perfect = evaluate({"a": predict(), "b": {"is_resume": False}}, truth)
    documents, partial = evaluate({"b": {"is_resume": False}}, truth)  # "a" has no output

    assert all(value == pytest.approx(1.0) for value in perfect.values())
    assert documents["a"] == dict.fromkeys(METRICS, 0.0)
    assert partial["is_resume"] == 0.5  # one of two documents classified correctly
    assert partial["name"] == 0.0  # averaged over true resumes only


def make_record(run_id: str, overall: float) -> RunRecord:
    """A run record with every metric set to the same value."""
    return RunRecord(
        run_id=run_id,
        mode="live",
        model="m",
        temperature=0.0,
        prompt_hash="abc12345",
        documents=1,
        failed=0,
        overall=overall,
        metrics=dict.fromkeys(METRICS, overall),
    )


def test_history_records_runs_and_returns_the_previous_one(tmp_path: Path) -> None:
    assert append_history(tmp_path, make_record("run1", 0.5)) is None

    previous = append_history(tmp_path, make_record("run2", 0.7))

    assert previous is not None
    assert (previous["run_id"], previous["overall"]) == ("run1", "0.5")
    assert len((tmp_path / "history.csv").read_text(encoding="utf-8").splitlines()) == 3  # header


def test_completeness_report_counts_without_showing_content() -> None:
    results = {
        "a": ExtractionResult(
            source="a",
            status="success",
            data=Resume.model_validate({"is_resume": True, "name": "Secret Name"}),
            attempts=2,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.0,
        ),
        "b": ExtractionResult(source="b"),
    }

    report = completeness_report(results)

    assert "Documents: 2 | valid JSON: 1 | failed: 1" in report
    assert "needed a repair round: 1" in report
    assert "name: 0/1 empty" in report
    assert "email: 1/1 empty" in report
    assert "Failed documents: b" in report
    assert "Secret Name" not in report


def to_ground_truth_style(doc: dict[str, Any]) -> dict[str, Any]:
    """Rename keys the way the real ground-truth files do."""
    renamed = {("full_name" if key == "name" else key): value for key, value in doc.items()}
    renamed["language"] = "en"
    job_names = {"job_title": "title", "job_location": "location"}
    renamed["experience"] = [
        {job_names.get(key, key): value for key, value in job.items()} for job in doc["experience"]
    ]
    renamed["education"] = [
        {("field" if key == "field_of_study" else key): value for key, value in school.items()}
        for school in doc["education"]
    ]
    return renamed


def test_ground_truth_key_names_and_language_codes_are_understood() -> None:
    scores = score_document(predict(), to_ground_truth_style(TRUTH))

    assert all(value == pytest.approx(1.0) for value in scores.values())


def test_language_names_match_language_codes() -> None:
    assert normalize_field("language", "English") == normalize_field("language", "en")


def test_unrecognized_keys_reports_extras_but_not_aliases_or_invoice_fields() -> None:
    resume = {**to_ground_truth_style(TRUTH), "links": ["x"]}
    resume["experience"] = [{**resume["experience"][0], "bonus": 1}]
    truth = {"a": resume, "b": {"is_resume": False, "total": 5}}

    assert unrecognized_keys(truth) == ["experience.bonus", "links"]
