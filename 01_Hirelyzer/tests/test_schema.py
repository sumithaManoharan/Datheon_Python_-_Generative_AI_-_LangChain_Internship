"""Tests for the Pydantic resume schema: patterns, cross-field rules and cleaning."""

from typing import Any

import pytest
from hirelyzer.schema import Resume
from pydantic import ValidationError


def resume_with(**fields: Any) -> dict[str, Any]:
    """A minimal resume payload; keyword arguments add or override fields."""
    return {"is_resume": True, "name": "Jordan Avery", **fields}


def job(**fields: Any) -> dict[str, Any]:
    """A minimal experience entry; keyword arguments add or override fields."""
    return {
        "company": "Northwind Labs",
        "job_title": "Developer",
        "start_date": "2021-01",
        **fields,
    }


def school(**fields: Any) -> dict[str, Any]:
    """A minimal education entry; keyword arguments add or override fields."""
    return {"institution": "Springfield University", "degree": "B.Sc.", **fields}


def test_minimal_resume_validates_with_empty_defaults() -> None:
    resume = Resume.model_validate(resume_with())

    assert resume.is_resume is True
    assert resume.skills == []
    assert resume.experience == []


def test_valid_email_is_accepted() -> None:
    resume = Resume.model_validate(resume_with(email="jordan.avery@example.com"))

    assert resume.email == "jordan.avery@example.com"


@pytest.mark.parametrize("phone", ["+1 555 010 0199", "(555) 010-0199", "+91 90000 31245"])
def test_valid_phone_numbers_are_accepted(phone: str) -> None:
    assert Resume.model_validate(resume_with(phone=phone)).phone == phone


@pytest.mark.parametrize("email", ["not-an-email", "name@", "@example.com"])
def test_invalid_email_is_rejected(email: str) -> None:
    with pytest.raises(ValidationError):
        Resume.model_validate(resume_with(email=email))


@pytest.mark.parametrize("phone", ["abc", "12345", "555-CALL-NOW"])
def test_invalid_phone_is_rejected(phone: str) -> None:
    with pytest.raises(ValidationError):
        Resume.model_validate(resume_with(phone=phone))


@pytest.mark.parametrize("start_date", ["March 2022", "2022/03", "22-03"])
def test_invalid_date_format_is_rejected(start_date: str) -> None:
    with pytest.raises(ValidationError):
        Resume.model_validate(resume_with(experience=[job(start_date=start_date)]))


@pytest.mark.parametrize("year", ["99", "twenty"])
def test_invalid_education_year_is_rejected(year: str) -> None:
    with pytest.raises(ValidationError):
        Resume.model_validate(resume_with(education=[school(start_year=year)]))


def test_end_date_before_start_date_is_rejected() -> None:
    entry = job(start_date="2022-05", end_date="2021-01")

    with pytest.raises(ValidationError):
        Resume.model_validate(resume_with(experience=[entry]))


@pytest.mark.parametrize("end_date", ["2021-01", "2021-06"], ids=["same_month", "later_month"])
def test_end_date_not_before_start_date_is_accepted(end_date: str) -> None:
    entry = job(start_date="2021-01", end_date=end_date)

    assert Resume.model_validate(resume_with(experience=[entry])).experience


def test_current_job_with_end_date_is_rejected() -> None:
    entry = job(end_date="2022-06", is_current=True)

    with pytest.raises(ValidationError):
        Resume.model_validate(resume_with(experience=[entry]))


def test_current_job_without_end_date_is_accepted() -> None:
    resume = Resume.model_validate(resume_with(experience=[job(is_current=True)]))

    assert resume.experience[0].is_current is True
    assert resume.experience[0].end_date is None


def test_education_end_year_before_start_year_is_rejected() -> None:
    entry = school(start_year="2019", end_year="2015")

    with pytest.raises(ValidationError):
        Resume.model_validate(resume_with(education=[entry]))


def test_education_end_year_equal_to_start_year_is_accepted() -> None:
    entry = school(start_year="2019", end_year="2019")

    assert Resume.model_validate(resume_with(education=[entry])).education


def test_skills_are_stripped_and_deduplicated() -> None:
    resume = Resume.model_validate(resume_with(skills=[" Python ", "Python", "SQL"]))

    assert resume.skills == ["Python", "SQL"]


def test_null_list_becomes_empty_list() -> None:
    assert Resume.model_validate(resume_with(skills=None)).skills == []


def test_blank_text_becomes_none() -> None:
    assert Resume.model_validate(resume_with(name="   ")).name is None


def test_extra_top_level_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Resume.model_validate(resume_with(nickname="JA"))


def test_extra_nested_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Resume.model_validate(resume_with(experience=[job(bogus=1)]))


def test_non_resume_is_cleared_instead_of_rejected() -> None:
    resume = Resume.model_validate(
        {"is_resume": False, "name": "Invoice Corp", "skills": ["Python"], "email": "a@b.com"}
    )

    assert resume.is_resume is False
    assert resume.name is None
    assert resume.email is None
    assert resume.skills == []
    assert resume.experience == []
