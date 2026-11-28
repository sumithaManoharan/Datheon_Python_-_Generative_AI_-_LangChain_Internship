"""Pydantic models describing the JSON that Hirelyzer extracts from a resume.
The Resume schema is also sent to the LLM inside the system prompt, so the
Field descriptions below double as instructions for the model.
"""

from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# Patterns avoid lookaheads: pydantic's default regex engine does not support them.
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
PHONE_PATTERN = r"^[0-9+()\-\s]+$"
YEAR_MONTH_PATTERN = r"^[0-9]{4}-(0[1-9]|1[0-2])$"
YEAR_PATTERN = r"^[0-9]{4}$"
MIN_PHONE_DIGITS = 7
MAX_PHONE_DIGITS = 15

YearMonth = Annotated[str, StringConstraints(pattern=YEAR_MONTH_PATTERN)]
Year = Annotated[str, StringConstraints(pattern=YEAR_PATTERN)]


def _blank_to_none(value: Any) -> Any:
    """Treat an empty or whitespace-only string as a missing value."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _clean_list(value: Any) -> Any:
    """Turn null into [], strip string items, drop blanks and case-insensitive duplicates."""
    if value is None:
        return []
    if not isinstance(value, list):
        return value  # let pydantic report the wrong type
    seen: set[str] = set()
    cleaned: list[Any] = []
    for item in value:
        if isinstance(item, str):
            item = item.strip()
            if not item or item.casefold() in seen:
                continue
            seen.add(item.casefold())
        cleaned.append(item)
    return cleaned


class Strict(BaseModel):
    """Base model: unknown keys are rejected and strings are stripped."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Experience(Strict):
    """One job entry from the experience section."""

    company: str | None = None
    job_title: str | None = None
    job_location: str | None = None
    start_date: YearMonth | None = Field(default=None, description="YYYY-MM, or null if not stated")
    end_date: YearMonth | None = Field(
        default=None, description="YYYY-MM, or null if the job is current"
    )
    is_current: bool = False
    highlights: list[str] = Field(
        default_factory=list,
        description=(
            "Key responsibilities and achievements for this job, one bullet per entry. "
            "Stay close to the resume's wording and do not invent anything. "
            "Empty list if the resume gives no details."
        ),
    )

    _coerce_blank = field_validator(
        "company", "job_title", "job_location", "start_date", "end_date", mode="before"
    )(_blank_to_none)
    _coerce_list = field_validator("highlights", mode="before")(_clean_list)

    @model_validator(mode="after")
    def _check_dates(self) -> Self:
        """A current job has no end date; an end date cannot precede the start date."""
        if self.is_current and self.end_date is not None:
            raise ValueError("end_date must be null when is_current is true")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        return self


class Education(Strict):
    """One education entry."""

    institution: str | None = None
    degree: str | None = Field(default=None, description="For example: B.Sc, M.Sc, MS, BE")
    field_of_study: str | None = Field(
        default=None, description="The subject studied, for example maths, robotics, data science"
    )
    start_year: Year | None = Field(default=None, description="YYYY only. Null if not stated.")
    end_year: Year | None = Field(default=None, description="YYYY only. Null if not stated.")
    grade: str | None = Field(
        default=None, description="Grade or GPA exactly as written, null if not stated"
    )

    _coerce_blank = field_validator(
        "institution", "degree", "field_of_study", "start_year", "end_year", "grade", mode="before"
    )(_blank_to_none)

    @model_validator(mode="after")
    def _check_years(self) -> Self:
        """The end year cannot precede the start year."""
        if self.start_year and self.end_year and self.end_year < self.start_year:
            raise ValueError("end_year must not be earlier than start_year")
        return self


class Project(Strict):
    """One project entry."""

    project_title: str | None = None
    project_description: str | None = None
    skills_used: list[str] = Field(
        default_factory=list, description="Skills or technologies used in the project"
    )

    _coerce_blank = field_validator("project_title", "project_description", mode="before")(
        _blank_to_none
    )
    _coerce_list = field_validator("skills_used", mode="before")(_clean_list)


class Resume(Strict):
    """The full structured resume. is_resume comes first so the model decides it first."""

    is_resume: bool = Field(
        description="true if the document is a resume or CV, false for anything else"
    )
    language: str | None = Field(
        default=None, description="The language the resume itself is written in"
    )
    name: str | None = Field(default=None, min_length=2)
    headline: str | None = None
    email: str | None = Field(default=None, pattern=EMAIL_PATTERN)
    phone: str | None = Field(default=None, pattern=PHONE_PATTERN)
    location: str | None = None
    skills: list[str] = Field(default_factory=list, description="One skill per entry")
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(
        default_factory=list, description="Certifications of the candidate"
    )
    projects: list[Project] = Field(default_factory=list)
    languages: list[str] = Field(
        default_factory=list,
        description="Languages the candidate speaks (not the language of the resume)",
    )

    _coerce_blank = field_validator(
        "language", "name", "headline", "email", "phone", "location", mode="before"
    )(_blank_to_none)
    _coerce_list = field_validator(
        "skills",
        "experience",
        "education",
        "certifications",
        "projects",
        "languages",
        mode="before",
    )(_clean_list)

    @field_validator("phone")
    @classmethod
    def _check_phone_digits(cls, value: str | None) -> str | None:
        """Allow only 7 to 15 digits in total, ignoring separators."""
        if value is None:
            return value
        digits = sum(char.isdigit() for char in value)
        if not MIN_PHONE_DIGITS <= digits <= MAX_PHONE_DIGITS:
            raise ValueError(f"phone must contain {MIN_PHONE_DIGITS} to {MAX_PHONE_DIGITS} digits")
        return value

    @model_validator(mode="after")
    def _blank_out_non_resumes(self) -> Self:
        """If the document is not a resume, reset every other field to its default."""
        if not self.is_resume:
            for name, info in type(self).model_fields.items():
                if name != "is_resume":
                    setattr(self, name, info.get_default(call_default_factory=True))
        return self
