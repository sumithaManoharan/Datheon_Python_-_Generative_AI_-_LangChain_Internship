"""Answer-level evaluation: run questions through the whole pipeline and score the answers.

Compares retrieval depths (top_k) by the share of a question's keywords found in the answer,
and checks that unanswerable questions are refused. Keyword scoring is an indicator, not
ground truth: a paraphrase or a different number format can miss a keyword.
Run with: ``uv run python -m docqa.evaluate_answers``. It makes real API calls.
"""

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from docqa.answer import Answer
from docqa.config import PROJECT_ROOT, Settings, load_settings
from docqa.errors import DocQAError, EvaluationError
from docqa.evaluate import DEFAULT_QUESTIONS, OVERALL, Question, load_questions
from docqa.logging_setup import configure_logging
from docqa.pipeline import build_pipeline
from docqa.retrieval_metrics import RELEVANT_SHARE, keyword_share

logger = logging.getLogger(__name__)

DEFAULT_KS = (6, 10)
EVAL_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_REFUSAL_FILES = (EVAL_DIR / "unanswerable.jsonl", EVAL_DIR / "near_topic.jsonl")
DEFAULT_SHOWN = 10
TEXT_PREVIEW = 160
PROGRESS_EVERY = 25

Ask = Callable[[str], Answer]
AskFactory = Callable[[Settings, int], Ask]
Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class AnswerRecord:
    """The pipeline's answer to one answerable question at one top_k."""

    question: str
    category: str
    keywords: tuple[str, ...]
    k: int
    text: str
    refused: bool
    cited: bool
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    error: str | None = None

    @property
    def coverage(self) -> float:
        """Share of the question's keywords found in the answer text."""
        return keyword_share(self.text, self.keywords)


@dataclass(frozen=True)
class AnswerSummary:
    """Aggregates over the answerable questions at one top_k (errors excluded)."""

    k: int
    n: int
    errors: int
    coverage: float
    full_rate: float
    hit_rate: float
    refusal_rate: float
    cited_rate: float
    mean_prompt_tokens: float
    mean_seconds: float


@dataclass(frozen=True)
class RefusalRecord:
    """The pipeline's answer to one question that the documents cannot answer."""

    source: str
    question: str
    k: int
    refused: bool
    llm_called: bool
    text: str
    error: str | None = None


def sample(questions: Sequence[Question], limit: int | None) -> list[Question]:
    """Every n-th question so a small limit still spans all categories."""
    if limit is None or limit >= len(questions):
        return list(questions)
    if limit < 1:
        raise EvaluationError("--limit must be at least 1")
    return list(questions[:: len(questions) // limit])[:limit]


def run_questions(
    ask: Ask,
    questions: Sequence[Question],
    k: int,
    clock: Callable[[], float] = time.perf_counter,
    progress: Progress | None = None,
) -> list[AnswerRecord]:
    """Ask every question once; a failed call is recorded and the run continues."""
    records: list[AnswerRecord] = []
    for index, question in enumerate(questions, start=1):
        started = clock()
        answer: Answer | None = None
        error: str | None = None
        try:
            answer = ask(question.question)
        except DocQAError as exc:
            logger.error("question failed at k=%d: %s", k, exc)
            error = str(exc)
        seconds = clock() - started
        records.append(
            AnswerRecord(
                question=question.question,
                category=question.category,
                keywords=question.keywords,
                k=k,
                text=answer.text if answer else "",
                refused=answer.refused if answer else False,
                cited=bool(answer.citations) if answer else False,
                prompt_tokens=(answer.prompt_tokens or 0) if answer else 0,
                completion_tokens=(answer.completion_tokens or 0) if answer else 0,
                seconds=seconds,
                error=error,
            )
        )
        if progress is not None:
            progress(index, len(questions))
    return records


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize_answers(records: Sequence[AnswerRecord], k: int) -> AnswerSummary:
    """Aggregate the records of one top_k; failed calls are counted but not averaged."""
    ok = [record for record in records if record.error is None]
    answered = [record for record in ok if not record.refused]
    return AnswerSummary(
        k=k,
        n=len(ok),
        errors=len(records) - len(ok),
        coverage=statistics.fmean(r.coverage for r in ok) if ok else 0.0,
        full_rate=_rate(sum(r.coverage == 1.0 for r in ok), len(ok)),
        hit_rate=_rate(sum(r.coverage >= RELEVANT_SHARE for r in ok), len(ok)),
        refusal_rate=_rate(sum(r.refused for r in ok), len(ok)),
        cited_rate=_rate(sum(r.cited for r in answered), len(answered)),
        mean_prompt_tokens=statistics.fmean(r.prompt_tokens for r in ok) if ok else 0.0,
        mean_seconds=statistics.fmean(r.seconds for r in ok) if ok else 0.0,
    )


def coverage_by_category(
    records_by_k: Mapping[int, Sequence[AnswerRecord]],
) -> dict[str, dict[int, float]]:
    """Mean keyword coverage per category and top_k, with an overall row last."""
    table: dict[str, dict[int, float]] = {}
    for k, records in records_by_k.items():
        groups: dict[str, list[float]] = defaultdict(list)
        for record in records:
            if record.error is None:
                groups[record.category].append(record.coverage)
                groups[OVERALL].append(record.coverage)
        for name, values in groups.items():
            table.setdefault(name, {})[k] = statistics.fmean(values)
    return dict(sorted(table.items(), key=lambda item: (item[0] == OVERALL, item[0])))


def biggest_differences(
    records_by_k: Mapping[int, Sequence[AnswerRecord]], first: int, last: int, limit: int
) -> list[tuple[str, float, float]]:
    """Questions whose coverage differs most between two top_k values: (question, first, last)."""
    by_question = {
        k: {r.question: r for r in records_by_k[k] if r.error is None} for k in (first, last)
    }
    rows = [
        (question, record.coverage, by_question[last][question].coverage)
        for question, record in by_question[first].items()
        if question in by_question[last]
    ]
    rows.sort(key=lambda row: (-abs(row[2] - row[1]), row[0]))
    return [row for row in rows[:limit] if row[1] != row[2]]


def load_refusal_questions(path: Path) -> list[str]:
    """Read questions the documents cannot answer: JSON lines of objects or plain strings."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EvaluationError(f"cannot read refusal questions {path}: {exc}") from exc
    questions: list[str] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc
        text = row.get("question") if isinstance(row, dict) else row
        if not isinstance(text, str) or not text.strip():
            raise EvaluationError(f"{path}:{number}: missing question text")
        questions.append(text.strip())
    if not questions:
        raise EvaluationError(f"no questions found in {path}")
    return questions


def check_refusals(ask: Ask, questions: Sequence[str], source: str, k: int) -> list[RefusalRecord]:
    """Ask each unanswerable question; `llm_called` is False when retrieval alone refused."""
    records: list[RefusalRecord] = []
    for question in questions:
        try:
            answer = ask(question)
        except DocQAError as exc:
            logger.error("refusal question failed at k=%d: %s", k, exc)
            records.append(RefusalRecord(source, question, k, False, False, "", str(exc)))
            continue
        records.append(
            RefusalRecord(
                source, question, k, answer.refused, (answer.prompt_tokens or 0) > 0, answer.text
            )
        )
    return records


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= TEXT_PREVIEW else flat[: TEXT_PREVIEW - 3] + "..."


def _answer_table(records_by_k: Mapping[int, Sequence[AnswerRecord]]) -> str:
    lines = [
        "### Answer quality by top_k",
        "",
        "| top_k | n | errors | keyword coverage | all keywords | at least half "
        "| wrongly refused | cited | prompt tokens | seconds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, records in records_by_k.items():
        s = summarize_answers(records, k)
        lines.append(
            f"| {k} | {s.n} | {s.errors} | {s.coverage:.3f} | {s.full_rate:.3f} "
            f"| {s.hit_rate:.3f} | {s.refusal_rate:.3f} | {s.cited_rate:.3f} "
            f"| {s.mean_prompt_tokens:.0f} | {s.mean_seconds:.2f} |"
        )
    return "\n".join(lines)


def _category_table(records_by_k: Mapping[int, Sequence[AnswerRecord]]) -> str:
    ks = list(records_by_k)
    lines = [
        "### Keyword coverage by category",
        "",
        "| category | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    for name, values in coverage_by_category(records_by_k).items():
        cells = " | ".join(f"{values[k]:.3f}" if k in values else "n/a" for k in ks)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


def _difference_section(
    records_by_k: Mapping[int, Sequence[AnswerRecord]], shown: int
) -> str | None:
    ks = list(records_by_k)
    if len(ks) < 2:
        return None
    first, last = ks[0], ks[-1]
    rows = biggest_differences(records_by_k, first, last, shown)
    lines = [f"### Answers that changed most between k={first} and k={last}"]
    if not rows:
        lines.append("\nnone")
    lines.extend(
        f"\n- {question}\n  coverage k={first}: {before:.2f} | k={last}: {after:.2f}"
        for question, before, after in rows
    )
    return "\n".join(lines)


def _wrongly_refused_section(records: Sequence[AnswerRecord], k: int, shown: int) -> str:
    refused = [r for r in records if r.refused and r.error is None]
    lines = [f"### Answerable questions refused at k={k}: {len(refused)}"]
    lines.extend(f"\n- [{r.category}] {r.question}" for r in refused[:shown])
    return "\n".join(lines)


def _refusal_section(refusals: Sequence[RefusalRecord]) -> str:
    groups: dict[tuple[int, str], list[RefusalRecord]] = defaultdict(list)
    for record in refusals:
        groups[(record.k, record.source)].append(record)
    lines = [
        "### Unanswerable questions",
        "",
        "| top_k | file | questions | refused | refused without LLM call | errors |",
        "|---|---|---|---|---|---|",
    ]
    for (k, source), group in groups.items():
        lines.append(
            f"| {k} | {source} | {len(group)} | {sum(r.refused for r in group)} "
            f"| {sum(r.refused and not r.llm_called for r in group)} "
            f"| {sum(r.error is not None for r in group)} |"
        )
    missed = [r for r in refusals if not r.refused and r.error is None]
    lines.append(f"\nNot refused: {len(missed)}")
    lines.extend(
        f"\n- [k={r.k}, {r.source}] {r.question}\n  answer: {_preview(r.text)}" for r in missed
    )
    return "\n".join(lines)


def format_report(
    records_by_k: Mapping[int, Sequence[AnswerRecord]],
    refusals: Sequence[RefusalRecord],
    shown: int = DEFAULT_SHOWN,
) -> str:
    """Markdown report for the whole answer-level evaluation."""
    last = list(records_by_k)[-1]
    sections = [_answer_table(records_by_k), _category_table(records_by_k)]
    differences = _difference_section(records_by_k, shown)
    if differences:
        sections.append(differences)
    sections.append(_wrongly_refused_section(records_by_k[last], last, shown))
    sections.append(_refusal_section(refusals))
    return "\n\n".join(sections) + "\n"


def _parse_ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if any(k < 1 for k in ks):
        raise argparse.ArgumentTypeError("every k must be at least 1")
    return ks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments (pass a list in tests)."""
    parser = argparse.ArgumentParser(description="Score full answers on a question file.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--refusal-files", type=Path, nargs="+", default=list(DEFAULT_REFUSAL_FILES)
    )
    parser.add_argument("--ks", type=_parse_ks, default=DEFAULT_KS, help="top_k values, e.g. 6,10")
    parser.add_argument("--limit", type=int, default=None, help="sample this many questions")
    parser.add_argument("--shown", type=int, default=DEFAULT_SHOWN)
    parser.add_argument("--out", type=Path, default=None, help="write every record as JSON here")
    return parser.parse_args(argv)


def build_ask(settings: Settings, k: int) -> Ask:
    """A real pipeline with this top_k, without writing usage telemetry."""
    pipeline = build_pipeline(settings.model_copy(update={"top_k": k}), record_telemetry=False)
    return pipeline.ask


def _report_progress(done: int, total: int) -> None:
    if done % PROGRESS_EVERY == 0 or done == total:
        sys.stderr.write(f"  answered {done} of {total}\n")


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    ask_factory: AskFactory = build_ask,
) -> int:
    """Run the evaluation; tests inject `settings` and a fake `ask_factory`."""
    args = parse_args(argv)
    try:
        questions = sample(load_questions(args.questions), args.limit)
        refusal_sets = [(path.stem, load_refusal_questions(path)) for path in args.refusal_files]
        if settings is None:
            load_dotenv(find_dotenv(usecwd=True))
            settings = load_settings()
        configure_logging("WARNING")  # keep the per-question INFO lines out of the report run
        records_by_k: dict[int, list[AnswerRecord]] = {}
        refusals: list[RefusalRecord] = []
        for k in args.ks:
            sys.stderr.write(f"top_k={k}\n")
            ask = ask_factory(settings, k)
            records_by_k[k] = run_questions(ask, questions, k, progress=_report_progress)
            for source, texts in refusal_sets:
                refusals.extend(check_refusals(ask, texts, source, k))
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "answers": [asdict(r) for records in records_by_k.values() for r in records],
                "refusals": [asdict(r) for r in refusals],
            }
            args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except (DocQAError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    sys.stdout.write(format_report(records_by_k, refusals, args.shown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
