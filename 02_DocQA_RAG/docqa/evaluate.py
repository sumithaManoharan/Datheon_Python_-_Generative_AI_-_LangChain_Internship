"""Retrieval evaluation: score how well the store ranks relevant chunks per question.

Relevance comes from each question's keywords (see docqa.retrieval_metrics).
Run with: ``uv run python -m docqa.evaluate``.
"""

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from docqa.config import PROJECT_ROOT, load_settings
from docqa.embeddings import OpenAIEmbedder
from docqa.errors import DocQAError, EvaluationError
from docqa.logging_setup import configure_logging
from docqa.retrieval_metrics import (
    hit_text,
    keyword_recall,
    keyword_share,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)
from docqa.store import SearchHit, VectorStore

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS = PROJECT_ROOT / "evaluation" / "tests.jsonl"
DEFAULT_KS = (4, 10)
DEFAULT_WORST = 10
OVERALL = "ALL"
TOP_HITS_SHOWN = 3

Search = Callable[[str, int], list[SearchHit]]


@dataclass(frozen=True)
class Question:
    """One evaluation question with the keywords that signal a relevant chunk."""

    question: str
    keywords: tuple[str, ...]
    category: str


@dataclass(frozen=True)
class QuestionScore:
    """Retrieval metrics for one question at one cutoff k."""

    question: str
    category: str
    keywords: tuple[str, ...]
    k: int
    recall: float
    precision: float
    mrr: float
    ndcg: float
    top_hits: tuple[str, ...]


@dataclass(frozen=True)
class Summary:
    """Mean metrics for one category (or for all questions)."""

    category: str
    n: int
    recall: float
    precision: float
    mrr: float
    ndcg: float


def _parse_row(row: Any, where: str) -> Question:
    """Validate one decoded JSON line."""
    if not isinstance(row, dict):
        raise EvaluationError(f"{where}: expected a JSON object")
    text = row.get("question")
    keywords = row.get("keywords")
    if not isinstance(text, str) or not text.strip():
        raise EvaluationError(f"{where}: missing question text")
    if (
        not isinstance(keywords, list)
        or not keywords
        or not all(isinstance(word, str) and word.strip() for word in keywords)
    ):
        raise EvaluationError(f"{where}: keywords must be a non-empty list of strings")
    category = row.get("category")
    return Question(
        question=text.strip(),
        keywords=tuple(word.strip() for word in keywords),
        category=category if isinstance(category, str) and category else "uncategorized",
    )


def load_questions(path: Path) -> list[Question]:
    """Read a JSON-lines question file, raising EvaluationError with the line number."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EvaluationError(f"cannot read questions file {path}: {exc}") from exc
    questions: list[Question] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc
        questions.append(_parse_row(row, f"{path}:{number}"))
    if not questions:
        raise EvaluationError(f"no questions found in {path}")
    return questions


def _describe(hit: SearchHit) -> str:
    """Short 'file | location' label for a hit."""
    location = f"page {hit.page}" if hit.page is not None else (hit.section or "no section")
    return f"{Path(hit.source).name} | {location}"


def score_question(question: Question, hits: Sequence[SearchHit], k: int) -> QuestionScore:
    """Score the top k hits for one question."""
    top = list(hits[:k])
    texts = [hit_text(hit) for hit in top]
    shares = [keyword_share(text, question.keywords) for text in texts]
    return QuestionScore(
        question=question.question,
        category=question.category,
        keywords=question.keywords,
        k=k,
        recall=keyword_recall(texts, question.keywords),
        precision=precision_at_k(shares, k),
        mrr=reciprocal_rank(shares, k),
        ndcg=ndcg_at_k(shares, k),
        top_hits=tuple(_describe(hit) for hit in top[:TOP_HITS_SHOWN]),
    )


def evaluate_retrieval(
    search: Search, questions: Sequence[Question], ks: Sequence[int]
) -> list[QuestionScore]:
    """Search once per question at the deepest k, then score every requested cutoff."""
    if not ks or any(k < 1 for k in ks):
        raise EvaluationError("ks must be a non-empty list of positive integers")
    deepest = max(ks)
    scores: list[QuestionScore] = []
    for index, question in enumerate(questions, start=1):
        hits = search(question.question, deepest)
        scores.extend(score_question(question, hits, k) for k in ks)
        if index % 25 == 0:
            logger.info("scored %d of %d questions", index, len(questions))
    return scores


def _mean_row(category: str, group: Sequence[QuestionScore]) -> Summary:
    """Average the metrics of one group."""
    return Summary(
        category=category,
        n=len(group),
        recall=statistics.fmean(s.recall for s in group),
        precision=statistics.fmean(s.precision for s in group),
        mrr=statistics.fmean(s.mrr for s in group),
        ndcg=statistics.fmean(s.ndcg for s in group),
    )


def summarize(scores: Sequence[QuestionScore], k: int) -> list[Summary]:
    """Mean metrics per category (alphabetical) followed by an overall row, at one k."""
    groups: dict[str, list[QuestionScore]] = defaultdict(list)
    overall: list[QuestionScore] = []
    for score in scores:
        if score.k == k:
            groups[score.category].append(score)
            overall.append(score)
    rows = [_mean_row(name, groups[name]) for name in sorted(groups)]
    if overall:
        rows.append(_mean_row(OVERALL, overall))
    return rows


def _worst_section(scores: Sequence[QuestionScore], k: int, limit: int) -> str:
    """Questions with no relevant chunk in the top k, lowest recall first."""
    at_k = [score for score in scores if score.k == k]
    failing = sorted((s for s in at_k if s.mrr == 0.0), key=lambda s: (s.recall, s.question))
    lines = [f"### No relevant chunk in the top {k}: {len(failing)} of {len(at_k)} questions"]
    if not failing:
        lines.append("\nnone")
    for score in failing[:limit]:
        lines.append(
            f"\n- [{score.category}] {score.question}\n"
            f"  keywords: {', '.join(score.keywords)} | recall {score.recall:.2f}\n"
            f"  top hits: {'; '.join(score.top_hits) or 'none'}"
        )
    return "\n".join(lines)


def format_report(
    scores: Sequence[QuestionScore], ks: Sequence[int], worst: int = DEFAULT_WORST
) -> str:
    """Markdown report: one table per k, then the worst-ranked questions at the first k."""
    sections = []
    for k in ks:
        lines = [
            f"### Retrieval at k={k}",
            "",
            "| category | n | recall | precision | MRR | nDCG |",
            "|---|---|---|---|---|---|",
        ]
        lines.extend(
            f"| {row.category} | {row.n} | {row.recall:.3f} | {row.precision:.3f} "
            f"| {row.mrr:.3f} | {row.ndcg:.3f} |"
            for row in summarize(scores, k)
        )
        sections.append("\n".join(lines))
    sections.append(_worst_section(scores, ks[0], worst))
    return "\n\n".join(sections) + "\n"


def _positive_ints(value: str) -> tuple[int, ...]:
    """argparse type: '4,10' becomes (4, 10)."""
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
    parser = argparse.ArgumentParser(description="Score retrieval quality on a question file.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--ks", type=_positive_ints, default=DEFAULT_KS, help="e.g. 4,10")
    parser.add_argument("--worst", type=int, default=DEFAULT_WORST)
    parser.add_argument("--out", type=Path, default=None, help="write per-question JSON here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation against the persisted store; return a process exit code."""
    args = parse_args(argv)
    load_dotenv(find_dotenv(usecwd=True))
    settings = load_settings()
    configure_logging(settings.log_level)
    try:
        questions = load_questions(args.questions)
        client = OpenAI(
            base_url=settings.base_url, api_key=settings.require_api_key(), max_retries=0
        )
        embedder = OpenAIEmbedder(
            client,
            settings.embedding_model,
            batch_size=settings.embed_batch_size,
            max_attempts=settings.max_attempts,
            base_delay=settings.retry_base_delay,
        )
        store = VectorStore(settings.store_dir, embedder)
        if store.count() == 0:
            raise EvaluationError("the vector store is empty; ingest the documents first")
        scores = evaluate_retrieval(store.search, questions, args.ks)
    except DocQAError as exc:
        logger.error("evaluation failed: %s", exc)
        return 1
    sys.stdout.write(format_report(scores, args.ks, args.worst))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([asdict(s) for s in scores], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
