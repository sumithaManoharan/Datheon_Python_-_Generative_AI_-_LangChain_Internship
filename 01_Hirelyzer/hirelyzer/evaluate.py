"""Score extraction output against ground truth and track the results from run to run.

Run these from the 01_Hirelyzer folder:
    python -m hirelyzer.evaluate                      # run the model on the synthetic resumes
    python -m hirelyzer.evaluate --outputs some/dir   # score saved outputs, no API calls
    python -m hirelyzer.evaluate --unlabeled my/dir   # resumes without ground truth
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from hirelyzer.cli import build_extractor, extract_one, write_result
from hirelyzer.config import Settings, load_settings
from hirelyzer.errors import HirelyzerError
from hirelyzer.extractor import ExtractionResult
from hirelyzer.prompts import EXAMPLES_FILE, PROMPTS_DIR, TEMPLATE_FILE
from hirelyzer.readers import read_document
from hirelyzer.schema import Education, Experience, Project, Resume
from hirelyzer.telemetry import (
    TELEMETRY_LOGGER_NAME,
    TelemetryRecorder,
    configure_telemetry_log,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 2
DEFAULT_RESUMES_DIR = Path("data/synthetic/resumes")
DEFAULT_TRUTH_DIR = Path("data/synthetic/ground_truth")
DEFAULT_RUNS_DIR = Path("evaluation/runs")
HISTORY_FILE = "history.csv"
RESULT_FILE = "result.json"
OUTPUTS_SUBDIR = "outputs"
TELEMETRY_FILE = "telemetry.jsonl"
KNOWN_EXTENSIONS = (".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png")
LOWEST_SHOWN = 8  # how many weakest documents the report lists
NOISY_LOGGERS = ("httpx", "httpx2", "httpcore")
COMPLETENESS_FIELDS = ("name", "email", "phone", "location", "skills", "experience", "education")

FLAT_FIELDS = ("language", "name", "headline", "email", "phone", "location")
LIST_FIELDS = ("skills", "certifications", "languages")

# The ground-truth files name some fields differently from the pipeline's schema.
# Each schema field lists the alternative names to try when the primary one is absent.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("full_name",),
    "job_title": ("title",),
    "job_location": ("location",),
    "field_of_study": ("field",),
    "project_title": ("title", "name"),
    "project_description": ("description",),
    "skills_used": ("technologies", "tech_stack", "skills"),
}

# The pipeline reports the resume language as a name ("English"); ground truth uses codes ("en").
LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "portuguese": "pt",
    "tamil": "ta",
    "marathi": "mr",
}

NESTED_MODELS: dict[str, type[BaseModel]] = {
    "experience": Experience,
    "education": Education,
    "projects": Project,
}


@dataclass(frozen=True)
class EntrySpec:
    """How to score a list of entries (jobs, schools, projects)."""

    key_fields: tuple[str, ...]  # fields that identify an entry, used to pair prediction and truth
    value_fields: tuple[str, ...]  # fields compared once an entry is paired
    list_field: str | None = None  # a list inside each entry, scored as a set


ENTRY_SPECS = {
    "experience": EntrySpec(
        ("company", "job_title"),
        ("job_location", "start_date", "end_date", "is_current"),
        "highlights",
    ),
    "education": EntrySpec(
        ("institution", "degree"), ("field_of_study", "start_year", "end_year", "grade")
    ),
    "projects": EntrySpec(("project_title",), (), "skills_used"),
}


def _metric_names() -> list[str]:
    """Every metric a scored resume produces, in the order the report shows them."""
    names = ["is_resume", *FLAT_FIELDS, "skills_precision", "skills_recall"]
    names += [f"{name}_f1" for name in LIST_FIELDS]
    for entry, spec in ENTRY_SPECS.items():
        names.append(f"{entry}_f1")
        if spec.value_fields:
            names.append(f"{entry}_fields")
        if spec.list_field:
            names.append(f"{entry}_{spec.list_field}_f1")
    return names


METRICS = _metric_names()
HISTORY_COLUMNS = [
    "run_id",
    "mode",
    "model",
    "prompt_hash",
    "documents",
    "failed",
    "overall",
    "overall_processed",
    *METRICS,
]


@dataclass
class RunRecord:
    """Everything saved about one evaluation run."""

    run_id: str
    mode: str  # "live" (model was called) or "outputs" (saved outputs were scored)
    model: str
    temperature: float | None
    prompt_hash: str
    documents: int
    failed: int
    overall: float  # mean of all metrics, failed documents counted as zero
    metrics: dict[str, float]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    avg_latency_ms: float = 0.0
    overall_processed: float = 0.0  # same, but only over documents the pipeline produced
    metrics_processed: dict[str, float] = field(default_factory=dict)
    unrecognized_truth_keys: list[str] = field(default_factory=list)
    per_document: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------- scoring


def mean(values: Iterable[float]) -> float:
    """Average of the values, or 0.0 when there are none."""
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def normalize(value: Any) -> str:
    """Lower-case, collapse whitespace; None becomes an empty string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return " ".join(str(value).casefold().split())


def normalize_field(name: str, value: Any) -> str:
    """Normalize one field: phones by digits only, language names to their codes."""
    text = normalize(value)
    if name == "phone":
        return "".join(char for char in text if char.isdigit())
    if name == "language":
        return LANGUAGE_CODES.get(text, text)
    return text


def field_value(entry: Mapping[str, Any], name: str) -> Any:
    """Read a field, falling back to its aliases when the primary name is absent."""
    if name in entry:
        return entry[name]
    for alias in FIELD_ALIASES.get(name, ()):
        if alias in entry:
            return entry[alias]
    return None


def to_set(values: Any) -> set[str]:
    """Turn a list of values into a set of normalized, non-empty strings."""
    return {item for item in (normalize(value) for value in (values or [])) if item}


def precision_recall_f1(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    """Set-based precision, recall and F1. Both sets empty counts as a perfect match."""
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    hits = len(predicted & expected)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(expected) if expected else 0.0
    total = precision + recall
    return precision, recall, (2 * precision * recall / total if total else 0.0)


def score_entries(
    predicted: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]], spec: EntrySpec
) -> dict[str, float]:
    """Pair predicted and expected entries by key, then score the pairs."""

    def key(entry: Mapping[str, Any]) -> str:
        return " | ".join(normalize(field_value(entry, name)) for name in spec.key_fields)

    found = {key(entry): entry for entry in predicted}
    wanted = {key(entry): entry for entry in expected}
    pairs = [(found[k], wanted[k]) for k in wanted if k in found]
    # With no pairs, the detail scores are 1.0 if nothing was expected, otherwise 0.0.
    fallback = 0.0 if wanted else 1.0

    scores = {"f1": precision_recall_f1(set(found), set(wanted))[2]}
    if spec.value_fields:
        per_pair = [
            mean(
                float(normalize(field_value(a, name)) == normalize(field_value(b, name)))
                for name in spec.value_fields
            )
            for a, b in pairs
        ]
        scores["fields"] = mean(per_pair) if per_pair else fallback
    list_field = spec.list_field
    if list_field:
        per_pair = [
            precision_recall_f1(
                to_set(field_value(a, list_field)), to_set(field_value(b, list_field))
            )[2]
            for a, b in pairs
        ]
        scores[f"{list_field}_f1"] = mean(per_pair) if per_pair else fallback
    return scores


def score_document(
    predicted: Mapping[str, Any] | None, truth: Mapping[str, Any]
) -> dict[str, float]:
    """Score one document. A predicted value of None means the pipeline failed on it."""
    if predicted is None:
        return dict.fromkeys(METRICS, 0.0)
    scores = {"is_resume": float(bool(predicted.get("is_resume")) == bool(truth["is_resume"]))}
    if not truth["is_resume"]:
        return scores  # a non-resume has no other fields to compare

    for name in FLAT_FIELDS:
        got = normalize_field(name, field_value(predicted, name))
        want = normalize_field(name, field_value(truth, name))
        scores[name] = float(got == want)
    precision, recall, _ = precision_recall_f1(
        to_set(predicted.get("skills")), to_set(truth.get("skills"))
    )
    scores["skills_precision"], scores["skills_recall"] = precision, recall
    for name in LIST_FIELDS:
        scores[f"{name}_f1"] = precision_recall_f1(
            to_set(predicted.get(name)), to_set(truth.get(name))
        )[2]
    for entry, spec in ENTRY_SPECS.items():
        entry_scores = score_entries(predicted.get(entry) or [], truth.get(entry) or [], spec)
        for suffix, value in entry_scores.items():
            scores[f"{entry}_{suffix}"] = value
    return scores


def aggregate(
    documents: Mapping[str, Mapping[str, float]], resume_docs: set[str]
) -> dict[str, float]:
    """Average each metric: is_resume over all documents, the rest over true resumes only."""
    metrics = {"is_resume": mean(scores["is_resume"] for scores in documents.values())}
    for name in METRICS[1:]:
        metrics[name] = mean(documents[doc][name] for doc in resume_docs)
    return metrics


def evaluate(
    predictions: Mapping[str, Mapping[str, Any] | None], truth: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Return per-document scores and the aggregated metrics."""
    documents = {
        name: score_document(predictions.get(name), doc) for name, doc in sorted(truth.items())
    }
    resume_docs = {name for name, doc in truth.items() if doc["is_resume"]}
    return documents, aggregate(documents, resume_docs)


# ---------------------------------------------------------------- loading


def document_key(name: str) -> str:
    """Match files across folders: '01_x.pdf.json', '01_x.json' and '01_x.pdf' all give '01_x'."""
    path = Path(name)
    if path.suffix.lower() == ".json":
        path = path.with_suffix("")
    if path.suffix.lower() in KNOWN_EXTENSIONS:
        path = path.with_suffix("")
    return path.name


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a file, with readable errors."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HirelyzerError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise HirelyzerError(f"{path} must contain a JSON object")
    return data


def load_truth(truth_dir: Path) -> dict[str, dict[str, Any]]:
    """Load every ground-truth file; each one must say whether it is a resume."""
    files = sorted(truth_dir.glob("*.json")) if truth_dir.is_dir() else []
    if not files:
        raise HirelyzerError(f"No ground-truth .json files found in {truth_dir}")
    truth = {document_key(path.name): load_json(path) for path in files}
    for name, doc in truth.items():
        if "is_resume" not in doc:
            raise HirelyzerError(f"Ground truth for {name} has no 'is_resume' field")
    return truth


def load_predictions(outputs_dir: Path) -> dict[str, dict[str, Any]]:
    """Load saved pipeline outputs (a folder of .json files)."""
    if not outputs_dir.is_dir():
        raise HirelyzerError(f"Outputs folder not found: {outputs_dir}")
    return {document_key(path.name): load_json(path) for path in sorted(outputs_dir.glob("*.json"))}


def known_keys(model: type[BaseModel]) -> set[str]:
    """The model's field names plus every alias the scorer understands for them."""
    names = set(model.model_fields)
    return names | {alias for name in names for alias in FIELD_ALIASES.get(name, ())}


def unrecognized_keys(truth: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Ground-truth keys the scorer does not use; they are reported once and not scored."""
    unknown: set[str] = set()
    top_level = known_keys(Resume)
    for doc in truth.values():
        if not doc.get("is_resume"):
            continue  # non-resumes carry their own fields (invoice number, total, ...)
        unknown |= {key for key in doc if key not in top_level}
        for section, model in NESTED_MODELS.items():
            allowed = known_keys(model)
            for entry in doc.get(section) or []:
                unknown |= {f"{section}.{key}" for key in entry if key not in allowed}
    return sorted(unknown)


def prompt_hash(prompts_dir: Path = PROMPTS_DIR) -> str:
    """Short fingerprint of the prompt files, so a score change can be tied to a prompt change."""
    digest = hashlib.sha256()
    for name in (TEMPLATE_FILE, EXAMPLES_FILE):
        digest.update((prompts_dir / name).read_bytes())
    return digest.hexdigest()[:8]


# ---------------------------------------------------------------- running the pipeline


def run_pipeline(
    resumes_dir: Path, settings: Settings, outputs_dir: Path | None = None
) -> dict[str, ExtractionResult]:
    """Run the real pipeline on every file in a folder; optionally save each JSON output."""
    if not resumes_dir.is_dir():
        raise HirelyzerError(f"Resumes folder not found: {resumes_dir}")
    extractor = build_extractor(settings, TelemetryRecorder())
    results: dict[str, ExtractionResult] = {}
    for path in sorted(p for p in resumes_dir.iterdir() if p.is_file()):
        loader = partial(read_document, path, settings.max_file_bytes, settings.max_text_chars)
        result = extract_one(extractor, path.name, loader)
        if result.status == "success" and outputs_dir is not None:
            write_result(result, outputs_dir)
        results[document_key(path.name)] = result
        logger.info(
            "%s: %s (%d attempt(s), %.0f ms)",
            path.name,
            result.status,
            result.attempts,
            result.latency_ms,
        )
    return results


def dump(result: ExtractionResult) -> dict[str, Any] | None:
    """The result's data as a plain dict, or None when the pipeline failed."""
    if result.status != "success" or result.data is None:
        return None
    return result.data.model_dump(mode="json")


# ---------------------------------------------------------------- history and reports


def append_history(runs_dir: Path, record: RunRecord) -> dict[str, str] | None:
    """Append this run to history.csv and return the previous row, if there was one."""
    path = runs_dir / HISTORY_FILE
    previous = None
    is_new = not path.exists() or path.stat().st_size == 0
    if not is_new:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        previous = rows[-1] if rows else None
    row = {
        "run_id": record.run_id,
        "mode": record.mode,
        "model": record.model,
        "prompt_hash": record.prompt_hash,
        "documents": record.documents,
        "failed": record.failed,
        "overall": round(record.overall, 4),
        "overall_processed": round(record.overall_processed, 4),
        **{name: round(record.metrics[name], 4) for name in METRICS},
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return previous


def format_report(record: RunRecord, previous: Mapping[str, str] | None) -> str:
    """Build the text report: each metric over all documents and over processed ones only."""
    lines = [
        f"Run {record.run_id} ({record.mode}) | model: {record.model} | "
        f"prompt: {record.prompt_hash} | {record.documents} documents, {record.failed} failed",
        "",
        f"{'metric':<28}{'all docs':>10}{'processed':>11}{'vs previous':>13}",
    ]
    rows = [("overall", record.overall, record.overall_processed)]
    rows += [
        (name, value, record.metrics_processed.get(name, 0.0))
        for name, value in record.metrics.items()
    ]
    for name, value, processed in rows:
        before = (previous or {}).get(name, "")
        delta = f"{value - float(before):+.3f}" if before else ""
        lines.append(f"{name:<28}{value:>10.3f}{processed:>11.3f}{delta:>13}")
    lines.append(
        "\n'all docs' counts a failed document as zero; 'processed' skips documents the "
        "pipeline could not produce."
    )
    if record.mode == "live":
        lines.append(
            f"Tokens: {record.prompt_tokens} prompt + {record.completion_tokens} completion"
            f" | Avg latency: {record.avg_latency_ms:.0f} ms"
        )
    weakest = sorted(record.per_document.items(), key=lambda item: mean(item[1].values()))
    shown = [
        f"  {name}: {mean(scores.values()):.2f}"
        for name, scores in weakest[:LOWEST_SHOWN]
        if mean(scores.values()) < 1.0
    ]
    if shown:
        lines += ["", "Lowest-scoring documents:", *shown]
    if previous:
        lines.append(
            f"\nCompared with run {previous.get('run_id')} (prompt {previous.get('prompt_hash')})."
        )
    return "\n".join(lines)


def completeness_report(results: Mapping[str, ExtractionResult]) -> str:
    """Summarize a run on resumes with no ground truth. Counts only, never resume content."""
    parsed = [r.data for r in results.values() if r.status == "success" and r.data is not None]
    resumes = [data for data in parsed if data.is_resume]
    repaired = sum(1 for r in results.values() if r.status == "success" and r.attempts > 1)
    total = len(results)
    lines = [
        f"Documents: {total} | valid JSON: {len(parsed)} | failed: {total - len(parsed)}",
        f"Classified as resumes: {len(resumes)} | needed a repair round: {repaired}",
        "Empty fields among the resumes:",
    ]
    for name in COMPLETENESS_FIELDS:
        empty = sum(1 for data in resumes if not getattr(data, name))
        lines.append(f"  {name}: {empty}/{len(resumes)} empty")
    prompt_tokens = sum(r.prompt_tokens for r in results.values())
    completion_tokens = sum(r.completion_tokens for r in results.values())
    average_ms = mean(r.latency_ms for r in results.values())
    lines.append(
        f"Tokens: {prompt_tokens} prompt + {completion_tokens} completion"
        f" | Avg latency: {average_ms:.0f} ms"
    )
    failed_names = sorted(name for name, r in results.items() if r.status != "success")
    if failed_names:
        lines.append("Failed documents: " + ", ".join(failed_names))
    lines.append(
        "Note: this measures validity and completeness, not correctness. "
        "Accuracy needs ground truth."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------- command line


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="evaluate", description="Evaluate extraction quality against ground truth."
    )
    parser.add_argument("--resumes", type=Path, default=DEFAULT_RESUMES_DIR)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_DIR)
    parser.add_argument(
        "--outputs", type=Path, help="Score saved JSON outputs instead of calling the model."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Where runs and history.csv are saved (default: %(default)s).",
    )
    parser.add_argument(
        "--unlabeled",
        type=Path,
        help="Folder of resumes with no ground truth: print a completeness report, save nothing.",
    )
    return parser.parse_args(argv)


def run_labeled(args: argparse.Namespace) -> int:
    """Score the pipeline (or saved outputs) against ground truth and record the run."""
    truth = load_truth(args.truth)
    unknown = unrecognized_keys(truth)
    if unknown:
        print(
            f"Note: ground truth has keys the pipeline does not extract (not scored): {unknown}",
            file=sys.stderr,
        )

    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, ExtractionResult] = {}
    if args.outputs is not None:
        predictions: dict[str, Mapping[str, Any] | None] = dict(load_predictions(args.outputs))
        mode, model, temperature = "outputs", "unknown", None
    else:
        settings = load_settings()
        configure_telemetry_log(run_dir / TELEMETRY_FILE)
        results = run_pipeline(args.resumes, settings, run_dir / OUTPUTS_SUBDIR)
        predictions = {name: dump(result) for name, result in results.items()}
        mode, model, temperature = "live", settings.text_model, settings.temperature

    extra = sorted(set(predictions) - set(truth))
    if extra:
        print(
            f"Warning: no ground truth for these documents (not scored): {extra}",
            file=sys.stderr,
        )

    documents, metrics = evaluate(predictions, truth)
    # The same scores again, leaving out documents the pipeline could not produce at all.
    produced = {name for name in truth if predictions.get(name) is not None}
    processed = aggregate(
        {name: documents[name] for name in produced},
        {name for name in produced if truth[name]["is_resume"]},
    )
    record = RunRecord(
        run_id=run_id,
        mode=mode,
        model=model,
        temperature=temperature,
        prompt_hash=prompt_hash(),
        documents=len(truth),
        failed=len(truth) - len(produced),
        overall=mean(metrics.values()),
        metrics=metrics,
        prompt_tokens=sum(r.prompt_tokens for r in results.values()),
        completion_tokens=sum(r.completion_tokens for r in results.values()),
        avg_latency_ms=mean(r.latency_ms for r in results.values()),
        overall_processed=mean(processed.values()),
        metrics_processed=processed,
        unrecognized_truth_keys=unknown,
        per_document=documents,
    )
    (run_dir / RESULT_FILE).write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    previous = append_history(args.runs_dir, record)

    print(format_report(record, previous))
    for name, result in sorted(results.items()):
        if result.status != "success":
            print(f"FAILED {name}: {result.errors[-1] if result.errors else 'unknown error'}")
    return EXIT_OK


def run_unlabeled(resumes_dir: Path) -> int:
    """Run the pipeline on unlabeled resumes and print a completeness report only."""
    settings = load_settings()
    # Per-request telemetry lines name the files, so keep them off the console.
    logging.getLogger(TELEMETRY_LOGGER_NAME).setLevel(logging.WARNING)
    print(completeness_report(run_pipeline(resumes_dir, settings)))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the exit code."""
    args = parse_args(argv)
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        if args.unlabeled is not None:
            return run_unlabeled(args.unlabeled)
        return run_labeled(args)
    except HirelyzerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
