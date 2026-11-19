"""Per-question telemetry: stage timings and token counts, one JSON line per question."""

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from docqa.config import Settings

logger = logging.getLogger(__name__)

TELEMETRY_FILENAME = "telemetry.jsonl"
P95 = 0.95


@dataclass(frozen=True)
class QueryRecord:
    """What one `ask` cost. The question text itself is never stored."""

    timestamp: str
    question_chars: int
    retrieved: int
    refused: bool
    retrieve_seconds: float
    generate_seconds: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_seconds(self) -> float:
        return self.retrieve_seconds + self.generate_seconds


@dataclass(frozen=True)
class Summary:
    """Aggregates over a set of records."""

    questions: int
    refused: int
    mean_retrieve: float
    mean_generate: float
    mean_total: float
    p95_total: float
    prompt_tokens: int
    completion_tokens: int


def utc_timestamp() -> str:
    """Current time in UTC, ISO 8601, to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def telemetry_path(settings: Settings) -> Path:
    """The log lives next to the vector store folder (data/telemetry.jsonl by default)."""
    return settings.store_dir.parent / TELEMETRY_FILENAME


class TelemetryLog:
    """Append-only JSON-lines log. Writing never raises: telemetry must not break an answer."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def record(self, record: QueryRecord) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record)) + "\n")
        except OSError as exc:
            logger.warning("could not write telemetry to %s: %s", self._path, exc)

    def read(self) -> list[QueryRecord]:
        """All valid records, oldest first. Missing file gives []; bad lines are skipped."""
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        except OSError as exc:
            logger.warning("could not read telemetry from %s: %s", self._path, exc)
            return []
        records: list[QueryRecord] = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append(QueryRecord(**json.loads(line)))
            except (ValueError, TypeError):
                logger.warning("skipping malformed telemetry line %d in %s", number, self._path)
        return records


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of a non-empty list."""
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(records: Iterable[QueryRecord]) -> Summary | None:
    """Aggregate records; None when there are none."""
    items = list(records)
    if not items:
        return None
    count = len(items)
    totals = [item.total_seconds for item in items]
    return Summary(
        questions=count,
        refused=sum(item.refused for item in items),
        mean_retrieve=sum(item.retrieve_seconds for item in items) / count,
        mean_generate=sum(item.generate_seconds for item in items) / count,
        mean_total=sum(totals) / count,
        p95_total=percentile(totals, P95),
        prompt_tokens=sum(item.prompt_tokens for item in items),
        completion_tokens=sum(item.completion_tokens for item in items),
    )
