"""Per-request telemetry: latency, token usage and retries, logged as JSON lines."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_TELEMETRY_LOG = "logs/telemetry.jsonl"
TELEMETRY_LOGGER_NAME = "hirelyzer.telemetry"

telemetry_logger = logging.getLogger(TELEMETRY_LOGGER_NAME)


@dataclass(frozen=True)
class RequestRecord:
    """Everything worth knowing about one LLM request."""

    source: str  # file name or "text-input"
    model: str
    pipeline: str  # "text" now, "vision" later
    latency_ms: float  # total time including any retry waits
    prompt_tokens: int
    completion_tokens: int
    retries: int  # attempts beyond the first
    success: bool
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def configure_telemetry_log(path: Path) -> logging.Handler:
    """Send telemetry JSON lines to a file. Call once at startup (the CLI does this)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop handlers from an earlier call, so repeated runs never write duplicate lines.
    for old in telemetry_logger.handlers[:]:
        telemetry_logger.removeHandler(old)
        old.close()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))  # the message is already JSON
    telemetry_logger.addHandler(handler)
    telemetry_logger.setLevel(logging.INFO)
    telemetry_logger.propagate = False  # keep the JSON lines out of the console output
    return handler


class TelemetryRecorder:
    """Collects request records in memory and logs each one as a JSON line."""

    def __init__(self) -> None:
        """Start with no records."""
        self._records: list[RequestRecord] = []

    @property
    def records(self) -> list[RequestRecord]:
        """A copy of the records so far."""
        return list(self._records)

    def record(self, entry: RequestRecord) -> None:
        """Store one record and write its JSON line."""
        self._records.append(entry)
        data = asdict(entry)
        data["latency_ms"] = round(data["latency_ms"], 1)
        telemetry_logger.info(json.dumps(data, ensure_ascii=False))

    def summary(self) -> str:
        """Build the end-of-run telemetry text for the CLI."""
        records = self._records
        if not records:
            return "LLM requests: 0"
        ok = sum(1 for r in records if r.success)
        retries = sum(r.retries for r in records)
        average_ms = sum(r.latency_ms for r in records) / len(records)
        prompt = sum(r.prompt_tokens for r in records)
        completion = sum(r.completion_tokens for r in records)
        return (
            f"LLM requests: {len(records)} ({ok} ok, {len(records) - ok} failed) "
            f"| retries: {retries}\n"
            f"Avg request latency: {average_ms:.0f} ms "
            f"| Tokens: {prompt} prompt + {completion} completion"
        )
