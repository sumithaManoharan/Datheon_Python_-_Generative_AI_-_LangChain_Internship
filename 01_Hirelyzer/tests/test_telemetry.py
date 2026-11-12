"""Tests for request telemetry records, the summary text and the JSON-lines log."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hirelyzer.telemetry import RequestRecord, TelemetryRecorder, configure_telemetry_log


def make_record(**overrides: Any) -> RequestRecord:
    """A successful record; keyword arguments override fields."""
    fields: dict[str, Any] = {
        "source": "a.pdf",
        "model": "m",
        "pipeline": "text",
        "latency_ms": 100.0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "retries": 0,
        "success": True,
    }
    return RequestRecord(**{**fields, **overrides})


def test_empty_recorder_summary() -> None:
    assert TelemetryRecorder().summary() == "LLM requests: 0"


def test_summary_aggregates_requests() -> None:
    recorder = TelemetryRecorder()
    recorder.record(make_record())
    recorder.record(
        make_record(
            latency_ms=300.0, prompt_tokens=0, completion_tokens=0, retries=2, success=False
        )
    )

    assert recorder.summary() == (
        "LLM requests: 2 (1 ok, 1 failed) | retries: 2\n"
        "Avg request latency: 200 ms | Tokens: 10 prompt + 5 completion"
    )


def test_records_property_returns_a_copy() -> None:
    recorder = TelemetryRecorder()
    recorder.record(make_record())

    recorder.records.clear()

    assert len(recorder.records) == 1


def test_timestamp_is_iso_format() -> None:
    assert datetime.fromisoformat(make_record().timestamp)


def test_each_record_is_written_as_one_json_line(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "t.jsonl"  # the parent folder is created for us
    configure_telemetry_log(log)

    TelemetryRecorder().record(make_record(latency_ms=12.3456))

    (line,) = log.read_text(encoding="utf-8").splitlines()
    data = json.loads(line)
    assert data["source"] == "a.pdf"
    assert data["latency_ms"] == 12.3
    assert data["prompt_tokens"] == 10
    assert data["success"] is True


def test_configuring_twice_does_not_duplicate_lines(tmp_path: Path) -> None:
    log = tmp_path / "t.jsonl"
    configure_telemetry_log(log)
    configure_telemetry_log(log)

    TelemetryRecorder().record(make_record())

    assert len(log.read_text(encoding="utf-8").splitlines()) == 1
