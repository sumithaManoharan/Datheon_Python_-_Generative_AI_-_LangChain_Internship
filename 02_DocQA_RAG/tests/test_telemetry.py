"""Tests for docqa.telemetry."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from docqa.config import load_settings
from docqa.telemetry import (
    QueryRecord,
    TelemetryLog,
    percentile,
    summarize,
    telemetry_path,
    utc_timestamp,
)


def _record(retrieve=1.0, generate=2.0, refused=False, prompt=100, completion=20) -> QueryRecord:
    return QueryRecord(
        timestamp="2026-11-19T10:00:00+00:00",
        question_chars=12,
        retrieved=10,
        refused=refused,
        retrieve_seconds=retrieve,
        generate_seconds=generate,
        prompt_tokens=prompt,
        completion_tokens=completion,
    )


def test_record_appends_one_json_line_per_call(tmp_path: Path):
    log = TelemetryLog(tmp_path / "t.jsonl")
    log.record(_record())
    log.record(_record(refused=True))
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["prompt_tokens"] == 100


def test_record_creates_missing_parent_folders(tmp_path: Path):
    log = TelemetryLog(tmp_path / "a" / "b" / "t.jsonl")
    log.record(_record())
    assert len(log.read()) == 1


def test_read_round_trips_records_in_order(tmp_path: Path):
    log = TelemetryLog(tmp_path / "t.jsonl")
    first, second = _record(retrieve=1.0), _record(retrieve=2.0)
    log.record(first)
    log.record(second)
    assert log.read() == [first, second]


def test_read_of_a_missing_file_is_empty(tmp_path: Path):
    assert TelemetryLog(tmp_path / "none.jsonl").read() == []


def test_read_skips_malformed_lines_and_warns(tmp_path: Path, caplog):
    path = tmp_path / "t.jsonl"
    good = json.dumps(asdict(_record()))
    path.write_text(f"not json\n{good}\n[1, 2]\n{{}}\n\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="docqa.telemetry"):
        records = TelemetryLog(path).read()
    assert records == [_record()]
    assert "malformed" in caplog.text


def test_the_question_text_is_never_stored(tmp_path: Path):
    log = TelemetryLog(tmp_path / "t.jsonl")
    log.record(_record())
    data = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8"))
    assert "question" not in data
    assert data["question_chars"] == 12


def test_a_write_failure_is_logged_not_raised(tmp_path: Path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    log = TelemetryLog(blocker / "t.jsonl")
    with caplog.at_level(logging.WARNING, logger="docqa.telemetry"):
        log.record(_record())
    assert "could not write telemetry" in caplog.text


def test_percentile_uses_the_nearest_rank():
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0


def test_summarize_of_nothing_is_none():
    assert summarize([]) is None


def test_summarize_aggregates_timings_refusals_and_tokens():
    summary = summarize(
        [
            _record(retrieve=1.0, generate=2.0, prompt=100, completion=20),
            _record(retrieve=0.5, generate=0.0, refused=True, prompt=0, completion=0),
        ]
    )
    assert summary is not None
    assert (summary.questions, summary.refused) == (2, 1)
    assert summary.mean_retrieve == 0.75
    assert summary.mean_generate == 1.0
    assert summary.mean_total == 1.75
    assert summary.p95_total == 3.0
    assert (summary.prompt_tokens, summary.completion_tokens) == (100, 20)


def test_utc_timestamp_is_iso_8601_in_utc():
    assert utc_timestamp().endswith("+00:00")


def test_telemetry_path_sits_next_to_the_store_folder(tmp_path: Path):
    settings = load_settings({"DOCQA_STORE_DIR": str(tmp_path / "chroma")})
    assert telemetry_path(settings) == tmp_path / "telemetry.jsonl"
