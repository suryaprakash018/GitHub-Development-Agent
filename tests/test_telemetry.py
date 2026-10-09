"""Tests for structured telemetry and audit logging."""

import json
from pathlib import Path

from agent.telemetry.recorder import (
    TelemetryRecord,
    TelemetryRecorder,
    redact_secrets,
    sanitize_data_structure,
)


def test_redact_secrets() -> None:
    gemini_key = "AIzaSyD-1234567890abcdefghijklmnopqrstuv"
    raw = f"Error communicating with Gemini: key={gemini_key} token=ghp_1234567890abcdefghijklmnopqrstuvwxyz Bearer my-secret-token"
    sanitized = redact_secrets(raw)

    assert gemini_key not in sanitized
    assert "ghp_" not in sanitized
    assert "[REDACTED" in sanitized


def test_redact_custom_secrets() -> None:
    custom_secret = "my_custom_super_secret_value"
    text = f"Connecting with password: {custom_secret}"
    sanitized = redact_secrets(text, custom_secrets=[custom_secret])

    assert custom_secret not in sanitized
    assert "[REDACTED_SECRET]" in sanitized


def test_sanitize_nested_data_structure() -> None:
    secret = "AIzaSyTestKey1234567890123456789012"
    nested = {
        "headers": {"Authorization": f"Bearer {secret}"},
        "query": [f"key={secret}", "normal_value"],
    }
    cleaned = sanitize_data_structure(nested)

    assert secret not in json.dumps(cleaned)
    assert cleaned["query"][1] == "normal_value"


def test_telemetry_record_atomic_persistence(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    custom_secret = "sensitive_database_password_999"
    recorder = TelemetryRecorder(runs_dir=runs_dir, custom_secrets=[custom_secret])

    record = TelemetryRecord(
        run_id="run_test_01",
        task_id="ms_ds_01_01",
        task_title="Imputation Module",
        status="DRY_RUN_APPROVED",
        duration_seconds=3.45,
        target_repo_path=str(tmp_path / "repo"),
        diff_summary={"files_changed": ["src/data.py"], "lines_added": 40, "lines_removed": 2},
        error=f"Attempted with secret {custom_secret}",
    )

    saved_path = recorder.record(record)
    assert saved_path.exists()
    assert saved_path.suffix == ".json"

    # Read back and verify content
    content = json.loads(saved_path.read_text(encoding="utf-8"))
    assert content["run_id"] == "run_test_01"
    assert content["task_id"] == "ms_ds_01_01"
    assert content["status"] == "DRY_RUN_APPROVED"
    assert custom_secret not in json.dumps(content)
    assert "[REDACTED_SECRET]" in content["error"]


def test_telemetry_get_recent_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    recorder = TelemetryRecorder(runs_dir=runs_dir)

    for i in range(3):
        rec = TelemetryRecord(
            run_id=f"run_0{i}",
            task_id=f"task_{i}",
            status="SUCCESS",
        )
        recorder.record(rec)

    recent = recorder.get_recent_runs(limit=2)
    assert len(recent) == 2
    assert all(isinstance(r, TelemetryRecord) for r in recent)
