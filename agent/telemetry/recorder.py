"""Structured telemetry and audit logger for autonomous development cycles."""

from __future__ import annotations

import contextlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent.utils.logger import setup_logger

logger = setup_logger("agent.telemetry")

# Regex patterns for sensitive tokens and credentials
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z-_]{20,50}"),  # Google/Gemini API key
    re.compile(r"gsk_[A-Za-z0-9_-]{20,80}"),  # Groq API key
    re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,50}"),  # GitHub PAT
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]+", re.IGNORECASE),  # Bearer tokens
    re.compile(
        r"(?:api[-_]?key|key|password|secret|token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{8,})['\"]?",
        re.IGNORECASE,
    ),
]


def redact_secrets(text: str, custom_secrets: list[str] | None = None) -> str:
    """Scrub sensitive credentials, tokens, and API keys from a string."""
    if not text:
        return text

    sanitized = text

    # Redact known explicit custom secrets if provided
    if custom_secrets:
        for secret in custom_secrets:
            if secret and len(secret) > 4:
                sanitized = sanitized.replace(secret, "[REDACTED_SECRET]")

    # Redact via regex patterns
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_CREDENTIAL]", sanitized)

    return sanitized


def sanitize_data_structure(data: Any, custom_secrets: list[str] | None = None) -> Any:
    """Recursively scrub secrets from nested dicts, lists, and strings."""
    if isinstance(data, str):
        return redact_secrets(data, custom_secrets)
    elif isinstance(data, dict):
        return {k: sanitize_data_structure(v, custom_secrets) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data_structure(item, custom_secrets) for item in data]
    return data


class TelemetryRecord(BaseModel):
    """Complete diagnostic audit record of an execution cycle."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    duration_seconds: float = 0.0
    mode: dict[str, bool] = Field(
        default_factory=lambda: {"dry_run": True, "auto_commit": False, "auto_push": False}
    )
    target_repo_path: str = ""
    target_branch: str = "main"
    task_id: str | None = None
    task_title: str | None = None
    status: str = "UNKNOWN"
    ast_metrics_before: dict[str, Any] | None = None
    ast_metrics_after: dict[str, Any] | None = None
    remediation_attempts: int = 0
    validation: dict[str, Any] = Field(
        default_factory=lambda: {
            "pytest_passed": False,
            "tests_run": 0,
            "failures": 0,
            "ruff_clean": False,
        }
    )
    quality_gate: dict[str, Any] = Field(
        default_factory=lambda: {
            "decision": "PENDING",
            "score": 0.0,
            "reasons": [],
        }
    )
    diff_summary: dict[str, Any] = Field(
        default_factory=lambda: {
            "files_changed": [],
            "lines_added": 0,
            "lines_removed": 0,
        }
    )
    commit_info: dict[str, Any] = Field(
        default_factory=lambda: {
            "proposed_subject": "",
            "created": False,
            "commit_hash": None,
            "pushed": False,
        }
    )
    error: str | None = None


class TelemetryRecorder:
    """Persists structured, sanitized execution audit records to disk."""

    def __init__(
        self,
        runs_dir: Path | str,
        custom_secrets: list[str] | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.custom_secrets = [s for s in (custom_secrets or []) if s]
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensures the telemetry directory exists."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def record(self, telemetry: TelemetryRecord) -> Path:
        """Sanitizes and atomically writes a TelemetryRecord to JSON."""
        self._ensure_dir()

        # Sanitize data
        raw_dict = telemetry.model_dump()
        sanitized_dict = sanitize_data_structure(raw_dict, self.custom_secrets)

        # Generate safe filename: <YYYYMMDD_HHMMSS>_<run_id>_<task_id>.json
        safe_time = (
            telemetry.timestamp_utc.replace(":", "")
            .replace("-", "")
            .replace("+00:00", "Z")
            .split(".")[0]
        )
        task_slug = (telemetry.task_id or "notask").replace("/", "_")
        filename = f"run_{safe_time}_{telemetry.run_id}_{task_slug}.json"
        target_path = self.runs_dir / filename

        # Atomic file write via temporary file
        temp_path = target_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(sanitized_dict, f, indent=2)
            temp_path.replace(target_path)
            logger.info(f"Persisted telemetry audit record: {target_path}")
            return target_path
        except Exception as e:
            if temp_path.exists():
                with contextlib.suppress(Exception):
                    temp_path.unlink()
            logger.error(f"Failed to persist telemetry audit record: {e}")
            raise

    def get_recent_runs(self, limit: int = 10) -> list[TelemetryRecord]:
        """Loads and returns recent telemetry records ordered latest first."""
        if not self.runs_dir.is_dir():
            return []

        json_files = sorted(self.runs_dir.glob("run_*.json"), reverse=True)
        records: list[TelemetryRecord] = []
        for file_path in json_files[:limit]:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                records.append(TelemetryRecord.model_validate(data))
            except Exception as e:
                logger.warning(f"Error reading telemetry file {file_path}: {e}")

        return records
