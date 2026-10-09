"""Persistent state manager with atomic JSON writes and SQLite run telemetry."""

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from agent.core.exceptions import StateManagerError
from agent.state.models import (
    AgentState,
    MilestoneProgress,
    MilestoneStatus,
    TaskExecutionRecord,
    TaskStatus,
)
from agent.utils.logger import setup_logger
from agent.utils.security import get_agent_root_path

logger = setup_logger("agent.state")


class StateManager:
    """Manages transactional state loading and saving."""

    def __init__(self, state_dir: Path | None = None) -> None:
        agent_root = get_agent_root_path()
        self.state_dir = state_dir or (agent_root / "state")
        self.state_file = self.state_dir / "agent_state.json"
        self.sqlite_file = self.state_dir / "agent_runs.db"

        self._ensure_storage_initialized()

    def _ensure_storage_initialized(self) -> None:
        """Initializes the state directory and SQLite database tables."""
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite run telemetry table
        try:
            with sqlite3.connect(self.sqlite_file) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        milestone_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        status TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        commit_hash TEXT,
                        tests_passed INTEGER,
                        tests_failed INTEGER,
                        quality_score REAL,
                        execution_time_seconds REAL,
                        error_message TEXT
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize SQLite telemetry store: {e}")

    def load_state(self) -> AgentState:
        """Loads state from JSON file or initializes a default state."""
        if not self.state_file.exists():
            default_state = AgentState()
            self.save_state(default_state)
            return default_state

        try:
            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)
            return AgentState.model_validate(data)
        except Exception as e:
            logger.error(
                f"Corrupt or unreadable state file at '{self.state_file}': {e}. Creating backup."
            )
            backup_file = (
                self.state_dir / f"agent_state.corrupt.{int(datetime.now().timestamp())}.bak"
            )
            import contextlib

            with contextlib.suppress(OSError):
                self.state_file.rename(backup_file)
            new_state = AgentState()
            self.save_state(new_state)
            return new_state

    def save_state(self, state: AgentState) -> None:
        """Atomically saves the state to disk using a temporary file and replace operation."""
        state.last_run_timestamp = datetime.now(UTC).isoformat()
        temp_file = self.state_dir / "agent_state.json.tmp"

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state.model_dump(mode="json"), f, indent=2)

            # Atomic rename / replace
            os.replace(temp_file, self.state_file)
            logger.debug(f"Atomically saved state to '{self.state_file}'.")
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise StateManagerError(f"Failed to atomically save agent state: {e}") from e

    def record_task_run(self, record: TaskExecutionRecord) -> None:
        """Records a task run into both the atomic state JSON and SQLite database."""
        state = self.load_state()

        # Update JSON state
        state.task_history.append(record)

        if record.status == TaskStatus.SUCCESS:
            state.metrics.total_tasks_completed += 1
            if record.commit_hash:
                state.metrics.total_commits_created += 1
            state.metrics.total_tests_written += record.tests_passed

            # Update milestone task list
            if record.milestone_id not in state.milestones:
                state.milestones[record.milestone_id] = MilestoneProgress(
                    milestone_id=record.milestone_id,
                    project_id=record.project_id,
                    status=MilestoneStatus.IN_PROGRESS,
                    started_at=record.timestamp,
                )
            milestone = state.milestones[record.milestone_id]
            if record.task_id not in milestone.task_ids:
                milestone.task_ids.append(record.task_id)

        elif record.status == TaskStatus.FAILED:
            if record.task_id not in state.blocked_tasks:
                state.blocked_tasks.append(record.task_id)

        self.save_state(state)

        # Log into SQLite
        try:
            with sqlite3.connect(self.sqlite_file) as conn:
                conn.execute(
                    """
                    INSERT INTO task_runs (
                        task_id, project_id, milestone_id, title, status, timestamp,
                        commit_hash, tests_passed, tests_failed, quality_score,
                        execution_time_seconds, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.task_id,
                        record.project_id,
                        record.milestone_id,
                        record.title,
                        record.status.value,
                        record.timestamp,
                        record.commit_hash,
                        record.tests_passed,
                        record.tests_failed,
                        record.quality_score,
                        record.execution_time_seconds,
                        record.error_message,
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"Failed to append task run to SQLite store: {e}")

    def is_task_completed(self, task_id: str) -> bool:
        """Checks if a task ID has already been completed successfully."""
        state = self.load_state()
        return any(
            t.task_id == task_id and t.status == TaskStatus.SUCCESS for t in state.task_history
        )

    def record_successful_task(
        self,
        task_id: str,
        track_id: str,
        project_id: str,
        milestone_id: str,
        commit_hash: str,
        tests_written: int = 0,
        quality_score: float = 0.95,
        execution_time_seconds: float = 0.0,
    ) -> None:
        """Convenience method to record a successful task run."""
        record = TaskExecutionRecord(
            task_id=task_id,
            project_id=project_id,
            milestone_id=milestone_id,
            title=task_id,
            status=TaskStatus.SUCCESS,
            commit_hash=commit_hash,
            tests_passed=tests_written,
            tests_failed=0,
            quality_score=quality_score,
            execution_time_seconds=execution_time_seconds,
        )
        self.record_task_run(record)
