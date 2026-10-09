"""State models for persistent task tracking, milestone progression, and metrics."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    DRY_RUN = "DRY_RUN"


class TaskExecutionRecord(BaseModel):
    """Record of a single task execution attempt."""

    task_id: str
    project_id: str
    milestone_id: str
    title: str
    status: TaskStatus
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    commit_hash: str | None = None
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    quality_score: float = 1.0
    execution_time_seconds: float = 0.0
    error_message: str | None = None


class MilestoneStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class MilestoneProgress(BaseModel):
    """Progress tracker for a specific roadmap milestone."""

    milestone_id: str
    project_id: str
    status: MilestoneStatus = MilestoneStatus.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    task_ids: list[str] = Field(default_factory=list)


class CumulativePortfolioMetrics(BaseModel):
    """Aggregated portfolio metrics across the target repository."""

    total_tasks_completed: int = 0
    total_commits_created: int = 0
    total_tests_written: int = 0
    active_track_id: str = "track_data_science"
    active_project_id: str = "proj_ds_01_feature_eng"
    active_milestone_id: str = "ms_ds_01_01"


class AgentState(BaseModel):
    """Master persistent state serialized to state/agent_state.json."""

    schema_version: str = "1.0.0"
    last_run_timestamp: str | None = None
    metrics: CumulativePortfolioMetrics = Field(default_factory=CumulativePortfolioMetrics)
    milestones: dict[str, MilestoneProgress] = Field(default_factory=dict)
    task_history: list[TaskExecutionRecord] = Field(default_factory=list)
    blocked_tasks: list[str] = Field(default_factory=list)
