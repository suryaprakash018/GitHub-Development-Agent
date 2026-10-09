"""Data models for candidate task selection, classification, and scoring."""

from enum import StrEnum

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    FOUNDATION_SETUP = "FOUNDATION_SETUP"
    NEW_MODULE = "NEW_MODULE"
    TEST_SUITE = "TEST_SUITE"
    PROJECT_HARDENING = "PROJECT_HARDENING"
    BENCHMARK = "BENCHMARK"
    DOCUMENTATION = "DOCUMENTATION"


class TaskCandidate(BaseModel):
    """Specification of an actionable development task selected for execution."""

    task_id: str
    track_id: str
    project_id: str
    milestone_id: str
    title: str
    description: str
    task_type: TaskType
    target_files: list[str] = Field(default_factory=list)
    test_files: list[str] = Field(default_factory=list)
    rationale: str
    portfolio_value_score: float = 0.9
    prerequisites_satisfied: bool = True
