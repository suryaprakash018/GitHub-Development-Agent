"""Data models for code synthesis, implementation plans, and auto-remediation."""

from pydantic import BaseModel, Field


class GeneratedFile(BaseModel):
    """A single generated or modified file specification."""

    relative_path: str = Field(
        description="Relative file path within the target repository, e.g., 'src/data_science/outliers.py'"
    )
    content: str = Field(
        description="Complete, syntactically valid source code or test content (no placeholders, no ellipses)"
    )
    description: str = Field(
        description="Explanation of what was implemented or modified in this file"
    )


class ImplementationPlan(BaseModel):
    """Comprehensive implementation plan containing code modules and test suites."""

    task_id: str = Field(description="The task ID being implemented")
    summary: str = Field(description="High-level engineering summary of the changes")
    files: list[GeneratedFile] = Field(
        description="List of files to create or modify, including functional code and tests"
    )
    expected_test_count: int = Field(
        default=3, description="Estimated number of unit tests implemented in the test files"
    )


class RemediationPlan(BaseModel):
    """Scoped remediation plan produced in response to test or linting diagnostics."""

    attempt_number: int
    diagnosis: str = Field(description="Diagnosis of why the tests or linter failed")
    fixed_files: list[GeneratedFile] = Field(
        description="Updated file contents containing targeted fixes for the reported errors"
    )
