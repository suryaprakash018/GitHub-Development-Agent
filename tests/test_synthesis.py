"""Unit tests for synthesis generation, remediation, and dry-run execution harness."""

import subprocess
from unittest.mock import MagicMock

from agent.core.config import AppConfig
from agent.git.analyzer import TargetRepoAnalysis
from agent.git.commit_engine import SemanticCommitMessage
from agent.llm.base import LLMProvider
from agent.selection.models import TaskCandidate, TaskType
from agent.synthesis.executor import SynthesisExecutionHarness
from agent.synthesis.generator import TaskImplementationGenerator
from agent.synthesis.models import GeneratedFile, ImplementationPlan, RemediationPlan
from agent.synthesis.remediator import FailureRemediator


def test_generator_creates_plan():
    mock_llm = MagicMock(spec=LLMProvider)
    mock_plan = ImplementationPlan(
        task_id="task_01",
        summary="Test summary",
        files=[GeneratedFile(relative_path="src/a.py", content="x = 1\n", description="A")],
    )
    mock_llm.generate_structured.return_value = mock_plan

    generator = TaskImplementationGenerator(mock_llm)
    candidate = TaskCandidate(
        task_id="task_01",
        track_id="track_1",
        project_id="proj_1",
        milestone_id="ms_1",
        title="Title",
        description="Desc",
        task_type=TaskType.NEW_MODULE,
        target_files=["src/a.py"],
        rationale="Rationale",
    )

    plan = generator.generate_plan(candidate, TargetRepoAnalysis(target_path="mock"))
    assert plan.task_id == "task_01"
    assert len(plan.files) == 1
    assert plan.files[0].relative_path == "src/a.py"


def test_remediator_replaces_fixed_files():
    mock_llm = MagicMock(spec=LLMProvider)
    remediation_plan = RemediationPlan(
        attempt_number=1,
        diagnosis="Fixed variable name typo",
        fixed_files=[
            GeneratedFile(relative_path="src/a.py", content="x = 42\n", description="Fixed A")
        ],
    )
    mock_llm.generate_structured.return_value = remediation_plan

    remediator = FailureRemediator(mock_llm)
    candidate = TaskCandidate(
        task_id="task_01",
        track_id="track_1",
        project_id="proj_1",
        milestone_id="ms_1",
        title="Title",
        description="Desc",
        task_type=TaskType.NEW_MODULE,
        target_files=["src/a.py"],
        rationale="Rationale",
    )

    current_plan = ImplementationPlan(
        task_id="task_01",
        summary="Initial",
        files=[GeneratedFile(relative_path="src/a.py", content="x = 1\n", description="A")],
    )

    updated_plan, diagnosis = remediator.remediate(
        candidate, current_plan, "Error traceback", attempt_number=1
    )

    assert diagnosis == "Fixed variable name typo"
    assert updated_plan.files[0].content == "x = 42\n"


def test_synthesis_execution_harness_dry_run_rollback(monkeypatch, tmp_path):
    external_repo = tmp_path / "External-AI-Portfolio"
    external_repo.mkdir()

    # Initialize git repo in external_repo
    subprocess.run(["git", "init"], cwd=external_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=external_repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=external_repo, check=True)

    init_file = external_repo / "init.txt"
    init_file.write_text("initial", encoding="utf-8")
    conftest = external_repo / "conftest.py"
    conftest.write_text("", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=external_repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=external_repo, check=True)

    config = AppConfig()
    config.repository.target_path = str(external_repo)
    config.operational_mode.dry_run = True

    # Mock LLM that returns a valid passing implementation plan
    mock_llm = MagicMock(spec=LLMProvider)
    mock_plan = ImplementationPlan(
        task_id="task_ds_01",
        summary="Valid module and passing test",
        files=[
            GeneratedFile(
                relative_path="src/module.py",
                content=(
                    "class Calculator:\n"
                    "    '''Production arithmetic kernel with validation.'''\n"
                    "    def add(self, a: float, b: float) -> float:\n"
                    "        '''Returns sum of numbers.'''\n"
                    "        return a + b\n\n"
                    "    def subtract(self, a: float, b: float) -> float:\n"
                    "        '''Returns difference of numbers.'''\n"
                    "        return a - b\n\n"
                    "    def multiply(self, a: float, b: float) -> float:\n"
                    "        '''Returns product of numbers.'''\n"
                    "        return a * b\n\n"
                    "    def divide(self, a: float, b: float) -> float:\n"
                    "        '''Returns quotient with zero division check.'''\n"
                    "        if b == 0:\n"
                    "            raise ValueError('Division by zero is not allowed.')\n"
                    "        return a / b\n"
                ),
                description="Calculator module",
            ),
            GeneratedFile(
                relative_path="tests/test_module.py",
                content=(
                    "import pytest\n"
                    "from src.module import Calculator\n\n"
                    "def test_calculator_operations():\n"
                    "    c = Calculator()\n"
                    "    assert c.add(10, 20) == 30\n"
                    "    assert c.subtract(20, 5) == 15\n"
                    "    assert c.multiply(3, 4) == 12\n"
                    "    assert c.divide(10, 2) == 5.0\n\n"
                    "def test_calculator_divide_by_zero():\n"
                    "    c = Calculator()\n"
                    "    with pytest.raises(ValueError, match='Division by zero'):\n"
                    "        c.divide(10, 0)\n"
                ),
                description="Calculator tests",
            ),
        ],
    )
    mock_remediation = RemediationPlan(
        attempt_number=1,
        diagnosis="Self-remediated test imports",
        fixed_files=mock_plan.files,
    )
    mock_commit = SemanticCommitMessage(
        subject="feat(data_science/proj_ds_01): implement calculator",
        body="Calculator arithmetic kernel",
        key_changes=["- Added Calculator class"],
        validation_summary="Pytest: 2 passed",
        footer="Milestone-ID: ms_01",
        formatted_message="feat(data_science/proj_ds_01): implement calculator\n\nCalculator arithmetic kernel",
    )

    def _mock_structured(prompt, schema=None, **kwargs):
        target_schema = schema or kwargs.get("schema")
        if target_schema == RemediationPlan:
            return mock_remediation
        elif target_schema == SemanticCommitMessage:
            return mock_commit
        return mock_plan

    mock_llm.generate_structured.side_effect = _mock_structured

    harness = SynthesisExecutionHarness(config, mock_llm)

    candidate = TaskCandidate(
        task_id="task_ds_01",
        track_id="track_data_science",
        project_id="proj_ds_01",
        milestone_id="ms_01",
        title="Implement Calculator",
        description="Add calculator",
        task_type=TaskType.NEW_MODULE,
        target_files=["src/module.py"],
        test_files=["tests/test_module.py"],
        rationale="Clean math logic",
    )

    report = harness.execute_dry_run(candidate)

    assert report.decision == "APPROVED_FOR_COMMIT"
    assert report.test_result.success is True
    assert report.test_result.passed_count >= 1
    assert report.quality_gate_result.passed is True

    # CRITICAL DRY-RUN GUARANTEE: Check that created files were cleanly rolled back!
    assert not (external_repo / "src" / "module.py").exists()
    assert not (external_repo / "tests" / "test_module.py").exists()
    assert (external_repo / "init.txt").exists()
