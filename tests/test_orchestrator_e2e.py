"""End-to-End integration tests for AutonomousDevelopmentOrchestrator and WindowsSchedulerManager."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from agent.core.config import AppConfig
from agent.core.context import ExecutionLock
from agent.git.commit_engine import SemanticCommitMessage
from agent.llm.base import LLMProvider
from agent.orchestrator.pipeline import AutonomousDevelopmentOrchestrator
from agent.scheduler.windows import WindowsSchedulerManager
from agent.state.manager import StateManager
from agent.state.models import (
    MilestoneProgress,
    MilestoneStatus,
    TaskExecutionRecord,
    TaskStatus,
)
from agent.synthesis.models import GeneratedFile, ImplementationPlan, RemediationPlan


def _init_test_git_repo(path):
    """Helper to initialize a clean git repository."""
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Portfolio Author"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "author@portfolio.local"], cwd=path, check=True)
    subprocess.run(["git", "checkout", "-B", "main"], cwd=path, check=True, capture_output=True)

    init_file = path / "README.md"
    init_file.write_text("# Portfolio Target\n", encoding="utf-8")
    pyproject = path / "pyproject.toml"
    pyproject_content = (
        "[project]\nname = 'portfolio'\nversion = '0.1.0'\n\n"
        "[tool.pytest.ini_options]\npythonpath = ['.']\n"
    )
    pyproject.write_text(pyproject_content, encoding="utf-8")
    conftest = path / "conftest.py"
    conftest.write_text("", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "chore: bootstrap portfolio"], cwd=path, check=True)


def test_orchestrator_dry_run_full_cycle(tmp_path):
    """Verifies complete autonomous pipeline with 100% clean rollback in DRY_RUN mode."""
    target_repo = tmp_path / "External-AI-Portfolio"
    target_repo.mkdir()
    _init_test_git_repo(target_repo)

    config = AppConfig()
    config.repository.target_path = str(target_repo)
    config.operational_mode.dry_run = True
    config.operational_mode.auto_commit = False
    config.operational_mode.auto_push = False

    # Capture initial commit hash
    init_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=target_repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    # Mock LLM providing valid code, test, and semantic commit message
    mock_llm = MagicMock(spec=LLMProvider)
    mock_plan = ImplementationPlan(
        task_id="task_ms_ds_01_01",
        summary="Vectorized outlier detector",
        files=[
            GeneratedFile(
                relative_path="src/data_science/ds_01_feature_eng/ds_01_01.py",
                content=(
                    "class OutlierDetector:\n"
                    "    '''Vectorized detector with bounds checking.'''\n"
                    "    def detect_iqr(self, values: list[float]) -> list[bool]:\n"
                    "        '''Detects numerical outliers via IQR.'''\n"
                    "        if not values:\n"
                    "            return []\n"
                    "        return [x > 100.0 for x in values]\n"
                ),
                description="Core detector module",
            ),
            GeneratedFile(
                relative_path="tests/data_science/ds_01_feature_eng/test_ds_01_01.py",
                content=(
                    "from src.data_science.ds_01_feature_eng.ds_01_01 import OutlierDetector\n\n"
                    "def test_detector_valid():\n"
                    "    d = OutlierDetector()\n"
                    "    res = d.detect_iqr([1.0, 2.0, 150.0])\n"
                    "    assert res == [False, False, True]\n\n"
                    "def test_detector_empty():\n"
                    "    d = OutlierDetector()\n"
                    "    assert d.detect_iqr([]) == []\n"
                ),
                description="Detector tests",
            ),
        ],
    )

    mock_commit = SemanticCommitMessage(
        subject="feat(data_science/ds_01_feature_eng): implement outlier detection",
        body="Vectorized IQR outlier detector for clean preprocessing.",
        key_changes=["- OutlierDetector class introduced"],
        validation_summary="Pytest: 2 passed, Ruff clean",
        footer="Milestone-ID: ms_ds_01_01",
        formatted_message="feat(data_science/ds_01_feature_eng): implement outlier detection\n\nBody",
    )

    mock_remediation = RemediationPlan(
        attempt_number=1,
        diagnosis="Fixed imports",
        fixed_files=mock_plan.files,
    )

    def _mock_structured(prompt, schema=None, **kwargs):
        target_schema = schema or kwargs.get("schema")
        if target_schema == RemediationPlan:
            return mock_remediation
        elif target_schema == SemanticCommitMessage:
            return mock_commit
        return mock_plan

    mock_llm.generate_structured.side_effect = _mock_structured

    orchestrator = AutonomousDevelopmentOrchestrator(config=config, llm_provider=mock_llm)
    result = orchestrator.run(force_dry_run=True)

    # Assert run outcome
    assert result.status == "DRY_RUN_APPROVED"
    assert result.quality_gate_passed is True
    assert result.committed is False
    assert result.pushed is False
    assert "src/" in result.diff_preview
    assert result.candidate is not None
    assert result.candidate.task_id == "task_ms_ds_01_01"

    # Assert DRY_RUN ROLLBACK GUARANTEE: Target repo is left pristine
    assert not (target_repo / "src" / "data_science" / "ds_01_feature_eng" / "ds_01_01.py").exists()
    assert not (
        target_repo / "tests" / "data_science" / "ds_01_feature_eng" / "test_ds_01_01.py"
    ).exists()

    # Assert no commit was created
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=target_repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert current_head == init_head

    # Assert git worktree is clean
    status_out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=target_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert status_out == ""

    # Assert telemetry audit log was recorded
    assert result.telemetry_file is not None
    assert Path(result.telemetry_file).exists()


def test_orchestrator_rejects_dirty_target_repository(tmp_path):
    """Orchestrator halts immediately if user has uncommitted manual work in target repo."""
    target_repo = tmp_path / "External-AI-Portfolio"
    target_repo.mkdir()
    _init_test_git_repo(target_repo)

    # Create uncommitted user work
    dirty_file = target_repo / "user_draft.py"
    dirty_file.write_text("# manual draft work", encoding="utf-8")

    config = AppConfig()
    config.repository.target_path = str(target_repo)
    config.operational_mode.dry_run = True

    orchestrator = AutonomousDevelopmentOrchestrator(config=config)
    result = orchestrator.run()

    assert result.status == "GIT_SAFETY_ERROR"
    assert "Safety Guard Activated" in result.explanation
    assert dirty_file.exists()  # Manual user work preserved!


def test_orchestrator_concurrency_lock_prevents_simultaneous_runs(tmp_path):
    """When a run is active, another instance cannot operate concurrently."""
    config = AppConfig()
    config.repository.target_path = str(tmp_path)

    orchestrator = AutonomousDevelopmentOrchestrator(config=config)

    # Hold the lock manually
    with ExecutionLock():
        # Second instance tries to run while lock is held
        result = orchestrator.run()
        assert result.status == "LOCK_BUSY"
        assert "another agent instance is currently executing" in result.explanation


def test_orchestrator_quality_gate_failure_aborts_without_commit(tmp_path):
    """Quality gate rejection aborts safely and never creates a commit."""
    target_repo = tmp_path / "External-AI-Portfolio"
    target_repo.mkdir()
    _init_test_git_repo(target_repo)

    config = AppConfig()
    config.repository.target_path = str(target_repo)
    config.operational_mode.dry_run = True

    # Mock LLM generating failing code (syntax error or zero tests)
    mock_llm = MagicMock(spec=LLMProvider)
    failing_plan = ImplementationPlan(
        task_id="task_ds_01_01",
        summary="Empty plan with no tests",
        files=[
            GeneratedFile(
                relative_path="src/module.py",
                content="x = 1\n",
                description="Trivial file violating quality gate LOC requirement",
            )
        ],
    )
    failing_remediation = RemediationPlan(
        attempt_number=1,
        diagnosis="Still invalid",
        fixed_files=failing_plan.files,
    )

    def _mock_failing(prompt, schema=None, **kwargs):
        target_schema = schema or kwargs.get("schema")
        if target_schema == RemediationPlan:
            return failing_remediation
        return failing_plan

    mock_llm.generate_structured.side_effect = _mock_failing

    orchestrator = AutonomousDevelopmentOrchestrator(config=config, llm_provider=mock_llm)
    result = orchestrator.run()

    assert result.status == "QUALITY_GATE_REJECTED"
    assert result.committed is False
    assert result.pushed is False
    assert result.quality_gate_passed is False


def test_orchestrator_no_task_available_when_all_completed(tmp_path):
    """Orchestrator exits cleanly when all roadmap tasks are finished."""
    target_repo = tmp_path / "External-AI-Portfolio"
    target_repo.mkdir()
    _init_test_git_repo(target_repo)

    config = AppConfig()
    config.repository.target_path = str(target_repo)

    state_mgr = StateManager(state_dir=tmp_path)
    state = state_mgr.load_state()

    # Mark all 16 roadmap milestones as completed
    from agent.roadmap.engine import RoadmapEngine

    roadmap = RoadmapEngine()
    for tr in roadmap.dag.tracks:
        for pr in tr.projects:
            for ms in pr.milestones:
                state.milestones[ms.id] = MilestoneProgress(
                    milestone_id=ms.id,
                    project_id=pr.id,
                    status=MilestoneStatus.COMPLETED,
                )
                state.task_history.append(
                    TaskExecutionRecord(
                        task_id=f"task_{ms.id}",
                        project_id=pr.id,
                        milestone_id=ms.id,
                        title=ms.title,
                        status=TaskStatus.SUCCESS,
                        commit_hash="hash",
                    )
                )
    state_mgr.save_state(state)

    orchestrator = AutonomousDevelopmentOrchestrator(
        config=config,
        state_manager=state_mgr,
        roadmap_engine=roadmap,
    )
    result = orchestrator.run()

    assert result.status == "NO_TASK_AVAILABLE"
    assert result.committed is False


def test_windows_scheduler_command_builder():
    """WindowsSchedulerManager builds correct, configurable schtasks command."""
    config = AppConfig()
    config.scheduling.preferred_time = "21:45"

    mgr = WindowsSchedulerManager(config=config, task_name="Test-Dev-Agent")
    cmd = mgr.build_registration_command(preferred_time="23:15")

    assert "schtasks" in cmd
    assert "/create" in cmd
    assert "Test-Dev-Agent" in cmd
    assert "/sc" in cmd
    assert "DAILY" in cmd
    assert "/st" in cmd
    assert "23:15" in cmd
    assert "--scheduled" in " ".join(cmd)
    # Ensure it does NOT run the task upon registration
    assert "/run" not in cmd
