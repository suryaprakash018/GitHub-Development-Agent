"""Unit tests for safe GitClient, selective staging, branch verification, and SemanticCommitEngine."""

import subprocess
from unittest.mock import MagicMock

import pytest

from agent.core.config import AppConfig
from agent.core.exceptions import DecoupledTargetViolationError, GitSafetyError, SecurityError
from agent.git.client import GitClient
from agent.git.commit_engine import SemanticCommitEngine, SemanticCommitMessage
from agent.llm.base import LLMProvider
from agent.selection.models import TaskCandidate, TaskType
from agent.synthesis.models import GeneratedFile, ImplementationPlan
from agent.utils.security import get_agent_root_path
from agent.validation.linter import LintRunResult
from agent.validation.test_runner import TestRunResult


def _init_local_git_repo(path):
    """Helper to initialize a real local git repo for isolated tests."""
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test Agent"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "agent@test.local"], cwd=path, check=True)
    # Ensure default branch is main
    subprocess.run(["git", "checkout", "-B", "main"], cwd=path, check=True, capture_output=True)

    init_file = path / "README.md"
    init_file.write_text("# Test Repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "chore: initial repository setup"], cwd=path, check=True)


def test_git_client_rejects_agent_root(monkeypatch):
    """GitClient must refuse to operate on the agent's own repository."""
    agent_root = get_agent_root_path()
    with pytest.raises(
        DecoupledTargetViolationError, match="cannot be the GitHub-Development-Agent"
    ):
        GitClient(target_path=agent_root)


def test_git_client_status_clean_repo(tmp_path):
    """GitClient accurately reports a clean repository state."""
    repo_dir = tmp_path / "Target-Repo"
    repo_dir.mkdir()
    _init_local_git_repo(repo_dir)

    client = GitClient(target_path=repo_dir)
    status = client.get_status()

    assert status.is_clean is True
    assert status.current_branch == "main"
    assert len(status.staged_files) == 0
    assert len(status.unstaged_files) == 0
    assert len(status.untracked_files) == 0


def test_git_client_status_detects_untracked_and_staged(tmp_path):
    """GitClient distinguishes untracked and staged files."""
    repo_dir = tmp_path / "Target-Repo"
    repo_dir.mkdir()
    _init_local_git_repo(repo_dir)

    # Create untracked file
    untracked_file = repo_dir / "untracked.py"
    untracked_file.write_text("x = 1\n", encoding="utf-8")

    # Create staged file
    staged_file = repo_dir / "staged.py"
    staged_file.write_text("y = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "staged.py"], cwd=repo_dir, check=True)

    client = GitClient(target_path=repo_dir)
    status = client.get_status()

    assert status.is_clean is False
    assert "staged.py" in status.staged_files
    assert "untracked.py" in status.untracked_files


def test_git_client_assert_clean_worktree_fails_on_unexpected_changes(tmp_path):
    """GitClient aborts if repository has uncommitted, unexpected modifications."""
    repo_dir = tmp_path / "Target-Repo"
    repo_dir.mkdir()
    _init_local_git_repo(repo_dir)

    dirty_file = repo_dir / "manual_user_change.py"
    dirty_file.write_text("# manual user work", encoding="utf-8")

    client = GitClient(target_path=repo_dir)
    with pytest.raises(GitSafetyError, match="Safety Guard Activated.*unexpected/unrelated"):
        client.assert_clean_worktree()


def test_git_client_assert_clean_worktree_passes_with_allowed_files(tmp_path):
    """GitClient permits dirty status if only explicitly allowed task files are dirty."""
    repo_dir = tmp_path / "Target-Repo"
    repo_dir.mkdir()
    _init_local_git_repo(repo_dir)

    task_file = repo_dir / "src" / "feature.py"
    task_file.parent.mkdir(parents=True)
    task_file.write_text("# generated code", encoding="utf-8")

    client = GitClient(target_path=repo_dir)
    # Should not raise because src/feature.py is allowed
    client.assert_clean_worktree(allowed_files=["src/feature.py"])


def test_git_client_verify_active_branch_success_and_failure(tmp_path):
    """GitClient confirms branch name and blocks commits if on incorrect branch."""
    repo_dir = tmp_path / "Target-Repo"
    repo_dir.mkdir()
    _init_local_git_repo(repo_dir)

    client = GitClient(target_path=repo_dir)
    assert client.verify_active_branch("main") == "main"

    # Checkout a feature branch
    subprocess.run(["git", "checkout", "-b", "experimental"], cwd=repo_dir, check=True)

    with pytest.raises(GitSafetyError, match="Branch mismatch: active branch is 'experimental'"):
        client.verify_active_branch("main")


def test_git_client_selective_stage_stages_only_specified_files(tmp_path):
    """GitClient stages ONLY requested files and never performs blanket staging."""
    repo_dir = tmp_path / "Target-Repo"
    repo_dir.mkdir()
    _init_local_git_repo(repo_dir)

    file_a = repo_dir / "file_a.py"
    file_b = repo_dir / "file_b.py"
    file_a.write_text("a = 10\n", encoding="utf-8")
    file_b.write_text("b = 20\n", encoding="utf-8")

    client = GitClient(target_path=repo_dir)
    staged = client.selective_stage(["file_a.py"])

    assert staged == ["file_a.py"]
    status = client.get_status()
    assert "file_a.py" in status.staged_files
    assert "file_b.py" in status.untracked_files  # file_b MUST NOT be staged!


def test_git_client_selective_stage_traversal_attack_blocked(tmp_path):
    """GitClient blocks staging files outside target repository root."""
    repo_dir = tmp_path / "Target-Repo"
    repo_dir.mkdir()
    _init_local_git_repo(repo_dir)

    client = GitClient(target_path=repo_dir)
    with pytest.raises(SecurityError, match=r"(?i)path traversal violation"):
        client.selective_stage(["../unrelated.py"])


def test_git_client_dry_run_prevents_commit(tmp_path):
    """When DRY_RUN=True, create_commit must NOT create a git commit."""
    repo_dir = tmp_path / "Target-Repo"
    repo_dir.mkdir()
    _init_local_git_repo(repo_dir)

    config = AppConfig()
    config.repository.target_path = str(repo_dir)
    config.operational_mode.dry_run = True
    config.operational_mode.auto_commit = False

    client = GitClient(target_path=repo_dir, config=config)

    # Stage a file
    test_f = repo_dir / "sample.py"
    test_f.write_text("print('test')\n", encoding="utf-8")
    client.selective_stage(["sample.py"])

    initial_head = client.get_head_commit_hash()
    result = client.create_commit("feat(test): sample feature")

    assert result.committed is False
    assert result.dry_run is True
    assert result.commit_hash is None
    # Verify git HEAD has not changed
    assert client.get_head_commit_hash() == initial_head


def test_git_client_auto_commit_false_prevents_commit_even_if_dry_run_false(tmp_path):
    """Even if DRY_RUN=False, AUTO_COMMIT=False must strictly prevent any commit."""
    repo_dir = tmp_path / "Target-Repo"
    repo_dir.mkdir()
    _init_local_git_repo(repo_dir)

    config = AppConfig()
    config.repository.target_path = str(repo_dir)
    config.operational_mode.dry_run = False
    config.operational_mode.auto_commit = False

    client = GitClient(target_path=repo_dir, config=config)

    test_f = repo_dir / "sample2.py"
    test_f.write_text("print('test2')\n", encoding="utf-8")
    client.selective_stage(["sample2.py"])

    initial_head = client.get_head_commit_hash()
    result = client.create_commit("feat(test): sample feature 2")

    assert result.committed is False
    assert result.dry_run is True
    assert client.get_head_commit_hash() == initial_head


def test_git_client_dry_run_prevents_push(tmp_path):
    """When DRY_RUN=True or AUTO_PUSH=False, push must NOT execute git push."""
    repo_dir = tmp_path / "Target-Repo"
    repo_dir.mkdir()
    _init_local_git_repo(repo_dir)

    config = AppConfig()
    config.repository.target_path = str(repo_dir)
    config.operational_mode.dry_run = True
    config.operational_mode.auto_push = False

    client = GitClient(target_path=repo_dir, config=config)
    res = client.push()

    assert res.pushed is False
    assert res.dry_run is True


def test_semantic_commit_engine_deterministic_generation():
    """SemanticCommitEngine generates Conventional Commits with AST details without generic filler."""
    engine = SemanticCommitEngine(llm_provider=None)

    candidate = TaskCandidate(
        task_id="task_ds_01_01",
        track_id="track_data_science",
        project_id="proj_data_cleaning",
        milestone_id="ms_ds_01_01",
        title="Implement Vectorized Outlier Detection & Robust Scaling",
        description="IQR and Modified Z-Score outlier detection with NaN handling",
        task_type=TaskType.NEW_MODULE,
        target_files=["src/data_science/data_cleaning/ds_01_01.py"],
        test_files=["tests/data_science/data_cleaning/test_ds_01_01.py"],
        rationale="Vectorized statistical outlier detection kernel for portfolio quality.",
    )

    module_code = (
        "class IQRFilter:\n"
        "    def fit_detect(self, values):\n"
        "        pass\n\n"
        "class ModifiedZScoreFilter:\n"
        "    def fit_detect(self, values):\n"
        "        pass\n"
    )

    test_code = (
        "def test_iqr_outlier_detection_bounds():\n"
        "    pass\n\n"
        "def test_zscore_with_nan_handling():\n"
        "    pass\n"
    )

    plan = ImplementationPlan(
        task_id="task_ds_01_01",
        summary="Vectorized outlier filters",
        files=[
            GeneratedFile(
                relative_path="src/data_science/data_cleaning/ds_01_01.py",
                content=module_code,
                description="Core outlier filter classes",
            ),
            GeneratedFile(
                relative_path="tests/data_science/data_cleaning/test_ds_01_01.py",
                content=test_code,
                description="Unit tests for outlier bounds",
            ),
        ],
    )

    test_res = TestRunResult(
        success=True,
        exit_code=0,
        total_tests=2,
        passed_count=2,
        failed_count=0,
        skipped_count=0,
        duration_seconds=0.45,
    )
    lint_res = LintRunResult(success=True, exit_code=0, output="All checks passed!")

    msg = engine.generate_commit_message(
        candidate=candidate,
        plan=plan,
        diff="mock diff content",
        test_result=test_res,
        lint_result=lint_res,
        quality_score=0.95,
    )

    assert msg.subject.startswith("feat(data_science/data_cleaning): implement")
    assert "IQRFilter" in msg.formatted_message
    assert "ModifiedZScoreFilter" in msg.formatted_message
    assert "test_iqr_outlier_detection_bounds" in msg.formatted_message
    assert "Milestone-ID: ms_ds_01_01" in msg.footer
    assert "Quality-Score: 0.95/1.00" in msg.footer


def test_semantic_commit_engine_llm_structured_generation():
    """SemanticCommitEngine delegates to LLM when provided."""
    mock_llm = MagicMock(spec=LLMProvider)
    expected_msg = SemanticCommitMessage(
        subject="feat(ml/supervised): implement gradient boosted decision stumps",
        body="Architectural addition of boosting kernel.",
        key_changes=["- Added DecisionStump class"],
        validation_summary="Pytest: 4 passed",
        footer="Milestone-ID: ms_ml_01",
        formatted_message="feat(ml/supervised): implement gradient boosted decision stumps\n\nBody",
    )
    mock_llm.generate_structured.return_value = expected_msg

    engine = SemanticCommitEngine(llm_provider=mock_llm)

    candidate = TaskCandidate(
        task_id="task_ml_01",
        track_id="track_ml",
        project_id="proj_supervised",
        milestone_id="ms_ml_01",
        title="Implement Decision Stump",
        description="Tree model",
        task_type=TaskType.NEW_MODULE,
        target_files=["src/ml/stump.py"],
        rationale="Boosted model",
    )
    plan = ImplementationPlan(task_id="task_ml_01", summary="Plan", files=[])
    test_res = TestRunResult(success=True, exit_code=0, total_tests=1, passed_count=1)
    lint_res = LintRunResult(success=True, exit_code=0, output="")

    msg = engine.generate_commit_message(
        candidate=candidate,
        plan=plan,
        diff="",
        test_result=test_res,
        lint_result=lint_res,
        quality_score=1.0,
    )

    assert msg.subject == expected_msg.subject
    assert msg.formatted_message == expected_msg.formatted_message
