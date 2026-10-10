"""Unit tests for the CLI subcommands (verify, status, plan, run)."""

import argparse
import subprocess
from unittest.mock import MagicMock

from agent.cli import cmd_plan, cmd_run, cmd_status, cmd_verify
from agent.git.commit_engine import SemanticCommitMessage
from agent.llm.base import LLMProvider
from agent.synthesis.models import GeneratedFile, ImplementationPlan, RemediationPlan


def test_cli_status_returns_zero(capsys):
    args = argparse.Namespace(command="status")
    exit_code = cmd_status(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "AUTONOMOUS DEVELOPMENT AGENT STATUS" in captured.out
    assert "Roadmap Progress:" in captured.out


def test_cli_verify_with_clean_environment(capsys, monkeypatch, tmp_path):
    external_repo = tmp_path / "External-AI-Portfolio"
    external_repo.mkdir()
    (external_repo / ".git").mkdir()

    monkeypatch.setenv("TARGET_REPOSITORY_PATH", str(external_repo))
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_gemini_key_for_test")

    args = argparse.Namespace(command="verify")
    exit_code = cmd_verify(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[PASS] Python Version" in captured.out
    assert "[PASS] Git CLI" in captured.out
    assert "[PASS] Decoupled Target Repository Path" in captured.out


def test_cli_verify_catches_decoupled_target_violation(capsys, monkeypatch):
    from agent.utils.security import get_agent_root_path

    # Set target to agent itself
    monkeypatch.setenv("TARGET_REPOSITORY_PATH", str(get_agent_root_path()))

    args = argparse.Namespace(command="verify")
    exit_code = cmd_verify(args)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "[FAIL] Decoupled Safety Violation" in captured.out


def test_cli_plan_command(capsys, monkeypatch, tmp_path):
    external_repo = tmp_path / "External-AI-Portfolio"
    external_repo.mkdir()

    monkeypatch.setenv("TARGET_REPOSITORY_PATH", str(external_repo))

    args = argparse.Namespace(command="plan")
    exit_code = cmd_plan(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "PROPOSED DEVELOPMENT TASK PLAN" in captured.out
    assert "Task ID:" in captured.out


def test_cli_run_dry_run_mode(monkeypatch, tmp_path):
    external_repo = tmp_path / "External-AI-Portfolio"
    external_repo.mkdir()
    subprocess.run(["git", "init"], cwd=external_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=external_repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.local"], cwd=external_repo, check=True
    )
    subprocess.run(
        ["git", "checkout", "-B", "main"], cwd=external_repo, check=True, capture_output=True
    )
    (external_repo / "README.md").write_text("# Test\n", encoding="utf-8")
    (external_repo / "pyproject.toml").write_text(
        "[project]\nname = 'portfolio'\nversion = '0.1.0'\n\n[tool.pytest.ini_options]\npythonpath = ['.']\n",
        encoding="utf-8",
    )
    (external_repo / "conftest.py").write_text("", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=external_repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=external_repo, check=True)

    monkeypatch.setenv("TARGET_REPOSITORY_PATH", str(external_repo))

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
    monkeypatch.setattr("agent.orchestrator.pipeline.get_llm_provider", lambda config: mock_llm)

    args = argparse.Namespace(
        command="run",
        dry_run=True,
        scheduled=False,
        run_now=True,
    )
    exit_code = cmd_run(args)
    assert exit_code == 0


def test_cli_run_halts_on_uninitialized_git_repo(monkeypatch, tmp_path):
    external_repo = tmp_path / "External-AI-Portfolio-No-Git"
    external_repo.mkdir()  # exists but no .git

    monkeypatch.setenv("TARGET_REPOSITORY_PATH", str(external_repo))

    args = argparse.Namespace(
        command="run",
        dry_run=True,
        scheduled=False,
        run_now=True,
    )
    exit_code = cmd_run(args)
    assert exit_code == 1


def test_cli_verify_removes_all_credential_previews(capsys, monkeypatch, tmp_path):
    external_repo = tmp_path / "External-AI-Portfolio"
    external_repo.mkdir()
    (external_repo / ".git").mkdir()

    groq_secret = "gsk_super_secret_unique_groq_key_9876543210"
    gemini_secret = "AIzaSyDummyGeminiKeySecret1234567890"

    monkeypatch.setenv("TARGET_REPOSITORY_PATH", str(external_repo))
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", groq_secret)
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("GEMINI_API_KEY", gemini_secret)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock()
    monkeypatch.setattr("groq.Groq", lambda *args, **kwargs: mock_client)

    args = argparse.Namespace(command="verify")
    exit_code = cmd_verify(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[PASS] Groq API Key: Configured" in captured.out
    assert groq_secret not in captured.out
    assert "9876543210" not in captured.out
    assert gemini_secret not in captured.out
