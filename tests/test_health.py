"""Unit tests for diagnostic health check system."""

import subprocess
from pathlib import Path

from agent.core.config import AppConfig
from agent.validation.health import SystemHealthChecker


def test_check_python_environment() -> None:
    checker = SystemHealthChecker()
    res = checker.check_python_environment()
    assert res.status == "PASS"
    assert "Python" in res.message


def test_check_git_cli() -> None:
    checker = SystemHealthChecker()
    res = checker.check_git_cli()
    assert res.status in {"PASS", "WARN"}
    assert "git version" in res.message.lower()


def test_check_target_repository_clean_external(tmp_path: Path) -> None:
    target = tmp_path / "external_repo"
    target.mkdir()
    subprocess.run(["git", "init"], cwd=target, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.local"], cwd=target, check=True)
    subprocess.run(["git", "checkout", "-B", "main"], cwd=target, check=True, capture_output=True)
    (target / "README.md").write_text("# Test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=target, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=target, check=True, capture_output=True)

    config = AppConfig()
    config.repository.target_path = str(target)
    config.repository.default_branch = "main"

    checker = SystemHealthChecker(config=config)
    res = checker.check_target_repository()
    assert res.status == "PASS"
    assert "Verified clean decoupled Git repository" in res.message


def test_check_target_repository_rejects_agent_root() -> None:
    config = AppConfig()
    # Force target path to be agent root
    checker = SystemHealthChecker(config=config)
    config.repository.target_path = str(checker.agent_root)

    res = checker.check_target_repository()
    assert res.status == "FAIL"
    assert "Decoupled security violation" in res.message


def test_check_git_remote_accessibility_no_remote(tmp_path: Path) -> None:
    target = tmp_path / "repo_no_remote"
    target.mkdir()
    subprocess.run(["git", "init"], cwd=target, check=True, capture_output=True)

    config = AppConfig()
    config.repository.target_path = str(target)
    checker = SystemHealthChecker(config=config)

    res = checker.check_git_remote_accessibility()
    assert res.status == "WARN"
    assert "No Git remote named" in res.message


def test_check_gemini_api_unconfigured() -> None:
    config = AppConfig()
    config.ai.gemini_api_key = ""
    checker = SystemHealthChecker(config=config)

    res = checker.check_gemini_api_connectivity()
    assert res.status == "WARN"
    assert "GEMINI_API_KEY is not configured" in res.message


def test_health_checker_run_all(tmp_path: Path) -> None:
    target = tmp_path / "valid_target"
    target.mkdir()
    subprocess.run(["git", "init"], cwd=target, check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-B", "main"], cwd=target, check=True, capture_output=True)
    (target / "README.md").write_text("# Test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=target, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=target, check=True, capture_output=True)

    config = AppConfig()
    config.repository.target_path = str(target)
    checker = SystemHealthChecker(config=config)

    report = checker.run_all()
    assert report.all_passed is True  # No FAIL statuses
    assert len(report.items) == 5
