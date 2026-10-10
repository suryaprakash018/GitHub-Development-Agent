"""Unit tests for diagnostic health check system."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import httpx
from groq import (
    APIConnectionError,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

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
    config.ai.provider = "gemini"
    config.ai.gemini_api_key = ""
    checker = SystemHealthChecker(config=config)

    report = checker.run_all()
    assert report.all_passed is True  # No FAIL statuses
    assert len(report.items) == 5
    assert report.items[-1].name == "Google Gemini API"


def test_health_checker_run_all_with_groq(tmp_path: Path, monkeypatch) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock()
    monkeypatch.setattr("groq.Groq", lambda *args, **kwargs: mock_client)

    target = tmp_path / "valid_target_groq"
    target.mkdir()
    subprocess.run(["git", "init"], cwd=target, check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-B", "main"], cwd=target, check=True, capture_output=True)
    (target / "README.md").write_text("# Test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=target, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=target, check=True, capture_output=True)

    config = AppConfig()
    config.repository.target_path = str(target)
    config.ai.provider = "groq"
    config.ai.groq_api_key = "gsk_test_key"
    config.ai.groq_model = "qwen/qwen3.8-27b"
    checker = SystemHealthChecker(config=config)

    report = checker.run_all()
    assert report.all_passed is True
    assert len(report.items) == 5
    assert report.items[-1].name == "Groq Cloud API"
    assert report.items[-1].status == "PASS"


def test_check_groq_api_missing_key() -> None:
    config = AppConfig()
    config.ai.groq_api_key = ""
    checker = SystemHealthChecker(config=config)
    res = checker.check_groq_api_connectivity()
    assert res.status == "WARN"
    assert "GROQ_API_KEY is not configured" in res.message


def test_check_groq_api_missing_model() -> None:
    config = AppConfig()
    config.ai.groq_api_key = "gsk_dummy_test_key"
    config.ai.groq_model = ""
    checker = SystemHealthChecker(config=config)
    res = checker.check_groq_api_connectivity()
    assert res.status == "WARN"
    assert "GROQ_MODEL is not configured" in res.message


def test_check_groq_api_success(monkeypatch) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock()
    monkeypatch.setattr("groq.Groq", lambda *args, **kwargs: mock_client)

    config = AppConfig()
    config.ai.groq_api_key = "gsk_test_secret_key_12345"
    config.ai.groq_model = "qwen/qwen3.8-27b"
    checker = SystemHealthChecker(config=config)
    res = checker.check_groq_api_connectivity()

    assert res.status == "PASS"
    assert "qwen/qwen3.8-27b" in res.message
    assert res.details == {"model": "qwen/qwen3.8-27b"}
    assert "gsk_" not in res.message
    mock_client.chat.completions.create.assert_called_once_with(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
    )


def test_check_groq_api_authentication_error(monkeypatch) -> None:
    secret_key = "gsk_super_secret_test_key_abc123"
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(401, request=req)
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = AuthenticationError(
        f"Invalid key {secret_key}", response=resp, body=None
    )
    monkeypatch.setattr("groq.Groq", lambda *args, **kwargs: mock_client)

    config = AppConfig()
    config.ai.groq_api_key = secret_key
    config.ai.groq_model = "qwen/qwen3.8-27b"
    checker = SystemHealthChecker(config=config)
    res = checker.check_groq_api_connectivity()

    assert res.status == "FAIL"
    assert "authentication failed" in res.message
    assert secret_key not in res.message
    assert secret_key not in str(res.details)
    assert "gsk_" not in res.message
    mock_client.chat.completions.create.assert_called_once()


def test_check_groq_api_not_found_error(monkeypatch) -> None:
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(404, request=req)
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = NotFoundError(
        "Model test not found", response=resp, body=None
    )
    monkeypatch.setattr("groq.Groq", lambda *args, **kwargs: mock_client)

    config = AppConfig()
    config.ai.groq_api_key = "gsk_dummy_test_key"
    config.ai.groq_model = "unknown-model"
    checker = SystemHealthChecker(config=config)
    res = checker.check_groq_api_connectivity()

    assert res.status == "FAIL"
    assert "not found or unavailable" in res.message
    mock_client.chat.completions.create.assert_called_once()


def test_check_groq_api_permission_denied_error(monkeypatch) -> None:
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(403, request=req)
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = PermissionDeniedError(
        "Access denied", response=resp, body=None
    )
    monkeypatch.setattr("groq.Groq", lambda *args, **kwargs: mock_client)

    config = AppConfig()
    config.ai.groq_api_key = "gsk_dummy_test_key"
    config.ai.groq_model = "restricted-model"
    checker = SystemHealthChecker(config=config)
    res = checker.check_groq_api_connectivity()

    assert res.status == "FAIL"
    assert "permission denied" in res.message
    mock_client.chat.completions.create.assert_called_once()


def test_check_groq_api_rate_limit_error(monkeypatch) -> None:
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RateLimitError(
        "Rate limit reached on requests per day", response=resp, body=None
    )
    monkeypatch.setattr("groq.Groq", lambda *args, **kwargs: mock_client)

    config = AppConfig()
    config.ai.groq_api_key = "gsk_dummy_test_key"
    config.ai.groq_model = "qwen/qwen3.8-27b"
    checker = SystemHealthChecker(config=config)
    res = checker.check_groq_api_connectivity()

    assert res.status == "WARN"
    assert "rate limit or quota exceeded" in res.message
    mock_client.chat.completions.create.assert_called_once()


def test_check_groq_api_connection_error(monkeypatch) -> None:
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(request=req)
    monkeypatch.setattr("groq.Groq", lambda *args, **kwargs: mock_client)

    config = AppConfig()
    config.ai.groq_api_key = "gsk_dummy_test_key"
    config.ai.groq_model = "qwen/qwen3.8-27b"
    checker = SystemHealthChecker(config=config)
    res = checker.check_groq_api_connectivity()

    assert res.status == "WARN"
    assert "connection or timeout error" in res.message
    mock_client.chat.completions.create.assert_called_once()


def test_check_groq_api_max_retries_zero(monkeypatch) -> None:
    init_kwargs = {}

    def fake_groq(**kwargs):
        init_kwargs.update(kwargs)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        return mock_client

    monkeypatch.setattr("groq.Groq", fake_groq)

    config = AppConfig()
    config.ai.groq_api_key = "gsk_test_key"
    config.ai.groq_model = "qwen/qwen3.8-27b"
    checker = SystemHealthChecker(config=config)
    res = checker.check_groq_api_connectivity()

    assert res.status == "PASS"
    assert init_kwargs.get("max_retries") == 0
    assert init_kwargs.get("timeout") == 10.0
