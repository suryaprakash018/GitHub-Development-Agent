"""Unit tests for configuration loading, environment overrides, and target validation."""

import pytest

from agent.core.config import load_config
from agent.core.exceptions import ConfigurationError, DecoupledTargetViolationError
from agent.utils.security import get_agent_root_path


def test_load_default_config(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    config = load_config()
    assert config.system.name == "Autonomous AI + Data Development Agent"
    assert config.operational_mode.dry_run is True
    assert config.operational_mode.auto_commit is False
    assert config.operational_mode.auto_push is False
    assert config.ai.provider == "gemini"
    assert config.ai.primary_model == "gemini-3.8-flash"
    assert config.ai.reasoning_model is None
    assert config.ai.thinking_level is None


def test_environment_variable_overrides(monkeypatch, tmp_path):
    external_repo = tmp_path / "External-AI-Portfolio"
    external_repo.mkdir()

    monkeypatch.setenv("TARGET_REPOSITORY_PATH", str(external_repo))
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_123")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-model")
    monkeypatch.setenv("THINKING_LEVEL", "LOW")
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("TARGET_BRANCH", "develop")

    config = load_config()

    assert config.repository.target_path == str(external_repo)
    assert config.ai.gemini_api_key == "test_gemini_key_123"
    assert config.ai.primary_model == "gemini-custom-model"
    assert config.ai.thinking_level == "LOW"
    assert config.operational_mode.dry_run is True
    assert config.repository.default_branch == "develop"

    validated_target = config.get_validated_target_path()
    assert validated_target == external_repo.resolve()


def test_target_path_unconfigured_raises_error(monkeypatch):
    monkeypatch.delenv("TARGET_REPOSITORY_PATH", raising=False)
    config = load_config()
    config.repository.target_path = ""

    with pytest.raises(ConfigurationError, match="Target repository path is not configured"):
        config.get_validated_target_path()


def test_target_path_pointing_to_agent_raises_decoupled_error(monkeypatch):
    agent_root = get_agent_root_path()
    monkeypatch.setenv("TARGET_REPOSITORY_PATH", str(agent_root))

    config = load_config()
    with pytest.raises(DecoupledTargetViolationError):
        config.get_validated_target_path()
