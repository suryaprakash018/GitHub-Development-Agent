"""Unit tests for security boundaries, path containment, and decoupled repository rules."""

import pytest

from agent.core.exceptions import DecoupledTargetViolationError, SecurityError
from agent.utils.security import (
    get_agent_root_path,
    validate_file_containment,
    validate_target_repository_path,
)


def test_agent_root_path_resolution():
    agent_root = get_agent_root_path()
    assert agent_root.is_dir()
    assert (agent_root / "pyproject.toml").is_file()


def test_target_path_cannot_be_empty():
    with pytest.raises(SecurityError, match="TARGET_REPOSITORY_PATH is not configured"):
        validate_target_repository_path("")

    with pytest.raises(SecurityError, match="TARGET_REPOSITORY_PATH is not configured"):
        validate_target_repository_path(None)


def test_target_path_cannot_be_agent_root():
    agent_root = get_agent_root_path()
    with pytest.raises(
        DecoupledTargetViolationError,
        match="cannot be the GitHub-Development-Agent repository itself",
    ):
        validate_target_repository_path(agent_root)


def test_target_path_cannot_be_inside_agent_root(tmp_path):
    agent_root = get_agent_root_path()
    nested_in_agent = agent_root / "agent" / "subfolder"
    with pytest.raises(
        DecoupledTargetViolationError,
        match="is located inside the GitHub-Development-Agent repository",
    ):
        validate_target_repository_path(nested_in_agent)


def test_agent_cannot_be_inside_target(monkeypatch, tmp_path):
    # Mock agent_root as a child of target
    target_dir = tmp_path / "mock_target"
    target_dir.mkdir()
    child_agent_root = target_dir / "nested_agent"
    child_agent_root.mkdir()

    monkeypatch.setattr("agent.utils.security.get_agent_root_path", lambda: child_agent_root)

    with pytest.raises(
        DecoupledTargetViolationError, match="is located inside the target repository"
    ):
        validate_target_repository_path(target_dir)


def test_valid_decoupled_target_path(tmp_path):
    external_repo = tmp_path / "External-AI-Portfolio"
    external_repo.mkdir()

    resolved = validate_target_repository_path(external_repo)
    assert resolved == external_repo.resolve()


def test_file_containment_valid(tmp_path):
    base_dir = tmp_path / "target_repo"
    base_dir.mkdir()

    contained_file = validate_file_containment("src/module.py", base_dir)
    assert contained_file == (base_dir / "src" / "module.py").resolve()


def test_file_containment_traversal_attack(tmp_path):
    base_dir = tmp_path / "target_repo"
    base_dir.mkdir()

    with pytest.raises(SecurityError, match="Path Traversal Violation"):
        validate_file_containment("../../../escape.py", base_dir)
