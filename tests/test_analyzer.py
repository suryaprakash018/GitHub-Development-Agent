"""Unit tests for the read-only target repository analyzer."""

import subprocess

import pytest

from agent.core.exceptions import DecoupledTargetViolationError, GitSafetyError
from agent.git.analyzer import TargetRepoAnalyzer
from agent.utils.security import get_agent_root_path


def test_analyzer_rejects_agent_repository():
    agent_root = get_agent_root_path()
    with pytest.raises(DecoupledTargetViolationError):
        TargetRepoAnalyzer(target_path=agent_root)


def test_analyzer_on_empty_target(tmp_path):
    external_repo = tmp_path / "External-AI-Portfolio"
    external_repo.mkdir()

    analyzer = TargetRepoAnalyzer(target_path=external_repo)
    analysis = analyzer.analyze()

    assert analysis.is_git_repo is False
    assert analysis.is_clean is True
    assert len(analysis.existing_files) == 0


def test_analyzer_scans_files_and_extracts_ast(tmp_path):
    external_repo = tmp_path / "External-AI-Portfolio"
    external_repo.mkdir()

    # Create dummy files
    src_dir = external_repo / "src"
    src_dir.mkdir()
    py_file = src_dir / "outliers.py"
    py_file.write_text(
        "class OutlierDetector:\n    def detect(self, data):\n        pass\n\ndef calculate_iqr(x):\n    return x\n",
        encoding="utf-8",
    )

    test_dir = external_repo / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_outliers.py"
    test_file.write_text("def test_dummy():\n    pass\n", encoding="utf-8")

    readme = external_repo / "README.md"
    readme.write_text("# AI Portfolio", encoding="utf-8")

    analyzer = TargetRepoAnalyzer(target_path=external_repo)
    analysis = analyzer.analyze()

    assert analysis.has_readme is True
    assert "src/outliers.py" in analysis.python_modules
    assert "tests/test_outliers.py" in analysis.test_files
    assert "src/outliers.py" in analysis.module_symbols
    assert "OutlierDetector" in analysis.module_symbols["src/outliers.py"]
    assert "detect" in analysis.module_symbols["src/outliers.py"]
    assert "calculate_iqr" in analysis.module_symbols["src/outliers.py"]


def test_analyzer_detects_dirty_worktree(tmp_path):
    external_repo = tmp_path / "External-AI-Portfolio"
    external_repo.mkdir()

    # Initialize a git repo and make initial commit
    subprocess.run(["git", "init"], cwd=external_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=external_repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=external_repo, check=True)

    dummy_file = external_repo / "init.txt"
    dummy_file.write_text("initial", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=external_repo, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=external_repo, check=True)

    analyzer = TargetRepoAnalyzer(target_path=external_repo)
    analysis = analyzer.analyze()
    assert analysis.is_git_repo is True
    assert analysis.is_clean is True

    # Now make an uncommitted modification
    dummy_file.write_text("uncommitted change", encoding="utf-8")
    analysis_dirty = analyzer.analyze()
    assert analysis_dirty.is_clean is False
    assert len(analysis_dirty.uncommitted_files) > 0

    # Ensure assert_safe_to_operate raises GitSafetyError to protect user work
    with pytest.raises(GitSafetyError, match="Safety Guard Activated"):
        analyzer.assert_safe_to_operate()
