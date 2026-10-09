"""Unit tests for subprocess TestRunner and LinterRunner."""

from agent.validation.linter import LinterRunner
from agent.validation.test_runner import TestRunner


def test_test_runner_successful_execution(tmp_path):
    target_dir = tmp_path / "target_repo"
    target_dir.mkdir()

    # Create a passing test
    test_file = target_dir / "test_math.py"
    test_file.write_text("def test_add():\n    assert 2 + 2 == 4\n", encoding="utf-8")

    runner = TestRunner(target_path=target_dir)
    res = runner.run_tests(["test_math.py"])

    assert res.success is True
    assert res.exit_code == 0
    assert res.passed_count == 1
    assert res.failed_count == 0


def test_test_runner_no_tests_collected_without_test_paths(tmp_path):
    target_dir = tmp_path / "target_repo"
    target_dir.mkdir()

    runner = TestRunner(target_path=target_dir)
    res = runner.run_tests(test_paths=None)

    # Pytest returns exit code 5 when no tests are found, which is accepted if no tests requested
    assert res.success is True
    assert res.exit_code == 5


def test_test_runner_failing_execution(tmp_path):
    target_dir = tmp_path / "target_repo"
    target_dir.mkdir()

    # Create a failing test
    test_file = target_dir / "test_fail.py"
    test_file.write_text("def test_broken():\n    assert 1 == 2\n", encoding="utf-8")

    runner = TestRunner(target_path=target_dir)
    res = runner.run_tests(["test_fail.py"])

    assert res.success is False
    assert res.exit_code != 0
    assert res.failed_count == 1
    assert res.failure_summary is not None
    assert "assert 1 == 2" in res.failure_summary


def test_linter_runner_clean_and_dirty(tmp_path):
    target_dir = tmp_path / "target_repo"
    target_dir.mkdir()

    # Clean file
    clean_file = target_dir / "clean.py"
    clean_file.write_text("x = 1\n", encoding="utf-8")

    linter = LinterRunner(target_path=target_dir)
    res_clean = linter.run_linter(["clean.py"])
    assert res_clean.success is True
    assert res_clean.error_count == 0

    # Dirty file with syntax error
    dirty_file = target_dir / "dirty.py"
    dirty_file.write_text("def broken_syntax(:\n", encoding="utf-8")

    res_dirty = linter.run_linter(["dirty.py"])
    assert res_dirty.success is False


def test_linter_runner_non_python_files_skipped(tmp_path):
    target_dir = tmp_path / "target_repo"
    target_dir.mkdir()

    # Non-Python files like markdown and gitignore should not be treated as Python
    md_file = target_dir / "README.md"
    md_file.write_text("# Documentation\n", encoding="utf-8")
    gi_file = target_dir / ".gitignore"
    gi_file.write_text("*.log\n/dist\n", encoding="utf-8")

    linter = LinterRunner(target_path=target_dir)
    res = linter.run_linter(["README.md", ".gitignore"])
    assert res.success is True
    assert res.error_count == 0

    fmt_res = linter.format_files(["README.md", ".gitignore"])
    assert fmt_res is True
