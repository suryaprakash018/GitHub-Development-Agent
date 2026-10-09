"""Subprocess Ruff linter and formatter runner."""

import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from agent.utils.logger import setup_logger

logger = setup_logger("agent.validation.linter")


class LintRunResult(BaseModel):
    """Structured result of a Ruff lint check execution."""

    success: bool
    exit_code: int = 0
    output: str = ""
    error_count: int = 0


class LinterRunner:
    """Executes Ruff linting and formatting within the target repository."""

    def __init__(self, target_path: Path, timeout_seconds: int = 30) -> None:
        self.target_path = Path(target_path).resolve()
        self.timeout_seconds = timeout_seconds

    def run_linter(self, file_paths: list[str] | None = None) -> LintRunResult:
        """Executes `ruff check` on specified file paths or the whole target repository."""
        py_files = (
            [f for f in file_paths if f.endswith(".py") or f.endswith(".pyi")]
            if file_paths is not None
            else None
        )
        if file_paths is not None and not py_files:
            return LintRunResult(
                success=True, exit_code=0, output="No Python files to lint.", error_count=0
            )

        cmd = [sys.executable, "-m", "ruff", "check"]
        if py_files:
            cmd.extend(py_files)

        logger.debug(f"Executing ruff check in '{self.target_path}': {' '.join(cmd)}")

        try:
            res = subprocess.run(
                cmd,
                cwd=self.target_path,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )

            error_count = len(
                [line for line in res.stdout.splitlines() if "-->" in line or ":" in line]
            )
            if res.returncode == 0:
                error_count = 0

            return LintRunResult(
                success=(res.returncode == 0),
                exit_code=res.returncode,
                output=res.stdout + res.stderr,
                error_count=error_count,
            )

        except subprocess.TimeoutExpired as e:
            logger.error(f"Ruff lint execution timed out: {e}")
            return LintRunResult(
                success=False,
                exit_code=124,
                output=f"Ruff check timed out after {self.timeout_seconds} seconds.",
                error_count=1,
            )
        except Exception as e:
            logger.error(f"Failed to execute ruff check: {e}")
            return LintRunResult(
                success=False,
                exit_code=1,
                output=f"Failed to execute linter subprocess: {e}",
                error_count=1,
            )

    def format_files(self, file_paths: list[str] | None = None) -> bool:
        """Executes `ruff check --select I --fix` and `ruff format` to auto-organize imports and format code."""
        py_files = (
            [f for f in file_paths if f.endswith(".py") or f.endswith(".pyi")]
            if file_paths is not None
            else None
        )
        if file_paths is not None and not py_files:
            return True

        fix_cmd = [sys.executable, "-m", "ruff", "check", "--select", "I", "--fix"]
        fmt_cmd = [sys.executable, "-m", "ruff", "format"]
        if py_files:
            fix_cmd.extend(py_files)
            fmt_cmd.extend(py_files)

        try:
            # 1. Organize imports
            subprocess.run(
                fix_cmd,
                cwd=self.target_path,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            # 2. Format code
            res = subprocess.run(
                fmt_cmd,
                cwd=self.target_path,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            return res.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to format files: {e}")
            return False
