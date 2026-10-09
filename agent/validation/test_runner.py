"""Subprocess Pytest runner for target repository validation."""

import re
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from agent.utils.logger import setup_logger

logger = setup_logger("agent.validation.test_runner")


class TestRunResult(BaseModel):
    """Structured result of a pytest execution run."""

    __test__ = False

    success: bool
    exit_code: int
    passed_count: int = 0
    failed_count: int = 0
    stdout: str = ""
    stderr: str = ""
    failure_summary: str | None = None


class TestRunner:
    """Executes Pytest suites in the target repository with timeout and isolation."""

    __test__ = False

    def __init__(self, target_path: Path, timeout_seconds: int = 60) -> None:
        self.target_path = Path(target_path).resolve()
        self.timeout_seconds = timeout_seconds

    def run_tests(self, test_paths: list[str] | None = None) -> TestRunResult:
        """Executes pytest against specified test paths or the entire test suite."""
        cmd = [sys.executable, "-m", "pytest", "-v"]
        if test_paths:
            cmd.extend(test_paths)

        logger.debug(f"Executing pytest in '{self.target_path}': {' '.join(cmd)}")

        try:
            res = subprocess.run(
                cmd,
                cwd=self.target_path,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )

            passed, failed = self._parse_counts(res.stdout)
            failure_summary = None
            is_success = (res.returncode == 0) or (res.returncode == 5 and not test_paths)
            if not is_success and res.returncode != 0:
                failure_summary = self._extract_failure_summary(res.stdout, res.stderr)

            return TestRunResult(
                success=is_success,
                exit_code=res.returncode,
                passed_count=passed,
                failed_count=failed,
                stdout=res.stdout,
                stderr=res.stderr,
                failure_summary=failure_summary,
            )

        except subprocess.TimeoutExpired as e:
            logger.error(f"Pytest execution timed out after {self.timeout_seconds} seconds.")
            return TestRunResult(
                success=False,
                exit_code=124,
                failure_summary=f"Pytest execution timed out after {self.timeout_seconds} seconds: {e}",
            )
        except Exception as e:
            logger.error(f"Failed to execute pytest: {e}")
            return TestRunResult(
                success=False,
                exit_code=1,
                failure_summary=f"Failed to launch pytest subprocess: {e}",
            )

    def _parse_counts(self, stdout: str) -> tuple[int, int]:
        """Extracts passed and failed test counts from pytest stdout."""
        passed = 0
        failed = 0

        # Matches e.g. "5 passed, 1 failed in 0.45s" or "3 passed in 0.20s"
        pass_match = re.search(r"(\d+)\s+passed", stdout)
        if pass_match:
            passed = int(pass_match.group(1))

        fail_match = re.search(r"(\d+)\s+failed", stdout)
        if fail_match:
            failed = int(fail_match.group(1))

        return passed, failed

    def _extract_failure_summary(self, stdout: str, stderr: str) -> str:
        """Extracts concise failure tracebacks to supply to the auto-remediator."""
        lines = stdout.splitlines()
        failure_lines: list[str] = []
        capture = False

        for line in lines:
            if "FAILURES" in line or "ERRORS" in line:
                capture = True
            if "short test summary info" in line:
                capture = True
            if capture:
                failure_lines.append(line)

        if failure_lines:
            # Limit length to avoid blowing prompt window
            return "\n".join(failure_lines[:60])

        return stdout[-1000:] if stdout else stderr[-1000:]
