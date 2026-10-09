"""Multi-tier quality gate enforcing authentic, production-grade development standards."""

import ast
from pathlib import Path

from pydantic import BaseModel, Field

from agent.core.config import QualityGateConfig
from agent.selection.models import TaskCandidate
from agent.synthesis.models import ImplementationPlan
from agent.utils.logger import setup_logger
from agent.utils.security import validate_file_containment
from agent.validation.linter import LintRunResult
from agent.validation.test_runner import TestRunResult

logger = setup_logger("agent.validation.quality_gate")


class QualityGateResult(BaseModel):
    """Evaluation outcome from the quality gatekeeper."""

    passed: bool
    score: float
    rejection_reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, int | float | str] = Field(default_factory=dict)


class QualityGate:
    """Enforces non-triviality, test coverage, clean lint, and strict task scope containment."""

    def __init__(self, config: QualityGateConfig | None = None) -> None:
        self.config = config or QualityGateConfig()

    def evaluate(
        self,
        candidate: TaskCandidate,
        plan: ImplementationPlan,
        test_result: TestRunResult,
        lint_result: LintRunResult,
        target_path: Path,
    ) -> QualityGateResult:
        """Evaluates whether the proposed changes meet production portfolio standards."""
        rejection_reasons: list[str] = []
        target_base = target_path.resolve()

        # Gate 1: Scope Containment & Security Check
        allowed_files = set(candidate.target_files + candidate.test_files)
        for gen_file in plan.files:
            rel_norm = Path(gen_file.relative_path).as_posix()
            try:
                validate_file_containment(rel_norm, target_base)
            except Exception as e:
                rejection_reasons.append(f"Security containment violation for '{rel_norm}': {e}")

            # Allow necessary root config files if bootstrap task
            if candidate.task_type.value == "FOUNDATION_SETUP":
                continue

            if rel_norm not in allowed_files:
                rejection_reasons.append(
                    f"Out-of-Scope File Violation: '{rel_norm}' was not in candidate task scope: {allowed_files}"
                )

        # Gate 2: AST Syntax and Importability
        total_loc = 0
        functional_definitions = 0
        for gen_file in plan.files:
            if gen_file.relative_path.endswith(".py"):
                try:
                    tree = ast.parse(gen_file.content, filename=gen_file.relative_path)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            functional_definitions += 1
                except SyntaxError as e:
                    rejection_reasons.append(
                        f"Syntax Error in '{gen_file.relative_path}' at line {e.lineno}: {e.msg}"
                    )

            # Count non-blank, non-comment lines
            lines = [
                line.strip()
                for line in gen_file.content.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            total_loc += len(lines)

        # Gate 3: Test Suite Passing
        if candidate.test_files:
            if not test_result.success or test_result.failed_count > 0:
                rejection_reasons.append(
                    f"Unit tests failed (exit code {test_result.exit_code}, {test_result.failed_count} failures): "
                    f"{test_result.failure_summary or 'See test logs.'}"
                )
            if test_result.passed_count == 0:
                rejection_reasons.append(
                    "Zero unit tests executed. A valid development task must implement and execute tests."
                )

        # Gate 4: Lint Cleanliness
        if self.config.enforce_clean_lint and not lint_result.success:
            rejection_reasons.append(
                f"Linter (Ruff) violations detected ({lint_result.error_count} errors): "
                f"{lint_result.output[:300]}"
            )

        # Gate 5: Non-Triviality Check (Reject empty / cosmetic commits)
        if total_loc < self.config.minimum_lines_changed:
            rejection_reasons.append(
                f"Trivial Change Rejection: Total non-comment lines ({total_loc}) is below "
                f"minimum threshold ({self.config.minimum_lines_changed}). Work must be substantive."
            )

        if total_loc > self.config.maximum_lines_changed:
            rejection_reasons.append(
                f"Change Too Large: Total lines ({total_loc}) exceeds maximum threshold "
                f"({self.config.maximum_lines_changed}). Task should be broken into discrete increments."
            )

        if functional_definitions == 0 and candidate.task_type.value not in (
            "FOUNDATION_SETUP",
            "DOCUMENTATION",
        ):
            rejection_reasons.append(
                "Artificial Change Rejection: No functions or classes were defined in the Python files."
            )

        passed = len(rejection_reasons) == 0
        score = 0.95 if passed else max(0.0, 0.9 - (len(rejection_reasons) * 0.3))

        logger.info(
            f"Quality Gate Outcome: {'PASSED' if passed else 'REJECTED'} (Score: {score:.2f}, "
            f"LOC: {total_loc}, Tests Passed: {test_result.passed_count})"
        )
        if rejection_reasons:
            for reason in rejection_reasons:
                logger.warning(f"Quality Gate Rejection Reason: {reason}")

        return QualityGateResult(
            passed=passed,
            score=round(score, 2),
            rejection_reasons=rejection_reasons,
            metrics={
                "total_lines_of_code": total_loc,
                "functional_definitions": functional_definitions,
                "tests_passed": test_result.passed_count,
                "tests_failed": test_result.failed_count,
                "lint_errors": lint_result.error_count,
            },
        )
