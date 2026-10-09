"""Synthesis execution harness managing implementation, remediation, quality gating, and dry-run rollback."""

import contextlib
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from agent.core.config import AppConfig
from agent.git.analyzer import TargetRepoAnalyzer
from agent.git.commit_engine import SemanticCommitEngine, SemanticCommitMessage
from agent.llm.base import LLMProvider
from agent.selection.models import TaskCandidate
from agent.synthesis.generator import TaskImplementationGenerator
from agent.synthesis.models import ImplementationPlan
from agent.synthesis.remediator import FailureRemediator
from agent.utils.logger import setup_logger
from agent.utils.security import validate_file_containment
from agent.validation.linter import LinterRunner, LintRunResult
from agent.validation.quality_gate import QualityGate, QualityGateResult
from agent.validation.test_runner import TestRunner, TestRunResult

logger = setup_logger("agent.synthesis.executor")


class DryRunExecutionReport(BaseModel):
    """Complete diagnostic report of a dry-run synthesis execution."""

    task_id: str
    task_title: str
    target_files: list[str]
    test_files: list[str]
    proposed_plan: ImplementationPlan
    test_result: TestRunResult
    lint_result: LintRunResult
    quality_gate_result: QualityGateResult
    remediation_attempts: list[str] = Field(default_factory=list)
    git_diff: str = ""
    decision: str = "PENDING"
    explanation: str = ""
    proposed_commit_message: SemanticCommitMessage | None = None


class SynthesisExecutionHarness:
    """Orchestrates code generation, validation, closed-loop remediation, and dry-run isolation."""

    def __init__(
        self,
        config: AppConfig,
        llm_provider: LLMProvider,
    ) -> None:
        self.config = config
        self.target_path = config.get_validated_target_path()
        self.llm = llm_provider
        self.generator = TaskImplementationGenerator(llm_provider)
        self.remediator = FailureRemediator(llm_provider)
        self.test_runner = TestRunner(self.target_path)
        self.linter_runner = LinterRunner(self.target_path)
        self.quality_gate = QualityGate(config.quality_gate)
        self.analyzer = TargetRepoAnalyzer(self.target_path)

    def execute_dry_run(self, candidate: TaskCandidate) -> DryRunExecutionReport:
        """Executes full synthesis, testing, and remediation in DRY_RUN mode, then restores the target repo."""
        logger.info(f"Starting dry-run execution harness for task '{candidate.task_id}'...")

        # Step 1: Enforce clean worktree safety
        self.analyzer.assert_safe_to_operate()
        analysis = self.analyzer.analyze()

        # Step 2: Track initial state of files to guarantee 100% rollback in dry-run mode
        initial_file_states = self._snapshot_target_files(
            candidate.target_files + candidate.test_files
        )
        created_paths: list[Path] = []

        remediation_diagnoses: list[str] = []
        git_diff_output = ""

        try:
            # Step 3: Synthesize implementation plan
            plan = self.generator.generate_plan(candidate, analysis)

            # Step 4: Apply files temporarily to disk
            created_paths = self._apply_plan_files(plan)

            # Format files
            self.linter_runner.format_files([f.relative_path for f in plan.files])

            # Step 5: Initial validation run
            lint_res = self.linter_runner.run_linter([f.relative_path for f in plan.files])
            test_res = self.test_runner.run_tests(
                candidate.test_files if candidate.test_files else None
            )

            # Step 6: Closed-loop remediation if test or lint failed
            attempt = 0
            max_retries = self.config.operational_mode.max_retries
            while (
                (candidate.test_files and not test_res.success) or not lint_res.success
            ) and attempt < max_retries:
                attempt += 1
                failure_summary = (
                    f"TEST FAILURES:\n{test_res.failure_summary or test_res.stdout}\n"
                    if not test_res.success
                    else f"LINT FAILURES:\n{lint_res.output}\n"
                )
                logger.warning(
                    f"Validation failed on attempt {attempt}. Triggering auto-remediator..."
                )

                plan, diagnosis = self.remediator.remediate(
                    candidate, plan, failure_summary, attempt
                )
                remediation_diagnoses.append(f"Attempt {attempt}: {diagnosis}")

                # Re-apply fixed files
                self._apply_plan_files(plan)
                self.linter_runner.format_files([f.relative_path for f in plan.files])

                # Re-run validation
                lint_res = self.linter_runner.run_linter([f.relative_path for f in plan.files])
                test_res = self.test_runner.run_tests(
                    candidate.test_files if candidate.test_files else None
                )

            # Step 7: Evaluate Quality Gate
            qg_result = self.quality_gate.evaluate(
                candidate=candidate,
                plan=plan,
                test_result=test_res,
                lint_result=lint_res,
                target_path=self.target_path,
            )

            # Step 8: Capture Git Diff
            git_diff_output = self._capture_git_diff()

            decision = "APPROVED_FOR_COMMIT" if qg_result.passed else "REJECTED_BY_QUALITY_GATE"
            explanation = (
                "Proposed work satisfied all tests, linting, and quality gate standards."
                if qg_result.passed
                else f"Quality gate rejected changes: {'; '.join(qg_result.rejection_reasons)}"
            )

            # Step 9: Generate Semantic Commit Message if approved
            proposed_commit_msg = None
            if qg_result.passed:
                commit_engine = SemanticCommitEngine(self.llm)
                proposed_commit_msg = commit_engine.generate_commit_message(
                    candidate=candidate,
                    plan=plan,
                    diff=git_diff_output,
                    test_result=test_res,
                    lint_result=lint_res,
                    quality_score=qg_result.score,
                )

            return DryRunExecutionReport(
                task_id=candidate.task_id,
                task_title=candidate.title,
                target_files=candidate.target_files,
                test_files=candidate.test_files,
                proposed_plan=plan,
                test_result=test_res,
                lint_result=lint_res,
                quality_gate_result=qg_result,
                remediation_attempts=remediation_diagnoses,
                git_diff=git_diff_output,
                decision=decision,
                explanation=explanation,
                proposed_commit_message=proposed_commit_msg,
            )

        finally:
            # Step 9: DRY_RUN Guarantees -> Always rollback target repository to initial state!
            logger.info(
                "Executing dry-run rollback to restore target repository to pristine state..."
            )
            self._rollback(initial_file_states, created_paths)

    def _snapshot_target_files(self, relative_files: list[str]) -> dict[str, str | None]:
        """Snapshots current content of files (or None if they do not exist yet)."""
        snapshot: dict[str, str | None] = {}
        for rel in relative_files:
            abs_p = self.target_path / rel
            if abs_p.is_file():
                try:
                    snapshot[rel] = abs_p.read_text(encoding="utf-8")
                except Exception:
                    snapshot[rel] = None
            else:
                snapshot[rel] = None
        return snapshot

    def _apply_plan_files(self, plan: ImplementationPlan) -> list[Path]:
        """Safely writes plan files to the target repository, enforcing containment boundaries."""
        created_paths: list[Path] = []
        for gen_file in plan.files:
            target_file_path = validate_file_containment(gen_file.relative_path, self.target_path)
            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            target_file_path.write_text(gen_file.content, encoding="utf-8")
            created_paths.append(target_file_path)
        return created_paths

    def _capture_git_diff(self) -> str:
        """Captures `git diff` or unstaged changes in target repository."""
        try:
            res = subprocess.run(
                ["git", "diff"],
                cwd=self.target_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            diff = res.stdout.strip()
            if not diff:
                # If newly created files are untracked, show status summary
                status_res = subprocess.run(
                    ["git", "status", "--short"],
                    cwd=self.target_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return f"[Untracked files created]:\n{status_res.stdout.strip()}"
            return diff
        except Exception as e:
            return f"[Error capturing diff: {e}]"

    def _rollback(
        self,
        initial_file_states: dict[str, str | None],
        created_paths: list[Path],
    ) -> None:
        """Restores modified files and removes newly created files."""
        # 1. Clean newly created files
        for p in created_paths:
            if p.is_file():
                rel = p.relative_to(self.target_path).as_posix()
                if rel in initial_file_states and initial_file_states[rel] is None:
                    try:
                        p.unlink()
                    except Exception as e:
                        logger.warning(f"Error removing dry-run file {p}: {e}")

        # 2. Clean bytecode directories generated during testing
        for pycache_dir in self.target_path.glob("**/__pycache__"):
            with contextlib.suppress(Exception):
                shutil.rmtree(pycache_dir, ignore_errors=True)

        # 3. Clean .pytest_cache if created during run
        pytest_cache = self.target_path / ".pytest_cache"
        if pytest_cache.is_dir() and ".pytest_cache" not in initial_file_states:
            with contextlib.suppress(Exception):
                shutil.rmtree(pytest_cache, ignore_errors=True)

        # 4. Clean empty parent directories for all created paths up to target root
        for p in created_paths:
            parent = p.parent
            while parent != self.target_path and parent.exists():
                try:
                    if not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                    else:
                        break
                except Exception:
                    break

        # 5. Restore modified files to initial content
        for rel, orig_content in initial_file_states.items():
            if orig_content is not None:
                p = self.target_path / rel
                try:
                    p.write_text(orig_content, encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Error restoring dry-run file {p}: {e}")
