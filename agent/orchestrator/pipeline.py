"""Autonomous Development Orchestrator running the complete unattended development loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent.core.config import AppConfig, load_config
from agent.core.context import ExecutionLock, LockError
from agent.core.exceptions import (
    ConfigurationError,
    DecoupledTargetViolationError,
    GitSafetyError,
    SecurityError,
)
from agent.git.analyzer import TargetRepoAnalyzer
from agent.git.client import GitClient
from agent.llm import get_llm_provider
from agent.llm.base import LLMProvider
from agent.roadmap.engine import RoadmapEngine
from agent.selection.models import TaskCandidate
from agent.selection.selector import TaskSelector
from agent.state.manager import StateManager
from agent.synthesis.executor import DryRunExecutionReport, SynthesisExecutionHarness
from agent.telemetry.recorder import TelemetryRecord, TelemetryRecorder
from agent.utils.logger import setup_logger

logger = setup_logger("agent.orchestrator.pipeline")


class OrchestratorRunResult(BaseModel):
    """Result summary of an orchestrator development cycle."""

    task_id: str | None = None
    status: str = Field(
        ...,
        description="Outcome status: DRY_RUN_APPROVED, QUALITY_GATE_REJECTED, NO_TASK_AVAILABLE, ERROR, SUCCESS",
    )
    candidate: TaskCandidate | None = None
    quality_gate_passed: bool = False
    committed: bool = False
    pushed: bool = False
    commit_message: str | None = None
    diff_preview: str = ""
    explanation: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    telemetry_file: str | None = None


class AutonomousDevelopmentOrchestrator:
    """End-to-end autonomous development agent orchestrating roadmap, synthesis, and validation."""

    def __init__(
        self,
        config: AppConfig | None = None,
        llm_provider: LLMProvider | None = None,
        state_manager: StateManager | None = None,
        roadmap_engine: RoadmapEngine | None = None,
    ) -> None:
        self.config = config or load_config()
        self.llm = llm_provider or (
            get_llm_provider(self.config) if self.config.is_llm_configured() else None
        )
        self.state_manager = state_manager or StateManager()
        self.roadmap_engine = roadmap_engine or RoadmapEngine()
        self.selector = TaskSelector(roadmap_engine=self.roadmap_engine)

    def run(
        self,
        scheduled: bool = False,
        force_dry_run: bool | None = None,
    ) -> OrchestratorRunResult:
        """Executes a full unattended development cycle protected by an exclusive concurrency lock.

        Full Pipeline:
        1. Acquire ExecutionLock (.agent.lock)
        2. Validate target repository path and decouple boundaries
        3. Assert clean worktree and verify active branch (main)
        4. Inspect AST and analyze target repo state (read-only)
        5. Select next candidate task from roadmap DAG
        6. Synthesize code, run tests, Ruff linting & closed-loop remediation
        7. Evaluate multi-tier Quality Gate
        8. Synthesize Conventional Commit message
        9. Enforce DRY_RUN / AUTO_COMMIT / AUTO_PUSH safety controls
        """
        logger.info(
            f"Initiating autonomous development run (scheduled={scheduled}, "
            f"dry_run={self.config.operational_mode.dry_run if force_dry_run is None else force_dry_run})..."
        )

        start_time = time.perf_counter()
        result: OrchestratorRunResult

        try:
            with ExecutionLock():
                result = self._execute_cycle(force_dry_run=force_dry_run)
        except LockError as e:
            logger.error(f"Concurrency lock failure: {e}")
            result = OrchestratorRunResult(
                status="LOCK_BUSY",
                explanation=f"Cannot run: another agent instance is currently executing ({e})",
            )
        except (DecoupledTargetViolationError, SecurityError) as e:
            logger.critical(f"Security boundary check failed: {e}")
            result = OrchestratorRunResult(
                status="SECURITY_ERROR",
                explanation=str(e),
            )
        except GitSafetyError as e:
            logger.error(f"Git safety check failed: {e}")
            result = OrchestratorRunResult(
                status="GIT_SAFETY_ERROR",
                explanation=str(e),
            )
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            result = OrchestratorRunResult(
                status="CONFIG_ERROR",
                explanation=str(e),
            )
        except Exception as e:
            logger.exception(f"Unexpected error in orchestrator: {e}")
            result = OrchestratorRunResult(
                status="ERROR",
                explanation=f"Execution halted due to unexpected error: {e}",
            )

        duration = round(time.perf_counter() - start_time, 3)

        # Record structured telemetry and audit trail
        try:
            is_dry_run = (
                self.config.operational_mode.dry_run if force_dry_run is None else force_dry_run
            )
            active_keys = [
                k for k in (self.config.ai.gemini_api_key, self.config.ai.groq_api_key) if k
            ]
            telemetry_recorder = TelemetryRecorder(
                runs_dir=self.state_manager.state_dir / "runs",
                custom_secrets=active_keys,
            )
            telemetry_entry = TelemetryRecord(
                duration_seconds=duration,
                mode={
                    "dry_run": is_dry_run,
                    "auto_commit": self.config.operational_mode.auto_commit,
                    "auto_push": self.config.operational_mode.auto_push,
                },
                target_repo_path=str(self.config.repository.target_path),
                target_branch=self.config.repository.default_branch,
                task_id=result.task_id,
                task_title=result.candidate.title if result.candidate else None,
                status=result.status,
                ast_metrics_before=result.details.get("ast_metrics_before"),
                ast_metrics_after=result.details.get("ast_metrics_after"),
                remediation_attempts=(
                    len(result.details["remediation_attempts"])
                    if isinstance(result.details.get("remediation_attempts"), list)
                    else int(result.details.get("remediation_attempts", 0) or 0)
                ),
                validation=result.details.get("validation", {}),
                quality_gate=result.details.get("quality_gate", {}),
                diff_summary=result.details.get("diff_summary", {}),
                commit_info={
                    "proposed_subject": result.commit_message or "",
                    "created": result.committed,
                    "commit_hash": result.details.get("commit_hash"),
                    "pushed": result.pushed,
                },
                error=(
                    result.explanation
                    if result.status
                    in {
                        "ERROR",
                        "LOCK_BUSY",
                        "SECURITY_ERROR",
                        "GIT_SAFETY_ERROR",
                        "CONFIG_ERROR",
                    }
                    else None
                ),
            )
            saved_file = telemetry_recorder.record(telemetry_entry)
            result.telemetry_file = str(saved_file)
        except Exception as tel_err:
            logger.warning(f"Failed to record execution telemetry: {tel_err}")

        return result

    def _execute_cycle(self, force_dry_run: bool | None = None) -> OrchestratorRunResult:
        """Internal execution cycle once lock has been acquired."""
        # Step 1: Validate target repository path
        target_path: Path = self.config.get_validated_target_path()
        logger.info(f"Operating on decoupled target repository: '{target_path}'")

        # Step 2: Pre-flight Git checks (clean worktree & correct branch)
        git_client = GitClient(target_path=target_path, config=self.config)
        if not git_client.is_git_repository():
            raise GitSafetyError(
                f"Target repository path '{target_path}' is not an initialized Git repository."
            )

        git_client.assert_clean_worktree()
        active_branch = git_client.verify_active_branch()
        logger.info(f"Target repository pre-flight verified. Active branch: '{active_branch}'")

        # Step 3: Read-only AST and repository structure analysis
        analyzer = TargetRepoAnalyzer(target_path=target_path)
        analysis = analyzer.analyze()

        # Step 4: Load persistent state and select next task
        state = self.state_manager.load_state()
        candidate = self.selector.select_next_task(analysis=analysis, state=state)

        if not candidate:
            logger.info(
                "No actionable candidate task selected: All milestones completed or blocked."
            )
            return OrchestratorRunResult(
                status="NO_TASK_AVAILABLE",
                explanation="All roadmap tracks and milestones are 100% completed or prerequisites are not yet met.",
            )

        logger.info(f"Selected candidate task: [{candidate.task_id}] '{candidate.title}'")

        # Step 5: Check AI provider availability
        if self.llm is None:
            return OrchestratorRunResult(
                task_id=candidate.task_id,
                status="AI_UNAVAILABLE",
                candidate=candidate,
                explanation=f"AI LLM Provider is not configured (missing credentials for '{self.config.ai.provider}').",
            )

        # Step 6: Execute synthesis harness in dry-run isolation
        harness = SynthesisExecutionHarness(config=self.config, llm_provider=self.llm)
        report: DryRunExecutionReport = harness.execute_dry_run(candidate)

        ast_metrics_before = {
            "total_python_files": len(analysis.python_modules),
            "total_test_files": len(analysis.test_files),
            "total_symbols": sum(len(s) for s in analysis.module_symbols.values()),
            "python_modules": analysis.python_modules,
        }
        val_summary = {
            "pytest_passed": report.test_result.success if report.test_result else False,
            "tests_run": (
                (report.test_result.passed_count + report.test_result.failed_count)
                if report.test_result
                else 0
            ),
            "failures": report.test_result.failed_count if report.test_result else 0,
            "ruff_clean": report.lint_result.success if report.lint_result else False,
        }
        qg_summary = {
            "decision": report.decision,
            "passed": report.quality_gate_result.passed if report.quality_gate_result else False,
            "score": (report.quality_gate_result.score if report.quality_gate_result else 0.0),
            "reasons": (
                report.quality_gate_result.rejection_reasons if report.quality_gate_result else []
            ),
        }
        diff_summary = {
            "files_changed": (
                [f.relative_path for f in report.proposed_plan.files]
                if report.proposed_plan
                else []
            ),
            "lines_added": (
                sum(len(f.content.splitlines()) for f in report.proposed_plan.files)
                if report.proposed_plan
                else 0
            ),
            "lines_removed": 0,
        }

        # Step 7: Evaluate Quality Gate outcome
        if report.decision != "APPROVED_FOR_COMMIT":
            logger.warning(
                f"Task '{candidate.task_id}' was rejected by quality gate: {report.explanation}"
            )
            return OrchestratorRunResult(
                task_id=candidate.task_id,
                status="QUALITY_GATE_REJECTED",
                candidate=candidate,
                quality_gate_passed=False,
                committed=False,
                pushed=False,
                explanation=report.explanation,
                diff_preview=report.git_diff,
                details={
                    "ast_metrics_before": ast_metrics_before,
                    "validation": val_summary,
                    "quality_gate": qg_summary,
                    "diff_summary": diff_summary,
                    "rejection_reasons": report.quality_gate_result.rejection_reasons,
                    "quality_score": report.quality_gate_result.score,
                    "remediation_attempts": len(report.remediation_attempts),
                    "remediation_history": report.remediation_attempts,
                },
            )

        # Determine effective operational mode
        is_dry_run = (
            self.config.operational_mode.dry_run if force_dry_run is None else force_dry_run
        )
        auto_commit = self.config.operational_mode.auto_commit
        auto_push = self.config.operational_mode.auto_push

        commit_msg_formatted = (
            report.proposed_commit_message.formatted_message
            if report.proposed_commit_message
            else f"feat({candidate.track_id}): implement {candidate.title}"
        )

        # Step 8: DRY_RUN / Safety Mode
        if is_dry_run or not auto_commit:
            logger.info(
                f"Task '{candidate.task_id}' passed all quality gates. "
                f"Simulating commit in DRY_RUN mode (DRY_RUN={is_dry_run}, AUTO_COMMIT={auto_commit})."
            )
            # Rollback was already guaranteed by SynthesisExecutionHarness
            return OrchestratorRunResult(
                task_id=candidate.task_id,
                status="DRY_RUN_APPROVED",
                candidate=candidate,
                quality_gate_passed=True,
                committed=False,
                pushed=False,
                commit_message=commit_msg_formatted,
                diff_preview=report.git_diff,
                explanation="Task passed all validation tests, linter checks, and quality gates. Rollback executed cleanly.",
                details={
                    "ast_metrics_before": ast_metrics_before,
                    "validation": val_summary,
                    "quality_gate": qg_summary,
                    "diff_summary": diff_summary,
                    "quality_score": report.quality_gate_result.score,
                    "passed_tests": report.test_result.passed_count,
                    "remediation_attempts": len(report.remediation_attempts),
                    "remediation_history": report.remediation_attempts,
                },
            )

        # Step 9: Live Commit Execution (Only when DRY_RUN=False and AUTO_COMMIT=True)
        logger.warning(f"Executing LIVE commit for task '{candidate.task_id}'...")
        # Write files permanently
        for gen_file in report.proposed_plan.files:
            target_f = target_path / gen_file.relative_path
            target_f.parent.mkdir(parents=True, exist_ok=True)
            target_f.write_text(gen_file.content, encoding="utf-8")

        # Selectively stage
        git_client.selective_stage([f.relative_path for f in report.proposed_plan.files])
        commit_res = git_client.create_commit(message=commit_msg_formatted)

        pushed = False
        if auto_push:
            push_res = git_client.push()
            pushed = push_res.pushed

        # Update persistent state
        self.state_manager.record_successful_task(
            task_id=candidate.task_id,
            track_id=candidate.track_id,
            project_id=candidate.project_id,
            milestone_id=candidate.milestone_id,
            commit_hash=commit_res.commit_hash or "",
            tests_written=report.test_result.passed_count,
        )

        return OrchestratorRunResult(
            task_id=candidate.task_id,
            status="SUCCESS",
            candidate=candidate,
            quality_gate_passed=True,
            committed=True,
            pushed=pushed,
            commit_message=commit_msg_formatted,
            diff_preview=report.git_diff,
            explanation=f"Successfully committed task '{candidate.task_id}' ({commit_res.commit_hash})",
            details={
                "commit_hash": commit_res.commit_hash,
                "ast_metrics_before": ast_metrics_before,
                "validation": val_summary,
                "quality_gate": qg_summary,
                "diff_summary": diff_summary,
                "quality_score": report.quality_gate_result.score,
                "passed_tests": report.test_result.passed_count,
                "remediation_attempts": len(report.remediation_attempts),
                "remediation_history": report.remediation_attempts,
            },
        )
