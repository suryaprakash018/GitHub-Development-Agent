"""Semantic Commit Engine generating authentic Conventional Commits based on verified diffs and ASTs."""

from __future__ import annotations

import ast

from pydantic import BaseModel, Field

from agent.llm.base import LLMProvider
from agent.selection.models import TaskCandidate, TaskType
from agent.synthesis.models import ImplementationPlan
from agent.utils.logger import setup_logger
from agent.validation.linter import LintRunResult
from agent.validation.test_runner import TestRunResult

logger = setup_logger("agent.git.commit_engine")


class SemanticCommitMessage(BaseModel):
    """Structured, Conventional-Commits compliant semantic commit message."""

    subject: str = Field(
        ...,
        description="Conventional commit header: <type>(<scope>): <concise imperative summary>",
    )
    body: str = Field(
        ...,
        description="Architectural rationale, technical background, and algorithmic implementation details.",
    )
    key_changes: list[str] = Field(
        default_factory=list,
        description="Bullet points of specific classes, functions, and tests introduced.",
    )
    validation_summary: str = Field(
        ...,
        description="Summary of test run execution and linter compliance.",
    )
    footer: str = Field(
        ...,
        description="Metadata footer including milestone ID, task type, and quality gate score.",
    )
    formatted_message: str = Field(
        ...,
        description="Full, ready-to-commit string combining subject, body, key changes, and footer.",
    )


class SemanticCommitEngine:
    """Generates authentic, high-signal commit messages grounded in actual diffs and AST structures."""

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self.llm = llm_provider

    def generate_commit_message(
        self,
        candidate: TaskCandidate,
        plan: ImplementationPlan,
        diff: str,
        test_result: TestRunResult,
        lint_result: LintRunResult,
        quality_score: float,
    ) -> SemanticCommitMessage:
        """Constructs a production-grade semantic commit message using LLM or deterministic AST analysis."""
        logger.debug(f"Generating semantic commit message for task '{candidate.task_id}'...")

        # If LLM is available, attempt structured LLM synthesis
        if self.llm is not None:
            try:
                commit_msg = self._generate_with_llm(
                    candidate, plan, diff, test_result, lint_result, quality_score
                )
                if commit_msg:
                    return commit_msg
            except Exception as e:
                logger.warning(
                    f"LLM commit message synthesis encountered error ({e}). Falling back to deterministic generator."
                )

        # Deterministic generation from AST symbols, diff statistics, and candidate metadata
        return self._generate_deterministic(
            candidate, plan, diff, test_result, lint_result, quality_score
        )

    def _generate_with_llm(
        self,
        candidate: TaskCandidate,
        plan: ImplementationPlan,
        diff: str,
        test_result: TestRunResult,
        lint_result: LintRunResult,
        quality_score: float,
    ) -> SemanticCommitMessage | None:
        """Uses LLM to synthesize an authentic Conventional Commits message based on the verified diff."""
        prompt = (
            f"You are a principal AI and Data engineer. Generate a Conventional Commits compliant commit message "
            f"for this newly implemented and verified development task.\n\n"
            f"TASK DETAILS:\n"
            f"- Title: {candidate.title}\n"
            f"- Track: {candidate.track_id}\n"
            f"- Project: {candidate.project_id}\n"
            f"- Milestone: {candidate.milestone_id}\n"
            f"- Type: {candidate.task_type.value}\n"
            f"- Rationale: {candidate.rationale}\n\n"
            f"VERIFIED TEST RESULTS:\n"
            f"- Passed: {test_result.passed_count}, Failed: {test_result.failed_count}\n"
            f"- Linter: {'Clean' if lint_result.success else 'Errors present'}\n"
            f"- Quality Score: {quality_score:.2f}/1.00\n\n"
            f"IMPLEMENTATION DIFF:\n"
            f"```diff\n{diff[:3000]}\n```\n\n"
            f"RULES:\n"
            f"1. Subject line MUST follow: <type>(<scope>): <imperative summary under 72 chars>.\n"
            f"   Valid types: feat, test, refactor, perf, chore, docs.\n"
            f"2. Body must explain WHY this work was implemented and algorithmic design decisions.\n"
            f"3. Key changes must enumerate specific classes, algorithms, and test edge-cases.\n"
            f"4. Do NOT use generic placeholder text. Ground all text directly in the actual diff."
        )

        result: SemanticCommitMessage = self.llm.generate_structured(
            prompt=prompt,
            schema=SemanticCommitMessage,
        )
        return result

    def _generate_deterministic(
        self,
        candidate: TaskCandidate,
        plan: ImplementationPlan,
        diff: str,
        test_result: TestRunResult,
        lint_result: LintRunResult,
        quality_score: float,
    ) -> SemanticCommitMessage:
        """Deterministic generator extracting AST symbols, diff metrics, and candidate rationale."""
        # Map TaskType to conventional commit prefix
        type_prefix_map = {
            TaskType.NEW_MODULE: "feat",
            TaskType.TEST_SUITE: "test",
            TaskType.PROJECT_HARDENING: "test",
            TaskType.FOUNDATION_SETUP: "chore",
            TaskType.BENCHMARK: "perf",
            TaskType.DOCUMENTATION: "docs",
        }
        conv_type = type_prefix_map.get(candidate.task_type, "feat")

        # Derive clean scope
        scope = self._derive_scope(candidate)

        # Subject: e.g. feat(data_science/data_cleaning): implement vectorized outlier detection
        clean_title = candidate.title.strip().rstrip(".")
        if clean_title.lower().startswith("implement "):
            clean_title = clean_title[10:].strip()
        elif clean_title.lower().startswith("add "):
            clean_title = clean_title[4:].strip()

        subject = f"{conv_type}({scope}): implement {clean_title.lower()}"
        if len(subject) > 72:
            subject = subject[:69] + "..."

        # Body: Technical rationale
        body_lines = [
            f"Implement {candidate.title.lower()} to advance {candidate.project_id} roadmap.",
            "",
            f"Architecture Rationale: {candidate.rationale}",
            f"Technical Scope: {candidate.description}",
        ]
        body = "\n".join(body_lines)

        # Key Changes from AST analysis of plan files
        key_changes: list[str] = []
        for gen_file in plan.files:
            symbols = self._extract_symbols_from_content(gen_file.content)
            file_desc_parts: list[str] = []
            if symbols["classes"]:
                file_desc_parts.append(f"classes ({', '.join(symbols['classes'])})")
            if symbols["functions"]:
                fn_count = len(symbols["functions"])
                sample_fns = ", ".join(symbols["functions"][:3])
                suffix = "..." if fn_count > 3 else ""
                file_desc_parts.append(f"methods ({sample_fns}{suffix})")

            sym_str = f": introduced {'; '.join(file_desc_parts)}" if file_desc_parts else ""
            key_changes.append(f"- `{gen_file.relative_path}`{sym_str}")

        # Validation summary
        lint_status = (
            "Ruff clean (formatting & imports verified)"
            if lint_result.success
            else "Ruff issues noted"
        )
        validation_summary = (
            f"Verified via Pytest ({test_result.passed_count} passed, {test_result.failed_count} failed) "
            f"and {lint_status}."
        )

        # Footer
        footer_lines = [
            f"Milestone-ID: {candidate.milestone_id}",
            f"Quality-Score: {quality_score:.2f}/1.00",
            f"Task-Type: {candidate.task_type.value}",
        ]
        footer = "\n".join(footer_lines)

        # Combine into complete formatted commit message
        formatted_parts = [
            subject,
            "",
            body,
            "",
            "Key Changes:",
            *key_changes,
            "",
            f"Validation: {validation_summary}",
            "",
            footer,
        ]
        formatted_message = "\n".join(formatted_parts)

        return SemanticCommitMessage(
            subject=subject,
            body=body,
            key_changes=key_changes,
            validation_summary=validation_summary,
            footer=footer,
            formatted_message=formatted_message,
        )

    def _derive_scope(self, candidate: TaskCandidate) -> str:
        """Derives a concise Conventional Commits scope from track and project identifiers."""
        clean_track = candidate.track_id.replace("track_", "").replace("-", "_")
        clean_proj = candidate.project_id.replace("proj_", "").replace("-", "_")
        if clean_track and clean_proj and clean_track != clean_proj:
            return f"{clean_track}/{clean_proj}"
        return clean_track or clean_proj or "portfolio"

    def _extract_symbols_from_content(self, content: str) -> dict[str, list[str]]:
        """Parses Python code to extract declared classes and top-level/method functions."""
        classes: list[str] = []
        functions: list[str] = []
        try:
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                            functions.append(f"{node.name}.{item.name}")
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
        except Exception:
            pass
        return {"classes": classes, "functions": functions}
