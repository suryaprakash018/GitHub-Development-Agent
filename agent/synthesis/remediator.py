"""Auto-remediation engine diagnosing test and lint failures to produce targeted fixes."""

from agent.llm.base import LLMProvider
from agent.selection.models import TaskCandidate
from agent.synthesis.models import ImplementationPlan, RemediationPlan
from agent.utils.logger import setup_logger

logger = setup_logger("agent.synthesis.remediator")

SYSTEM_REMEDIATION_PROMPT = """You are an expert software engineer and debugger.
Your task is to analyze test execution failures, compiler/syntax errors, or linter violations, and produce a TARGETED, MINIMAL fix.

STRICT INSTRUCTIONS:
1. Carefully inspect the error log, traceback, and failed assertions.
2. Fix ONLY the specific lines, imports, or assertion mismatches causing the failure.
3. Preserve the full functional implementation and architectural structure.
4. Return COMPLETE updated file contents (not diffs or partial snippets) for every file that required fixing.
"""


class FailureRemediator:
    """Diagnoses validation failures and produces targeted code patches."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm = llm_provider

    def remediate(
        self,
        candidate: TaskCandidate,
        current_plan: ImplementationPlan,
        failure_log: str,
        attempt_number: int,
    ) -> tuple[ImplementationPlan, str]:
        """Analyzes failure diagnostics and returns an updated ImplementationPlan with the targeted fix."""
        logger.warning(
            f"Initiating remediation attempt {attempt_number} for task '{candidate.task_id}'..."
        )

        # Build context of files currently in the plan
        files_context = "\n\n".join(
            f"--- FILE: {f.relative_path} ---\n{f.content}" for f in current_plan.files
        )

        prompt = f"""REMEDIATION REQUEST:
Task ID: {candidate.task_id}
Title: {candidate.title}
Attempt Number: {attempt_number}

FAILURE LOG / TRACEBACK:
{failure_log}

CURRENT IMPLEMENTATION FILES:
{files_context}

INSTRUCTIONS:
1. Diagnose the exact cause of the failure shown in the traceback.
2. Provide updated, complete source code for each file that needs modification to fix the issue.
3. Ensure all tests will pass after your fix.
"""

        remediation = self.llm.generate_structured(
            prompt=prompt,
            schema=RemediationPlan,
            system_prompt=SYSTEM_REMEDIATION_PROMPT,
        )

        logger.info(f"Remediation Diagnosis (Attempt {attempt_number}): {remediation.diagnosis}")

        # Update files in current_plan
        file_map = {f.relative_path: f for f in current_plan.files}
        for fixed_file in remediation.fixed_files:
            file_map[fixed_file.relative_path] = fixed_file

        updated_plan = ImplementationPlan(
            task_id=current_plan.task_id,
            summary=f"{current_plan.summary} (Remediated: {remediation.diagnosis})",
            files=list(file_map.values()),
            expected_test_count=current_plan.expected_test_count,
        )

        return updated_plan, remediation.diagnosis
