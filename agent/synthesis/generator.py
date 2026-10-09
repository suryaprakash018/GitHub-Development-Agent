"""Code and test synthesizer leveraging LLM structured output schemas."""

from agent.git.analyzer import TargetRepoAnalysis
from agent.llm.base import LLMProvider
from agent.selection.models import TaskCandidate
from agent.synthesis.models import ImplementationPlan
from agent.utils.logger import setup_logger

logger = setup_logger("agent.synthesis.generator")

SYSTEM_SYNTHESIS_PROMPT = """You are a senior software architect and AI/data engineer building a high-quality, production-grade technical portfolio.
Your mission is to generate clean, robust, and mathematically sound Python implementations and unit tests.

STRICT ENGINEERING STANDARDS:
1. Production Code:
   - Provide COMPLETE, functional implementations. NO placeholders, NO `TODO`, NO ellipses (`...`), and NO stubbed `pass` functions.
   - Use strict type annotations everywhere (Python 3.11+).
   - Write clean docstrings (Google style or NumPy style) explaining inputs, outputs, and edge cases.
   - Handle numerical stability, division by zero, and invalid types cleanly.

2. Comprehensive Testing:
   - Every module must be accompanied by comprehensive `pytest` unit tests.
   - Tests must validate normal behavior, boundary conditions, edge cases, and error handling.
   - Tests must use standard assertions and fixtures where appropriate.

3. Scope Discipline:
   - ONLY produce the files listed in the target files and test files specifications.
   - Return valid JSON strictly adhering to the ImplementationPlan schema.

4. Foundation Setup & Dependency Discipline:
   - For foundation setup tasks (task_type=FOUNDATION_SETUP, e.g. pyproject.toml):
     * Minimize runtime dependencies. If no functional runtime Python code is introduced in this foundation commit, set runtime `dependencies = []`.
     * Do NOT add heavy track-specific runtime dependencies (such as torch/PyTorch) unless they are genuinely required by code introduced in that foundation commit.
     * Keep only dependencies genuinely required for development, testing, and tooling in `[project.optional-dependencies] dev` (e.g., pytest, pytest-cov, ruff, mypy).
"""


class TaskImplementationGenerator:
    """Synthesizes complete, tested code implementations for a selected TaskCandidate."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm = llm_provider

    def generate_plan(
        self,
        candidate: TaskCandidate,
        analysis: TargetRepoAnalysis,
    ) -> ImplementationPlan:
        """Generates an ImplementationPlan containing all functional code and tests."""
        logger.info(
            f"Generating implementation plan for task: [{candidate.task_id}] '{candidate.title}'"
        )

        # Compile repository context
        existing_py = analysis.python_modules[:15]
        existing_tests = analysis.test_files[:15]

        foundation_instructions = ""
        if candidate.task_type.value == "FOUNDATION_SETUP":
            foundation_instructions = """
FOUNDATION SETUP SPECIFIC RULES:
- Minimize runtime dependencies in pyproject.toml: keep runtime `dependencies = []` since no runtime Python code is introduced in this foundation commit.
- Do NOT include heavy track-specific runtime dependencies like PyTorch (torch) in dependencies.
- Place development/testing/tooling dependencies in `[project.optional-dependencies] dev` (pytest, pytest-cov, ruff, mypy).
"""

        prompt = f"""Generate the complete implementation plan for the following development task:

TASK SPECIFICATION:
- Task ID: {candidate.task_id}
- Title: {candidate.title}
- Description: {candidate.description}
- Task Type: {candidate.task_type.value}
- Required Target Files: {candidate.target_files}
- Required Test Files: {candidate.test_files}
- Engineering Rationale: {candidate.rationale}
{foundation_instructions}
REPOSITORY ARCHITECTURE CONTEXT:
- Existing Modules in Target Repository: {existing_py}
- Existing Test Files: {existing_tests}

REQUIREMENT:
Generate the COMPLETE code and complete test files. Each file must have the full, valid source content ready to be saved to disk.
"""

        plan = self.llm.generate_structured(
            prompt=prompt,
            schema=ImplementationPlan,
            system_prompt=SYSTEM_SYNTHESIS_PROMPT,
        )

        logger.info(
            f"Successfully generated implementation plan with {len(plan.files)} files "
            f"for task '{candidate.task_id}'"
        )
        return plan
