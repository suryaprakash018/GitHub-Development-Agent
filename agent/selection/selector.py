"""Intelligent task selection and anti-duplicate decision engine."""

from agent.git.analyzer import TargetRepoAnalysis
from agent.roadmap.engine import RoadmapEngine
from agent.roadmap.models import MilestoneSpec, ProjectSpec, TrackSpec
from agent.selection.models import TaskCandidate, TaskType
from agent.state.models import AgentState, TaskStatus
from agent.utils.logger import setup_logger

logger = setup_logger("agent.selection")


class TaskSelector:
    """Evaluates repository state and roadmap DAG to select the highest-leverage task."""

    def __init__(self, roadmap_engine: RoadmapEngine | None = None) -> None:
        self.roadmap_engine = roadmap_engine or RoadmapEngine()

    def select_next_task(
        self,
        analysis: TargetRepoAnalysis,
        state: AgentState,
    ) -> TaskCandidate | None:
        """Determines the next development task based on portfolio health, test coverage, and roadmap progression."""
        logger.debug("Evaluating target repository analysis and roadmap for task selection...")

        # Rule 1: Portfolio Infrastructure Bootstrap (if brand new target repo)
        if not analysis.has_readme or not analysis.has_dependency_spec:
            bootstrap_task_id = "task_portfolio_bootstrap"
            if not self._is_task_already_done(bootstrap_task_id, state):
                logger.info("Selected portfolio infrastructure bootstrap task.")
                return TaskCandidate(
                    task_id=bootstrap_task_id,
                    track_id="foundation",
                    project_id="infrastructure",
                    milestone_id="ms_bootstrap",
                    title="Initialize Portfolio Architecture and Engineering Infrastructure",
                    description="Set up production-grade pyproject.toml dependencies, master README, and modular layout.",
                    task_type=TaskType.FOUNDATION_SETUP,
                    target_files=["pyproject.toml", "README.md", ".gitignore"],
                    test_files=[],
                    rationale="A production AI repository requires rigorous dependency management and clean portfolio documentation.",
                    portfolio_value_score=0.95,
                )

        # Rule 2: Project Hardening / Missing Test Coverage
        untested_module = self._find_untested_module(analysis)
        if untested_module:
            module_name = untested_module.split("/")[-1].replace(".py", "")
            hardening_task_id = f"task_test_coverage_{module_name}"
            if not self._is_task_already_done(hardening_task_id, state):
                test_path = f"tests/test_{module_name}.py"
                logger.info(
                    f"Selected project hardening task for untested module: '{untested_module}'"
                )
                return TaskCandidate(
                    task_id=hardening_task_id,
                    track_id=state.metrics.active_track_id,
                    project_id=state.metrics.active_project_id,
                    milestone_id=state.metrics.active_milestone_id,
                    title=f"Add Comprehensive Test Suite for {module_name}",
                    description=f"Implement edge-case unit tests and numerical stability validations for '{untested_module}'.",
                    task_type=TaskType.PROJECT_HARDENING,
                    target_files=[untested_module],
                    test_files=[test_path],
                    rationale="Code without tests violates portfolio quality standards. Hardening existing code takes priority.",
                    portfolio_value_score=0.90,
                )

        # Rule 3: Sequential Roadmap Progression
        next_milestone_tuple = self.roadmap_engine.get_next_available_milestone(state)
        if not next_milestone_tuple:
            logger.info("All roadmap tracks and milestones are 100% completed.")
            return None

        track, project, milestone = next_milestone_tuple
        candidate = self._generate_milestone_task(track, project, milestone, analysis, state)

        if candidate:
            logger.info(
                f"Selected roadmap milestone task: [{candidate.task_id}] '{candidate.title}'"
            )
            return candidate

        return None

    def _generate_milestone_task(
        self,
        track: TrackSpec,
        project: ProjectSpec,
        milestone: MilestoneSpec,
        analysis: TargetRepoAnalysis,
        state: AgentState,
    ) -> TaskCandidate | None:
        """Constructs a high-value task candidate for a specific roadmap milestone."""
        # Derive standard directory and module names
        clean_track = track.id.replace("track_", "")
        clean_proj = project.id.replace("proj_", "")
        clean_ms = milestone.id.replace("ms_", "")

        module_filename = f"{clean_ms}.py"
        target_file = f"src/{clean_track}/{clean_proj}/{module_filename}"
        test_file = f"tests/{clean_track}/{clean_proj}/test_{clean_ms}.py"

        task_id = f"task_{milestone.id}"

        # Anti-duplicate check
        if self._is_task_already_done(task_id, state):
            logger.debug(f"Task '{task_id}' already completed in state. Skipping.")
            return None

        if target_file in analysis.existing_files and test_file in analysis.existing_files:
            logger.debug(
                f"Both '{target_file}' and '{test_file}' already exist in target repository. Anti-duplicate triggered."
            )
            return None

        return TaskCandidate(
            task_id=task_id,
            track_id=track.id,
            project_id=project.id,
            milestone_id=milestone.id,
            title=f"Implement {milestone.title}",
            description=f"{milestone.description}. Required validations: {', '.join(milestone.required_tests)}.",
            task_type=TaskType.NEW_MODULE,
            target_files=[target_file],
            test_files=[test_file],
            rationale=f"Advances the {project.title} project in track {track.title} with tested, portfolio-ready code.",
            portfolio_value_score=0.92,
        )

    def _is_task_already_done(self, task_id: str, state: AgentState) -> bool:
        """Verifies if a task ID has already succeeded in persistent state."""
        return any(
            t.task_id == task_id and t.status == TaskStatus.SUCCESS for t in state.task_history
        )

    def _find_untested_module(self, analysis: TargetRepoAnalysis) -> str | None:
        """Finds any existing python module in src/ that lacks a corresponding test file."""
        src_modules = [m for m in analysis.python_modules if m.startswith("src/")]
        test_files = set(analysis.test_files)

        for mod in src_modules:
            base_name = mod.split("/")[-1].replace(".py", "")
            # Check if any test file corresponds to this base_name
            has_test = any(
                f"test_{base_name}.py" in t or f"{base_name}_test.py" in t for t in test_files
            )
            if not has_test:
                return mod
        return None
