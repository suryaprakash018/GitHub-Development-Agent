"""Unit tests for task selection heuristics, priority rules, and anti-duplicate guards."""

from agent.git.analyzer import TargetRepoAnalysis
from agent.roadmap.engine import RoadmapEngine
from agent.selection.models import TaskType
from agent.selection.selector import TaskSelector
from agent.state.models import AgentState, TaskExecutionRecord, TaskStatus


def test_selector_proposes_bootstrap_for_fresh_repo():
    selector = TaskSelector()
    empty_analysis = TargetRepoAnalysis(
        target_path="mock/path",
        has_readme=False,
        has_dependency_spec=False,
    )
    state = AgentState()

    candidate = selector.select_next_task(empty_analysis, state)
    assert candidate is not None
    assert candidate.task_type == TaskType.FOUNDATION_SETUP
    assert candidate.task_id == "task_portfolio_bootstrap"
    assert "README.md" in candidate.target_files


def test_selector_prioritizes_hardening_for_untested_module():
    selector = TaskSelector()
    analysis = TargetRepoAnalysis(
        target_path="mock/path",
        has_readme=True,
        has_dependency_spec=True,
        python_modules=["src/data_science/outliers.py"],
        test_files=[],  # No tests exist for outliers.py!
    )
    state = AgentState()

    candidate = selector.select_next_task(analysis, state)
    assert candidate is not None
    assert candidate.task_type == TaskType.PROJECT_HARDENING
    assert "outliers" in candidate.task_id
    assert "tests/test_outliers.py" in candidate.test_files


def test_selector_proposes_next_roadmap_milestone():
    selector = TaskSelector()
    analysis = TargetRepoAnalysis(
        target_path="mock/path",
        has_readme=True,
        has_dependency_spec=True,
        python_modules=["src/data_science/outliers.py"],
        test_files=["tests/data_science/test_outliers.py"],  # Existing code is tested!
    )
    state = AgentState()

    candidate = selector.select_next_task(analysis, state)
    assert candidate is not None
    assert candidate.task_type == TaskType.NEW_MODULE
    assert candidate.milestone_id == "ms_ds_01_01"
    assert len(candidate.target_files) == 1
    assert len(candidate.test_files) == 1


def test_selector_anti_duplicate_skips_completed_task():
    roadmap_engine = RoadmapEngine()
    selector = TaskSelector(roadmap_engine=roadmap_engine)

    analysis = TargetRepoAnalysis(
        target_path="mock/path",
        has_readme=True,
        has_dependency_spec=True,
        python_modules=["src/existing.py"],
        test_files=["tests/test_existing.py"],
    )
    state = AgentState()

    # Record first milestone task as SUCCESS
    state.task_history.append(
        TaskExecutionRecord(
            task_id="task_ms_ds_01_01",
            project_id="proj_ds_01_feature_eng",
            milestone_id="ms_ds_01_01",
            title="Implement outlier detection",
            status=TaskStatus.SUCCESS,
        )
    )

    candidate = selector.select_next_task(analysis, state)
    # Selector should not select task_ms_ds_01_01
    assert candidate is None or candidate.milestone_id != "ms_ds_01_01"
