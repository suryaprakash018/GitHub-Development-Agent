"""Unit tests for ExecutionLock, StateManager atomic updates, and SQLite telemetry."""

import pytest

from agent.core.context import ExecutionLock, LockError
from agent.state.manager import StateManager
from agent.state.models import TaskExecutionRecord, TaskStatus


def test_execution_lock_lifecycle(tmp_path):
    lock_file = tmp_path / ".agent.lock"

    # Acquire and release using context manager
    with ExecutionLock(lockfile_path=lock_file):
        assert lock_file.exists()
        # Attempting second lock should raise LockError
        with pytest.raises(LockError):
            ExecutionLock(lockfile_path=lock_file).acquire()

    assert not lock_file.exists()


def test_state_manager_default_initialization(tmp_path):
    manager = StateManager(state_dir=tmp_path)
    state = manager.load_state()

    assert state.schema_version == "1.0.0"
    assert state.metrics.total_tasks_completed == 0
    assert state.metrics.total_commits_created == 0
    assert (tmp_path / "agent_state.json").exists()
    assert (tmp_path / "agent_runs.db").exists()


def test_state_manager_record_successful_task(tmp_path):
    manager = StateManager(state_dir=tmp_path)

    record = TaskExecutionRecord(
        task_id="task_ds_001",
        project_id="proj_ds_01",
        milestone_id="ms_ds_01_01",
        title="Vectorized outlier detection",
        status=TaskStatus.SUCCESS,
        commit_hash="a1b2c3d4",
        tests_passed=4,
        quality_score=0.95,
        execution_time_seconds=2.5,
    )

    manager.record_task_run(record)

    # Reload state and verify
    state = manager.load_state()
    assert state.metrics.total_tasks_completed == 1
    assert state.metrics.total_commits_created == 1
    assert state.metrics.total_tests_written == 4
    assert len(state.task_history) == 1
    assert state.task_history[0].task_id == "task_ds_001"
    assert manager.is_task_completed("task_ds_001") is True
    assert manager.is_task_completed("non_existent_task") is False


def test_state_manager_atomic_save_robustness(tmp_path):
    manager = StateManager(state_dir=tmp_path)
    state = manager.load_state()
    state.metrics.total_tasks_completed = 42

    manager.save_state(state)

    reloaded = manager.load_state()
    assert reloaded.metrics.total_tasks_completed == 42
    # Ensure temporary file was cleanly moved
    assert not (tmp_path / "agent_state.json.tmp").exists()
