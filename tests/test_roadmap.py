"""Unit tests for the Roadmap DAG engine, prerequisite checks, and progression."""

from agent.roadmap.engine import RoadmapEngine
from agent.state.models import AgentState, MilestoneProgress, MilestoneStatus


def test_roadmap_dag_loads_tracks():
    engine = RoadmapEngine()
    assert len(engine.dag.tracks) >= 6
    track_ids = [t.id for t in engine.dag.tracks]
    assert "track_data_science" in track_ids
    assert "track_machine_learning" in track_ids
    assert "track_deep_learning" in track_ids
    assert "track_generative_ai" in track_ids
    assert "track_data_engineering" in track_ids
    assert "track_mlops_deployment" in track_ids


def test_next_milestone_on_fresh_state():
    engine = RoadmapEngine()
    state = AgentState()

    res = engine.get_next_available_milestone(state)
    assert res is not None
    track, project, milestone = res

    assert track.id == "track_data_science"
    assert project.id == "proj_ds_01_feature_eng"
    assert milestone.id == "ms_ds_01_01"


def test_next_milestone_advances_after_completion():
    engine = RoadmapEngine()
    state = AgentState()

    # Mark first milestone as completed
    state.milestones["ms_ds_01_01"] = MilestoneProgress(
        milestone_id="ms_ds_01_01",
        project_id="proj_ds_01_feature_eng",
        status=MilestoneStatus.COMPLETED,
    )

    res = engine.get_next_available_milestone(state)
    assert res is not None
    track, project, milestone = res

    assert track.id == "track_data_science"
    assert milestone.id == "ms_ds_01_02"


def test_prerequisite_track_blocking():
    engine = RoadmapEngine()
    state = AgentState()

    # Track 2 (track_machine_learning) requires track_data_science.
    # While track_data_science has uncompleted milestones, track_machine_learning cannot be selected.
    ml_track = engine.dag.get_track("track_machine_learning")
    assert ml_track is not None
    assert engine.are_track_prerequisites_met(ml_track, state) is False


def test_all_milestones_completed_returns_none():
    engine = RoadmapEngine()
    state = AgentState()

    # Mark every milestone across all tracks as completed
    for track in engine.dag.tracks:
        for project in track.projects:
            for ms in project.milestones:
                state.milestones[ms.id] = MilestoneProgress(
                    milestone_id=ms.id,
                    project_id=project.id,
                    status=MilestoneStatus.COMPLETED,
                )

    res = engine.get_next_available_milestone(state)
    assert res is None

    summary = engine.get_roadmap_summary(state)
    assert summary["completion_percentage"] == 100.0
