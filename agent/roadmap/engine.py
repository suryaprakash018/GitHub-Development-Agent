"""Roadmap DAG engine managing milestone dependency resolution and progression."""

from pathlib import Path
from typing import Any

import yaml

from agent.core.exceptions import ConfigurationError
from agent.roadmap.models import MilestoneSpec, ProjectSpec, RoadmapDAG, TrackSpec
from agent.state.models import AgentState, MilestoneStatus
from agent.utils.logger import setup_logger
from agent.utils.security import get_agent_root_path

logger = setup_logger("agent.roadmap")


class RoadmapEngine:
    """Manages roadmap traversal, prerequisite validation, and progression tracking."""

    def __init__(self, roadmap_path: Path | None = None) -> None:
        agent_root = get_agent_root_path()
        self.roadmap_path = roadmap_path or (agent_root / "config" / "roadmap.yaml")
        self.dag = self._load_roadmap()

    def _load_roadmap(self) -> RoadmapDAG:
        """Loads and parses the declarative roadmap YAML."""
        if not self.roadmap_path.exists():
            raise ConfigurationError(f"Roadmap file not found at '{self.roadmap_path}'")

        try:
            with open(self.roadmap_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ConfigurationError("Roadmap YAML content must be a dictionary")
            return RoadmapDAG.model_validate(data)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to parse roadmap DAG at '{self.roadmap_path}': {e}"
            ) from e

    def is_milestone_completed(self, milestone_id: str, state: AgentState) -> bool:
        """Determines if a milestone is marked COMPLETED in persistent state."""
        milestone = state.milestones.get(milestone_id)
        return bool(milestone and milestone.status == MilestoneStatus.COMPLETED)

    def is_project_completed(self, project: ProjectSpec, state: AgentState) -> bool:
        """Checks if all milestones in a given project are completed."""
        if not project.milestones:
            return False
        return all(self.is_milestone_completed(ms.id, state) for ms in project.milestones)

    def is_track_completed(self, track: TrackSpec, state: AgentState) -> bool:
        """Checks if all projects in a given track are completed."""
        if not track.projects:
            return False
        return all(self.is_project_completed(proj, state) for proj in track.projects)

    def are_track_prerequisites_met(self, track: TrackSpec, state: AgentState) -> bool:
        """Verifies that all prerequisite tracks have been fully completed."""
        for prereq_track_id in track.prerequisites:
            prereq_track = self.dag.get_track(prereq_track_id)
            if not prereq_track or not self.is_track_completed(prereq_track, state):
                return False
        return True

    def get_next_available_milestone(
        self, state: AgentState
    ) -> tuple[TrackSpec, ProjectSpec, MilestoneSpec] | None:
        """Identifies the next sequential, uncompleted milestone whose prerequisites are 100% satisfied.

        Never re-selects already completed milestones.
        """
        for track in self.dag.tracks:
            # Check prerequisites
            if not self.are_track_prerequisites_met(track, state):
                logger.debug(f"Track '{track.id}' skipped: Prerequisites not satisfied.")
                continue

            for project in track.projects:
                for milestone in project.milestones:
                    if not self.is_milestone_completed(milestone.id, state):
                        logger.info(
                            f"Selected next milestone: [{track.title}] -> [{project.title}] -> {milestone.title}"
                        )
                        return track, project, milestone

        logger.info("All tracks and milestones in the roadmap DAG are 100% complete.")
        return None

    def get_roadmap_summary(self, state: AgentState) -> dict[str, Any]:
        """Generates an overall completion summary of the roadmap."""
        total_milestones = 0
        completed_milestones = 0

        tracks_summary = []
        for track in self.dag.tracks:
            track_total = 0
            track_completed = 0
            for project in track.projects:
                for milestone in project.milestones:
                    total_milestones += 1
                    track_total += 1
                    if self.is_milestone_completed(milestone.id, state):
                        completed_milestones += 1
                        track_completed += 1

            tracks_summary.append(
                {
                    "track_id": track.id,
                    "title": track.title,
                    "completed": track_completed,
                    "total": track_total,
                    "is_complete": track_completed == track_total and track_total > 0,
                }
            )

        pct = (completed_milestones / total_milestones * 100) if total_milestones > 0 else 0.0

        return {
            "total_milestones": total_milestones,
            "completed_milestones": completed_milestones,
            "completion_percentage": round(pct, 1),
            "tracks": tracks_summary,
        }
