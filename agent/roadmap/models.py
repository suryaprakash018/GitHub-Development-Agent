"""Data models for Roadmap DAG tracks, projects, and milestones."""

from pydantic import BaseModel, Field


class MilestoneSpec(BaseModel):
    """Specification of a discrete milestone within a project."""

    id: str
    title: str
    description: str
    required_tests: list[str] = Field(default_factory=list)


class ProjectSpec(BaseModel):
    """Specification of a portfolio project consisting of sequential milestones."""

    id: str
    title: str
    description: str
    milestones: list[MilestoneSpec] = Field(default_factory=list)

    def get_milestone(self, milestone_id: str) -> MilestoneSpec | None:
        """Finds a milestone by ID."""
        for ms in self.milestones:
            if ms.id == milestone_id:
                return ms
        return None


class TrackSpec(BaseModel):
    """Specification of a domain track in the AI + Data Engineering roadmap."""

    id: str
    title: str
    difficulty: str
    prerequisites: list[str] = Field(default_factory=list)
    projects: list[ProjectSpec] = Field(default_factory=list)

    def get_project(self, project_id: str) -> ProjectSpec | None:
        """Finds a project by ID."""
        for proj in self.projects:
            if proj.id == project_id:
                return proj
        return None


class RoadmapDAG(BaseModel):
    """Master Directed Acyclic Graph containing all roadmap tracks and projects."""

    version: str
    description: str
    tracks: list[TrackSpec] = Field(default_factory=list)

    def get_track(self, track_id: str) -> TrackSpec | None:
        """Finds a track by ID."""
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None
