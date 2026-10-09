"""Security utilities and containment validators for decoupled repository execution."""

from pathlib import Path

from agent.core.exceptions import DecoupledTargetViolationError, SecurityError


def get_agent_root_path() -> Path:
    """Returns the canonical root directory of the GitHub-Development-Agent repository."""
    # This file is located at <agent_root>/agent/utils/security.py
    return Path(__file__).resolve().parent.parent.parent


def validate_target_repository_path(target_path: str | Path | None) -> Path:
    """Validates that the target repository path is safe, absolute, and completely decoupled from

    the agent repository.

    Raises:
        DecoupledTargetViolationError: If target points to or is inside the agent repository.
        SecurityError: If target path is invalid or empty.
    """
    if not target_path or str(target_path).strip() == "":
        raise SecurityError(
            "TARGET_REPOSITORY_PATH is not configured. A separate target repository must be specified."
        )

    resolved_target = Path(target_path).resolve()
    resolved_agent = get_agent_root_path().resolve()

    # Rule 1: The target cannot be identical to the agent codebase
    if resolved_target == resolved_agent:
        raise DecoupledTargetViolationError(
            f"Security Violation: Target repository path '{resolved_target}' cannot be "
            f"the GitHub-Development-Agent repository itself. The agent must operate on a decoupled repository."
        )

    # Rule 2: The target cannot be inside the agent directory tree
    try:
        resolved_target.relative_to(resolved_agent)
        # If relative_to succeeds, resolved_target is a child of resolved_agent
        raise DecoupledTargetViolationError(
            f"Security Violation: Target repository path '{resolved_target}' is located inside the "
            f"GitHub-Development-Agent repository ('{resolved_agent}'). It must be completely outside."
        )
    except ValueError:
        # Not a child of agent, which is required
        pass

    # Rule 3: The agent codebase cannot be inside the target directory tree
    try:
        resolved_agent.relative_to(resolved_target)
        raise DecoupledTargetViolationError(
            f"Security Violation: GitHub-Development-Agent ('{resolved_agent}') is located inside the "
            f"target repository ('{resolved_target}'). The repositories must be completely decoupled."
        )
    except ValueError:
        # Not a parent of agent, which is required
        pass

    return resolved_target


def validate_file_containment(file_relative_path: str | Path, base_dir: Path) -> Path:
    """Ensures that a relative file path is strictly contained within base_dir and does not escape

    via directory traversal (e.g. '../').

    Raises:
        SecurityError: If the target file escapes base_dir.
    """
    resolved_base = base_dir.resolve()
    candidate_path = (resolved_base / file_relative_path).resolve()

    try:
        candidate_path.relative_to(resolved_base)
    except ValueError:
        raise SecurityError(
            f"Path Traversal Violation: Path '{file_relative_path}' resolves outside the allowed base directory '{resolved_base}'."
        ) from None

    return candidate_path
