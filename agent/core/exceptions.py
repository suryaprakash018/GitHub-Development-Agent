"""Domain exception hierarchy for the Autonomous Development Agent."""


class AgentError(Exception):
    """Base exception for all agent domain errors."""

    pass


class ConfigurationError(AgentError):
    """Raised when configuration validation or loading fails."""

    pass


class SecurityError(AgentError):
    """Raised when a security boundary or path containment rule is violated."""

    pass


class DecoupledTargetViolationError(SecurityError):
    """Raised when the target repository path attempts to point to or overlap with the agent codebase."""

    pass


class GitSafetyError(AgentError):
    """Raised when Git state is unsafe (e.g., uncommitted manual changes, detached HEAD)."""

    pass


class LLMProviderError(AgentError):
    """Raised when the LLM provider fails, times out, or returns invalid schema data."""

    pass


class ValidationError(AgentError):
    """Raised when generated code fails linting, type checks, or unit tests."""

    pass


class QualityGateError(AgentError):
    """Raised when generated changes do not satisfy quality or utility thresholds."""

    pass


class StateManagerError(AgentError):
    """Raised when state serialization or transaction updates fail."""

    pass
