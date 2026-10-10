"""Configuration models, YAML loader, and environment variable bindings."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.core.exceptions import ConfigurationError
from agent.utils.security import get_agent_root_path, validate_target_repository_path


class SystemConfig(BaseModel):
    name: str = "Autonomous AI + Data Development Agent"
    version: str = "0.1.0"
    niche: str = "AI + Data Engineering / Intelligent Software Development"


class OperationalModeConfig(BaseModel):
    dry_run: bool = True
    auto_commit: bool = False
    auto_push: bool = False
    require_tests: bool = True
    max_retries: int = 5


class AIConfig(BaseModel):
    provider: str = "gemini"
    primary_model: str = "gemini-3.8-flash"
    reasoning_model: str | None = None
    thinking_level: str | None = None
    max_tokens: int = 8192
    request_timeout_seconds: int = 60
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    groq_model: str | None = None


class RepositoryConfig(BaseModel):
    target_path: str = ""
    default_branch: str = "main"
    remote_name: str = "origin"


class QualityGateConfig(BaseModel):
    minimum_test_coverage_pct: float = 80.0
    minimum_lines_changed: int = 15
    maximum_lines_changed: int = 500
    maximum_files_changed: int = 10
    enforce_clean_lint: bool = True
    enforce_type_hints: bool = True
    enforce_semantic_utility: bool = True


class SchedulingConfig(BaseModel):
    frequency: str = "daily"
    preferred_time: str = "22:00"
    timezone: str = "local"


class AppConfig(BaseSettings):
    """Master application configuration supporting YAML file defaults and environment variable overrides."""

    system: SystemConfig = Field(default_factory=SystemConfig)
    operational_mode: OperationalModeConfig = Field(default_factory=OperationalModeConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    repository: RepositoryConfig = Field(default_factory=RepositoryConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)

    # Direct environment variable bindings
    ai_provider: str | None = Field(default=None, alias="AI_PROVIDER")
    target_repository_path: str | None = Field(default=None, alias="TARGET_REPOSITORY_PATH")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str | None = Field(default=None, alias="GEMINI_MODEL")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str | None = Field(default=None, alias="GROQ_MODEL")
    dry_run: bool | None = Field(default=None, alias="DRY_RUN")
    auto_commit: bool | None = Field(default=None, alias="AUTO_COMMIT")
    auto_push: bool | None = Field(default=None, alias="AUTO_PUSH")
    target_branch: str | None = Field(default=None, alias="TARGET_BRANCH")
    thinking_level: str | None = Field(default=None, alias="THINKING_LEVEL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context: Any) -> None:
        """Apply environment variable overrides onto nested configuration objects."""
        if self.ai_provider:
            self.ai.provider = self.ai_provider.strip().lower()
        if self.target_repository_path:
            self.repository.target_path = self.target_repository_path
        if self.gemini_api_key:
            self.ai.gemini_api_key = self.gemini_api_key
        if self.gemini_model:
            self.ai.primary_model = self.gemini_model
        if self.groq_api_key:
            self.ai.groq_api_key = self.groq_api_key
        if self.groq_model:
            self.ai.groq_model = self.groq_model
        if self.thinking_level:
            self.ai.thinking_level = self.thinking_level
        if self.dry_run is not None:
            self.operational_mode.dry_run = self.dry_run
        if self.auto_commit is not None:
            self.operational_mode.auto_commit = self.auto_commit
        if self.auto_push is not None:
            self.operational_mode.auto_push = self.auto_push
        if self.target_branch:
            self.repository.default_branch = self.target_branch

    def is_llm_configured(self) -> bool:
        """Checks whether the currently active LLM provider has its API key configured."""
        provider = self.ai.provider.lower()
        if provider == "gemini":
            return bool(self.ai.gemini_api_key and self.ai.gemini_api_key.strip())
        elif provider == "groq":
            return bool(self.ai.groq_api_key and self.ai.groq_api_key.strip())
        return False

    def get_validated_target_path(self) -> Path:
        """Returns the canonical target repository path after verifying decoupled security rules."""
        target_path_str = self.repository.target_path
        if not target_path_str:
            raise ConfigurationError(
                "Target repository path is not configured. Specify TARGET_REPOSITORY_PATH in .env "
                "or repository.target_path in config.yaml."
            )
        return validate_target_repository_path(target_path_str)


def load_config(config_yaml_path: Path | None = None) -> AppConfig:
    """Loads configuration from YAML file (if present) merged with environment variables."""
    agent_root = get_agent_root_path()
    yaml_file = config_yaml_path or (agent_root / "config" / "config.yaml")

    data: dict[str, Any] = {}
    if yaml_file.exists():
        try:
            with open(yaml_file, encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if isinstance(content, dict):
                    data = content
        except Exception as e:
            raise ConfigurationError(
                f"Failed to parse configuration YAML at '{yaml_file}': {e}"
            ) from e

    return AppConfig(**data)
