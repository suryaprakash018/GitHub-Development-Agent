"""Comprehensive diagnostic health checks for system, Git, target repo, and AI provider."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent.core.config import AppConfig, load_config
from agent.telemetry.recorder import redact_secrets
from agent.utils.logger import setup_logger
from agent.utils.security import (
    get_agent_root_path,
    validate_target_repository_path,
)

logger = setup_logger("agent.validation.health")


class HealthCheckItem(BaseModel):
    """Result of an individual subsystem health check."""

    name: str
    status: str = Field(..., description="'PASS', 'WARN', or 'FAIL'")
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class SystemHealthReport(BaseModel):
    """Consolidated health report across all subsystems."""

    overall_status: str = Field(..., description="'HEALTHY', 'DEGRADED', or 'CRITICAL'")
    all_passed: bool = False
    items: list[HealthCheckItem] = Field(default_factory=list)


class SystemHealthChecker:
    """Executes deterministic and safe connectivity health checks."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.agent_root = get_agent_root_path()

    def check_python_environment(self) -> HealthCheckItem:
        """Verifies Python version and critical dependencies."""
        py_version = sys.version.split()[0]
        if (sys.version_info.major, sys.version_info.minor) < (3, 11):
            return HealthCheckItem(
                name="Python Environment",
                status="FAIL",
                message=f"Python version {py_version} is unsupported (requires >= 3.11)",
            )

        missing_packages = []
        for pkg in ["pydantic", "yaml", "pytest", "ruff"]:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing_packages.append(pkg)

        if missing_packages:
            return HealthCheckItem(
                name="Python Environment",
                status="FAIL",
                message=f"Missing required packages: {', '.join(missing_packages)}",
            )

        return HealthCheckItem(
            name="Python Environment",
            status="PASS",
            message=f"Python {py_version} with all core packages verified",
            details={"python_version": py_version},
        )

    def check_git_cli(self) -> HealthCheckItem:
        """Verifies Git CLI availability and user identity."""
        try:
            res = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            git_version = res.stdout.strip()
        except Exception as e:
            return HealthCheckItem(
                name="Git CLI Availability",
                status="FAIL",
                message=f"Git CLI not found or executable error: {e}",
            )

        # Check git identity
        user_name = ""
        user_email = ""
        try:
            user_name = subprocess.run(
                ["git", "config", "user.name"],
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
            user_email = subprocess.run(
                ["git", "config", "user.email"],
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
        except Exception:
            pass

        has_identity = bool(user_name and user_email)
        status = "PASS" if has_identity else "WARN"
        identity_msg = (
            f"Git identity configured ({user_name} <{user_email}>)"
            if has_identity
            else "Git user.name or user.email not configured"
        )

        return HealthCheckItem(
            name="Git CLI Availability",
            status=status,
            message=f"{git_version}. {identity_msg}",
            details={"version": git_version, "user_name": user_name, "user_email": user_email},
        )

    def check_target_repository(self) -> HealthCheckItem:
        """Validates target repository decoupling, existence, and git status."""
        target_path_str = self.config.repository.target_path
        if not target_path_str:
            return HealthCheckItem(
                name="Target Repository",
                status="WARN",
                message="TARGET_REPOSITORY_PATH is unconfigured in .env / config.yaml",
            )

        target_path = Path(target_path_str).resolve()

        # Decoupled boundary safety check
        try:
            validate_target_repository_path(target_path)
        except Exception as e:
            return HealthCheckItem(
                name="Target Repository",
                status="FAIL",
                message=f"Decoupled security violation: {e}",
            )

        if not target_path.exists() or not target_path.is_dir():
            return HealthCheckItem(
                name="Target Repository",
                status="WARN",
                message=f"Target directory '{target_path}' does not exist on disk yet.",
            )

        # Git repository check
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=target_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode != 0 or res.stdout.strip() != "true":
                return HealthCheckItem(
                    name="Target Repository",
                    status="WARN",
                    message=f"Target directory '{target_path}' is not an initialized Git repository.",
                )
        except Exception as e:
            return HealthCheckItem(
                name="Target Repository",
                status="WARN",
                message=f"Error inspecting target Git repository: {e}",
            )

        # Clean worktree check
        try:
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=target_path,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            if status_res.stdout.strip():
                return HealthCheckItem(
                    name="Target Repository",
                    status="WARN",
                    message=f"Target repository has uncommitted modifications:\n{status_res.stdout.strip()[:200]}",
                )
        except Exception as e:
            return HealthCheckItem(
                name="Target Repository",
                status="WARN",
                message=f"Failed to check worktree status: {e}",
            )

        # Active branch check
        try:
            branch_res = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=target_path,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            branch = branch_res.stdout.strip()
            expected = self.config.repository.default_branch
            if branch != expected:
                return HealthCheckItem(
                    name="Target Repository",
                    status="WARN",
                    message=f"Active branch is '{branch}' (expected '{expected}')",
                )
        except Exception:
            branch = "unknown"

        return HealthCheckItem(
            name="Target Repository",
            status="PASS",
            message=f"Verified clean decoupled Git repository at '{target_path}' (branch: '{branch}')",
            details={"path": str(target_path), "branch": branch},
        )

    def check_git_remote_accessibility(self) -> HealthCheckItem:
        """Verifies remote origin reachability without mutating state."""
        target_path_str = self.config.repository.target_path
        if not target_path_str:
            return HealthCheckItem(
                name="Git Remote Accessibility",
                status="WARN",
                message="Skipped: Target repository path not configured",
            )

        target_path = Path(target_path_str).resolve()
        if not target_path.exists() or not (target_path / ".git").exists():
            return HealthCheckItem(
                name="Git Remote Accessibility",
                status="WARN",
                message="Skipped: Target is not an initialized Git repository",
            )

        remote_name = self.config.repository.remote_name
        try:
            url_res = subprocess.run(
                ["git", "remote", "get-url", remote_name],
                cwd=target_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if url_res.returncode != 0:
                return HealthCheckItem(
                    name="Git Remote Accessibility",
                    status="WARN",
                    message=f"No Git remote named '{remote_name}' configured in target repository",
                )

            raw_url = url_res.stdout.strip()
            sanitized_url = redact_secrets(raw_url)

            # Test reachability with git ls-remote (5s timeout)
            ls_res = subprocess.run(
                ["git", "ls-remote", "--heads", remote_name],
                cwd=target_path,
                capture_output=True,
                text=True,
                timeout=8,
            )
            if ls_res.returncode == 0:
                return HealthCheckItem(
                    name="Git Remote Accessibility",
                    status="PASS",
                    message=f"Remote '{remote_name}' ({sanitized_url}) is reachable and accessible",
                    details={"remote_url": sanitized_url},
                )
            else:
                err_msg = redact_secrets(ls_res.stderr.strip() or "Remote connection refused")
                return HealthCheckItem(
                    name="Git Remote Accessibility",
                    status="WARN",
                    message=f"Remote '{remote_name}' ({sanitized_url}) unreachable or requires credentials: {err_msg[:120]}",
                )
        except subprocess.TimeoutExpired:
            return HealthCheckItem(
                name="Git Remote Accessibility",
                status="WARN",
                message=f"Connection to remote '{remote_name}' timed out after 8s (offline or slow network)",
            )
        except Exception as e:
            return HealthCheckItem(
                name="Git Remote Accessibility",
                status="WARN",
                message=f"Remote check error: {redact_secrets(str(e))}",
            )

    def check_gemini_api_connectivity(self) -> HealthCheckItem:
        """Safely verifies Gemini API key presence and connectivity with secret redaction."""
        api_key = self.config.ai.gemini_api_key
        if not api_key:
            return HealthCheckItem(
                name="Google Gemini API",
                status="WARN",
                message="GEMINI_API_KEY is not configured in .env. LLM operations will be unavailable.",
            )

        # Mask key display: show first 6 chars, redact rest
        masked_key = f"{api_key[:6]}...[REDACTED]" if len(api_key) > 8 else "[REDACTED]"

        try:
            # Lightweight verification using google-genai Client
            from google import genai

            client = genai.Client(api_key=api_key)
            # Verify access to the configured primary model
            target_model = self.config.ai.primary_model
            try:
                model_obj = client.models.get(model=target_model)
                model_info = model_obj.name
            except Exception:
                pager = client.models.list(config={"page_size": 1})
                first_model = next(iter(pager), None)
                model_info = first_model.name if first_model else target_model

            return HealthCheckItem(
                name="Google Gemini API",
                status="PASS",
                message=f"Gemini API connectivity verified successfully using key {masked_key} ({model_info})",
                details={"masked_key": masked_key, "model": model_info},
            )
        except Exception as e:
            err_msg = redact_secrets(str(e), custom_secrets=[api_key])
            return HealthCheckItem(
                name="Google Gemini API",
                status="WARN",
                message=f"Gemini API check warning using key {masked_key}: {err_msg[:150]}",
                details={"masked_key": masked_key, "error": err_msg},
            )

    def check_groq_api_connectivity(self) -> HealthCheckItem:
        """Safely verifies Groq API key presence and live model connectivity with secret redaction."""
        api_key = self.config.ai.groq_api_key
        if not api_key or not api_key.strip():
            return HealthCheckItem(
                name="Groq Cloud API",
                status="WARN",
                message="GROQ_API_KEY is not configured in .env. LLM operations will be unavailable.",
            )

        target_model = self.config.ai.groq_model
        if not target_model or not target_model.strip():
            return HealthCheckItem(
                name="Groq Cloud API",
                status="WARN",
                message="GROQ_MODEL is not configured in .env or config. Please set an available Groq model.",
            )

        try:
            from groq import (
                APIConnectionError,
                APITimeoutError,
                AuthenticationError,
                Groq,
                NotFoundError,
                PermissionDeniedError,
                RateLimitError,
            )

            client = Groq(api_key=api_key, max_retries=0, timeout=10.0)
            client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return HealthCheckItem(
                name="Groq Cloud API",
                status="PASS",
                message=f"Groq API connectivity verified successfully ({target_model})",
                details={"model": target_model},
            )
        except AuthenticationError as e:
            err_msg = redact_secrets(str(e), custom_secrets=[api_key])
            return HealthCheckItem(
                name="Groq Cloud API",
                status="FAIL",
                message=f"Groq authentication failed (invalid or expired API key): {err_msg[:120]}",
                details={"error": err_msg, "model": target_model},
            )
        except NotFoundError as e:
            err_msg = redact_secrets(str(e), custom_secrets=[api_key])
            return HealthCheckItem(
                name="Groq Cloud API",
                status="FAIL",
                message=f"Groq model '{target_model}' not found or unavailable: {err_msg[:120]}",
                details={"error": err_msg, "model": target_model},
            )
        except PermissionDeniedError as e:
            err_msg = redact_secrets(str(e), custom_secrets=[api_key])
            return HealthCheckItem(
                name="Groq Cloud API",
                status="FAIL",
                message=f"Groq permission denied for model '{target_model}': {err_msg[:120]}",
                details={"error": err_msg, "model": target_model},
            )
        except RateLimitError as e:
            err_msg = redact_secrets(str(e), custom_secrets=[api_key])
            return HealthCheckItem(
                name="Groq Cloud API",
                status="WARN",
                message=f"Groq rate limit or quota exceeded: {err_msg[:120]}",
                details={"error": err_msg, "model": target_model},
            )
        except (APIConnectionError, APITimeoutError) as e:
            err_msg = redact_secrets(str(e), custom_secrets=[api_key])
            return HealthCheckItem(
                name="Groq Cloud API",
                status="WARN",
                message=f"Groq API connection or timeout error: {err_msg[:120]}",
                details={"error": err_msg, "model": target_model},
            )
        except Exception as e:
            err_msg = redact_secrets(str(e), custom_secrets=[api_key])
            return HealthCheckItem(
                name="Groq Cloud API",
                status="WARN",
                message=f"Groq API check error: {err_msg[:120]}",
                details={"error": err_msg, "model": target_model},
            )

    def run_all(self) -> SystemHealthReport:
        """Executes all system health checks and compiles consolidated report."""
        llm_check = (
            self.check_groq_api_connectivity()
            if self.config.ai.provider.lower() == "groq"
            else self.check_gemini_api_connectivity()
        )
        checks = [
            self.check_python_environment(),
            self.check_git_cli(),
            self.check_target_repository(),
            self.check_git_remote_accessibility(),
            llm_check,
        ]

        has_fail = any(c.status == "FAIL" for c in checks)
        has_warn = any(c.status == "WARN" for c in checks)

        if has_fail:
            overall = "CRITICAL"
        elif has_warn:
            overall = "DEGRADED"
        else:
            overall = "HEALTHY"

        all_passed = not has_fail

        return SystemHealthReport(
            overall_status=overall,
            all_passed=all_passed,
            items=checks,
        )
