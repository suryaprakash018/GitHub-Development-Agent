"""Windows Task Scheduler manager for unattended autonomous execution."""

from __future__ import annotations

import subprocess
import sys

from pydantic import BaseModel

from agent.core.config import AppConfig, load_config
from agent.utils.logger import setup_logger
from agent.utils.security import get_agent_root_path

logger = setup_logger("agent.scheduler.windows")


class ScheduledTaskStatus(BaseModel):
    """Status details of the registered Windows Scheduled Task."""

    is_registered: bool = False
    task_name: str
    status: str = "NOT_FOUND"
    next_run_time: str | None = None
    last_run_time: str | None = None
    last_result: str | None = None
    author: str | None = None


class WindowsSchedulerManager:
    """Manages creation, inspection, and removal of Windows Task Scheduler tasks."""

    DEFAULT_TASK_NAME = "GitHub-Development-Agent"

    def __init__(
        self,
        config: AppConfig | None = None,
        task_name: str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.task_name = task_name or self.DEFAULT_TASK_NAME
        self.agent_root = get_agent_root_path()

    def get_status(self) -> ScheduledTaskStatus:
        """Queries Windows Task Scheduler to inspect task status."""
        try:
            res = subprocess.run(
                ["schtasks", "/query", "/tn", self.task_name, "/fo", "LIST", "/v"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode != 0:
                return ScheduledTaskStatus(
                    is_registered=False,
                    task_name=self.task_name,
                    status="NOT_REGISTERED",
                )

            status_dict: dict[str, str] = {}
            for line in res.stdout.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    status_dict[key.strip()] = val.strip()

            return ScheduledTaskStatus(
                is_registered=True,
                task_name=self.task_name,
                status=status_dict.get("Status", "READY"),
                next_run_time=status_dict.get("Next Run Time"),
                last_run_time=status_dict.get("Last Run Time"),
                last_result=status_dict.get("Last Result"),
                author=status_dict.get("Author"),
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return ScheduledTaskStatus(
                is_registered=False,
                task_name=self.task_name,
                status="QUERY_ERROR",
            )

    def build_registration_command(
        self,
        preferred_time: str | None = None,
        python_exe: str | None = None,
    ) -> list[str]:
        """Constructs the schtasks /create command arguments with configurable time.

        Does NOT execute the task upon registration.
        """
        run_time = preferred_time or self.config.scheduling.preferred_time or "22:00"
        py_path = python_exe or sys.executable
        # Action executes python with -m agent.cli run --scheduled
        action_cmd = f'"{py_path}" -m agent.cli run --scheduled'

        return [
            "schtasks",
            "/create",
            "/tn",
            self.task_name,
            "/tr",
            action_cmd,
            "/sc",
            "DAILY",
            "/st",
            run_time,
            "/f",  # Force overwrite if already exists
        ]

    def register_task(
        self,
        preferred_time: str | None = None,
        python_exe: str | None = None,
    ) -> tuple[bool, str]:
        """Registers the unattended daily task with Windows Task Scheduler.

        Does NOT start or trigger the task automatically.
        """
        cmd = self.build_registration_command(preferred_time=preferred_time, python_exe=python_exe)
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if res.returncode == 0:
                logger.info(f"Successfully registered Windows Scheduled Task '{self.task_name}'.")
                return True, res.stdout.strip()
            else:
                err_msg = res.stderr.strip() or res.stdout.strip()
                logger.error(f"Failed to register scheduled task: {err_msg}")
                return False, err_msg
        except Exception as e:
            return False, str(e)

    def unregister_task(self) -> tuple[bool, str]:
        """Removes the scheduled task from Windows Task Scheduler."""
        try:
            res = subprocess.run(
                ["schtasks", "/delete", "/tn", self.task_name, "/f"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0:
                logger.info(f"Successfully deleted scheduled task '{self.task_name}'.")
                return True, "Task successfully deleted."
            else:
                return False, res.stderr.strip() or res.stdout.strip()
        except Exception as e:
            return False, str(e)
