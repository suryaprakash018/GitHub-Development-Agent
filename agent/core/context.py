"""Execution context and lockfile concurrency manager."""

import json
import os
import time
from pathlib import Path
from typing import Any

from agent.core.exceptions import AgentError
from agent.utils.logger import setup_logger
from agent.utils.security import get_agent_root_path

logger = setup_logger("agent.context")


def is_pid_running(pid: int) -> bool:
    """Checks whether a given process ID is currently running on the operating system."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            process = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if process != 0:
                kernel32.CloseHandle(process)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except OSError:
        return False


class LockError(AgentError):
    """Raised when an active execution lock prevents running the agent."""

    pass


class ExecutionLock:
    """Manages an atomic lockfile (.agent.lock) to prevent concurrent execution runs."""

    def __init__(self, lockfile_path: Path | None = None) -> None:
        agent_root = get_agent_root_path()
        self.lockfile_path = lockfile_path or (agent_root / ".agent.lock")
        self._acquired = False

    def acquire(self) -> None:
        """Acquires the execution lock. If a stale lock exists, recovers gracefully."""
        if self.lockfile_path.exists():
            try:
                with open(self.lockfile_path, encoding="utf-8") as f:
                    data = json.load(f)
                locked_pid = data.get("pid")
                if locked_pid and isinstance(locked_pid, int):
                    if is_pid_running(locked_pid):
                        raise LockError(
                            f"Another instance of GitHub-Development-Agent is already running (PID: {locked_pid}). "
                            f"Lock held since {data.get('timestamp')}."
                        )
                    else:
                        logger.warning(
                            f"Detected stale lock from inactive process PID {locked_pid}. Overriding lock."
                        )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read existing lockfile: {e}. Overriding corrupt lock.")

        # Atomic write of current PID
        lock_data = {
            "pid": os.getpid(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.lockfile_path, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)

        self._acquired = True
        logger.debug(f"Acquired execution lock at '{self.lockfile_path}'.")

    def release(self) -> None:
        """Releases the execution lock."""
        if self.lockfile_path.exists() and self._acquired:
            try:
                self.lockfile_path.unlink()
                self._acquired = False
                logger.debug("Released execution lock.")
            except OSError as e:
                logger.warning(f"Error releasing lockfile: {e}")

    def __enter__(self) -> "ExecutionLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
