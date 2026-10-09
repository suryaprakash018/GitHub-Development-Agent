"""Safe, typed Git client enforcing decoupled repository boundaries, selective staging, and branch verification."""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from agent.core.config import AppConfig, load_config
from agent.core.exceptions import GitSafetyError
from agent.utils.logger import setup_logger
from agent.utils.security import validate_file_containment, validate_target_repository_path

logger = setup_logger("agent.git.client")


class GitStatus(BaseModel):
    """Structured representation of repository git status."""

    is_clean: bool = True
    current_branch: str | None = None
    staged_files: list[str] = Field(default_factory=list)
    unstaged_files: list[str] = Field(default_factory=list)
    untracked_files: list[str] = Field(default_factory=list)
    raw_porcelain: list[str] = Field(default_factory=list)


class CommitResult(BaseModel):
    """Result of a commit operation or simulation."""

    committed: bool
    dry_run: bool
    commit_hash: str | None = None
    message: str
    error: str | None = None


class PushResult(BaseModel):
    """Result of a push operation or simulation."""

    pushed: bool
    dry_run: bool
    remote: str
    branch: str
    error: str | None = None


class GitRemoteInfo(BaseModel):
    """Information about configured Git remotes."""

    remotes: dict[str, str] = Field(default_factory=dict)
    has_origin: bool = False
    origin_url: str | None = None


class GitClient:
    """Safe, typed Git client operating strictly on the decoupled target repository.

    Guarantees:
    - Never operates on the agent repository itself (enforces anti-self-targeting).
    - Never uses blanket `git add .` (selective staging only).
    - Never overwrites or commits unexpected user changes.
    - Honors DRY_RUN, AUTO_COMMIT, and AUTO_PUSH safety flags.
    - Verifies active branch before any commit operation.
    """

    def __init__(self, target_path: Path | str, config: AppConfig | None = None) -> None:
        # Enforce decoupled path security rule
        self.target_path = validate_target_repository_path(target_path)
        self.config = config or load_config()

    def _run_git(
        self,
        args: list[str],
        timeout: int = 15,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Runs a Git subprocess command confined to the target repository directory."""
        cmd = ["git", *args]
        try:
            return subprocess.run(
                cmd,
                cwd=self.target_path,
                capture_output=True,
                text=True,
                check=check,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as e:
            err_msg = (e.stderr or e.stdout or "").strip()
            raise GitSafetyError(
                f"Git command '{' '.join(cmd)}' failed with exit code {e.returncode}: {err_msg}"
            ) from e
        except FileNotFoundError as e:
            raise GitSafetyError("Git CLI executable not found on system PATH.") from e
        except subprocess.TimeoutExpired as e:
            raise GitSafetyError(
                f"Git command '{' '.join(cmd)}' timed out after {timeout}s."
            ) from e

    def is_git_repository(self) -> bool:
        """Checks if the target path is an initialized Git repository."""
        if not self.target_path.exists():
            return False
        return (self.target_path / ".git").is_dir()

    def get_current_branch(self) -> str | None:
        """Determines the current checked-out branch name."""
        if not self.is_git_repository():
            return None

        try:
            res = self._run_git(["branch", "--show-current"], check=False)
            branch = res.stdout.strip()
            if branch:
                return branch
        except GitSafetyError:
            pass

        # Fallback for older git or detached HEAD
        try:
            res = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"], check=False)
            branch = res.stdout.strip()
            if branch and branch != "HEAD":
                return branch
        except GitSafetyError:
            pass

        return None

    def get_status(self) -> GitStatus:
        """Parses `git status --porcelain` into structured GitStatus."""
        if not self.is_git_repository():
            return GitStatus(is_clean=True, current_branch=None)

        current_branch = self.get_current_branch()
        res = self._run_git(["status", "--porcelain=v1", "-uall"])
        output = res.stdout.strip()

        if not output:
            return GitStatus(is_clean=True, current_branch=current_branch)

        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        raw_lines: list[str] = []

        for line in output.splitlines():
            if not line.strip():
                continue
            raw_lines.append(line)
            if len(line) < 3:
                continue

            index_status = line[0]
            worktree_status = line[1]
            path_part = line[3:].strip()

            # Handle renamed files: "R  orig.py -> new.py"
            if " -> " in path_part:
                path_part = path_part.split(" -> ")[-1].strip()

            # Untracked files
            if line.startswith("??"):
                untracked.append(path_part)
                continue

            # Staged changes (Index)
            if index_status in {"M", "A", "D", "R", "C"}:
                staged.append(path_part)

            # Unstaged changes (Worktree)
            if worktree_status in {"M", "D"}:
                unstaged.append(path_part)

        is_clean = len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0

        return GitStatus(
            is_clean=is_clean,
            current_branch=current_branch,
            staged_files=staged,
            unstaged_files=unstaged,
            untracked_files=untracked,
            raw_porcelain=raw_lines,
        )

    def assert_clean_worktree(self, allowed_files: list[str] | None = None) -> None:
        """Ensures the target repository does not contain unexpected or unrelated uncommitted changes.

        If allowed_files is specified, only those paths are permitted to have modifications or be untracked.
        If any unrelated files are dirty or untracked, raises GitSafetyError to preserve user changes.
        """
        status = self.get_status()
        if status.is_clean:
            return

        allowed_normalized = {Path(p).as_posix().lower() for p in (allowed_files or [])}

        all_dirty = set(status.staged_files + status.unstaged_files + status.untracked_files)
        unexpected: list[str] = []
        for f in all_dirty:
            norm = Path(f).as_posix().lower()
            if norm in allowed_normalized:
                continue
            if f.endswith("/") and any(
                a.startswith(norm.rstrip("/") + "/") for a in allowed_normalized
            ):
                continue
            unexpected.append(f)

        if unexpected:
            raise GitSafetyError(
                f"Safety Guard Activated: Target repository contains unexpected/unrelated uncommitted changes: {unexpected}. "
                f"Aborting Git operation to avoid clobbering or overwriting user work."
            )

    def verify_active_branch(self, expected_branch: str | None = None) -> str:
        """Verifies that the target repository is checked out to the expected branch and not in detached HEAD."""
        expected = expected_branch or self.config.repository.default_branch
        current = self.get_current_branch()

        if not current:
            raise GitSafetyError(
                "Target repository is in detached HEAD state or has no active branch."
            )

        if current != expected:
            raise GitSafetyError(
                f"Branch mismatch: active branch is '{current}', but expected branch is '{expected}'. "
                f"Aborting Git operation to prevent committing to the wrong branch."
            )

        return current

    def get_remote_info(self) -> GitRemoteInfo:
        """Inspects configured remotes in the target repository."""
        if not self.is_git_repository():
            return GitRemoteInfo()

        try:
            res = self._run_git(["remote", "-v"], check=False)
            lines = res.stdout.strip().splitlines()
            remotes: dict[str, str] = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    url = parts[1]
                    remotes[name] = url

            has_origin = "origin" in remotes
            origin_url = remotes.get("origin")
            return GitRemoteInfo(
                remotes=remotes,
                has_origin=has_origin,
                origin_url=origin_url,
            )
        except GitSafetyError:
            return GitRemoteInfo()

    def selective_stage(self, relative_paths: list[str]) -> list[str]:
        """Stages ONLY the explicitly approved files.

        NEVER executes blanket `git add .` or `git add -A`.
        Validates containment boundaries for each path prior to staging.
        """
        if not relative_paths:
            return []

        staged: list[str] = []
        for rel_path in relative_paths:
            # Containment check
            abs_path = validate_file_containment(rel_path, self.target_path)
            if not abs_path.exists():
                raise GitSafetyError(
                    f"Cannot stage non-existent file: '{rel_path}' (resolved: '{abs_path}')"
                )

            # Stage single file explicitly
            posix_path = Path(rel_path).as_posix()
            self._run_git(["add", "--", posix_path])
            staged.append(rel_path)
            logger.debug(f"Selectively staged: '{posix_path}'")

        return staged

    def unstage_all(self) -> None:
        """Unstages all currently staged files without discarding file contents."""
        if not self.is_git_repository():
            return

        with contextlib.suppress(GitSafetyError):
            self._run_git(["restore", "--staged", "."], check=False)
            return

        with contextlib.suppress(GitSafetyError):
            self._run_git(["reset", "HEAD"], check=False)

    def get_diff(self, staged: bool = False, relative_paths: list[str] | None = None) -> str:
        """Retrieves git diff for staged or unstaged changes."""
        if not self.is_git_repository():
            return ""

        args = ["diff"]
        if staged:
            args.append("--cached")
        if relative_paths:
            args.append("--")
            args.extend([Path(p).as_posix() for p in relative_paths])

        try:
            res = self._run_git(args, check=False)
            return res.stdout
        except GitSafetyError:
            return ""

    def get_head_commit_hash(self) -> str | None:
        """Returns the full SHA-1 hash of HEAD, or None if repo has no commits."""
        if not self.is_git_repository():
            return None

        try:
            res = self._run_git(["rev-parse", "HEAD"], check=False)
            sha = res.stdout.strip()
            return sha if len(sha) == 40 else None
        except GitSafetyError:
            return None

    def create_commit(
        self,
        message: str,
        author_name: str | None = None,
        author_email: str | None = None,
    ) -> CommitResult:
        """Creates a Git commit ONLY if DRY_RUN=False and AUTO_COMMIT=True.

        Safety Rules:
        - If DRY_RUN is True or AUTO_COMMIT is False, aborts execution and returns a dry-run result.
        - Verifies that staged files actually exist.
        - Verifies active branch matches configured target branch.
        - NEVER creates an empty commit.
        """
        if not message or not message.strip():
            raise GitSafetyError("Commit message cannot be empty.")

        status = self.get_status()
        if not status.staged_files:
            raise GitSafetyError("No files are staged for commit.")

        # Ensure active branch is valid
        self.verify_active_branch()

        # HARD SAFETY GATE: DRY_RUN or AUTO_COMMIT=False
        if self.config.operational_mode.dry_run or not self.config.operational_mode.auto_commit:
            logger.info(
                f"[DRY_RUN / AUTO_COMMIT=False] Skipping live commit execution. "
                f"Proposed commit message:\n{message}"
            )
            return CommitResult(
                committed=False,
                dry_run=True,
                commit_hash=None,
                message=message,
            )

        # LIVE EXECUTION ONLY WHEN BOTH FLAGS ARE EXPLICITLY PERMISSIVE
        logger.warning(f"Creating live commit on branch '{status.current_branch}'...")
        cmd_args = ["commit", "-m", message]
        if author_name and author_email:
            cmd_args.extend(["--author", f"{author_name} <{author_email}>"])

        self._run_git(cmd_args)
        new_sha = self.get_head_commit_hash()
        logger.info(f"Successfully committed changes (SHA: {new_sha})")

        return CommitResult(
            committed=True,
            dry_run=False,
            commit_hash=new_sha,
            message=message,
        )

    def push(
        self,
        remote: str | None = None,
        branch: str | None = None,
    ) -> PushResult:
        """Pushes committed changes to remote repository ONLY if DRY_RUN=False and AUTO_PUSH=True.

        Safety Rules:
        - If DRY_RUN is True or AUTO_PUSH is False, aborts execution and returns a dry-run result.
        - Verifies remote existence before attempting live push.
        """
        target_remote = remote or self.config.repository.remote_name
        target_branch = branch or self.config.repository.default_branch

        # HARD SAFETY GATE: DRY_RUN or AUTO_PUSH=False
        if self.config.operational_mode.dry_run or not self.config.operational_mode.auto_push:
            logger.info(
                f"[DRY_RUN / AUTO_PUSH=False] Skipping live push execution. "
                f"Target remote: '{target_remote}', Branch: '{target_branch}'"
            )
            return PushResult(
                pushed=False,
                dry_run=True,
                remote=target_remote,
                branch=target_branch,
            )

        # LIVE EXECUTION ONLY WHEN BOTH FLAGS ARE EXPLICITLY PERMISSIVE
        logger.warning(
            f"Executing live git push to remote '{target_remote}', branch '{target_branch}'..."
        )
        self._run_git(["push", target_remote, target_branch])
        logger.info(f"Successfully pushed to {target_remote}/{target_branch}")

        return PushResult(
            pushed=True,
            dry_run=False,
            remote=target_remote,
            branch=target_branch,
        )
