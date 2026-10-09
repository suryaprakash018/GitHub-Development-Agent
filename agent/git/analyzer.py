"""Read-only target repository analyzer and AST inspector."""

import ast
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from agent.core.exceptions import GitSafetyError
from agent.utils.logger import setup_logger
from agent.utils.security import validate_target_repository_path

logger = setup_logger("agent.git.analyzer")


class TargetRepoAnalysis(BaseModel):
    """Snapshot of target repository structure, files, git state, and AST symbols."""

    target_path: str
    is_git_repo: bool = False
    is_clean: bool = True
    uncommitted_files: list[str] = Field(default_factory=list)
    current_branch: str | None = None
    recent_commits: list[str] = Field(default_factory=list)
    existing_files: list[str] = Field(default_factory=list)
    python_modules: list[str] = Field(default_factory=list)
    test_files: list[str] = Field(default_factory=list)
    has_readme: bool = False
    has_dependency_spec: bool = False
    module_symbols: dict[str, list[str]] = Field(default_factory=dict)


class TargetRepoAnalyzer:
    """Strictly READ-ONLY analyzer for the external target repository.

    Never modifies files, creates commits, or alters git state.
    """

    def __init__(self, target_path: Path | str) -> None:
        # Enforce decoupled path safety check
        self.target_path = validate_target_repository_path(target_path)

    def analyze(self) -> TargetRepoAnalysis:
        """Performs a comprehensive read-only scan of the target repository."""
        logger.debug(f"Starting read-only analysis of target repository at '{self.target_path}'")

        if not self.target_path.exists():
            return TargetRepoAnalysis(
                target_path=str(self.target_path),
                is_git_repo=False,
                is_clean=True,
            )

        is_git = (self.target_path / ".git").is_dir()
        is_clean, uncommitted = self._check_git_status() if is_git else (True, [])
        branch = self._get_current_branch() if is_git else None
        commits = self._get_recent_commits() if is_git else []

        existing_files, py_modules, test_files = self._scan_files()
        has_readme = (self.target_path / "README.md").is_file()
        has_dep = (self.target_path / "pyproject.toml").is_file() or (
            self.target_path / "requirements.txt"
        ).is_file()

        symbols = self._extract_symbols(py_modules)

        return TargetRepoAnalysis(
            target_path=str(self.target_path),
            is_git_repo=is_git,
            is_clean=is_clean,
            uncommitted_files=uncommitted,
            current_branch=branch,
            recent_commits=commits,
            existing_files=existing_files,
            python_modules=py_modules,
            test_files=test_files,
            has_readme=has_readme,
            has_dependency_spec=has_dep,
            module_symbols=symbols,
        )

    def assert_safe_to_operate(self) -> None:
        """Verifies that the target repository exists, is a git repo, and has no dirty uncommitted user changes.

        Raises GitSafetyError if unsafe.
        """
        if not self.target_path.exists():
            return  # Brand new repository path is safe to initialize

        if not (self.target_path / ".git").is_dir():
            raise GitSafetyError(
                f"Target repository path '{self.target_path}' exists but is not an initialized Git repository."
            )

        is_clean, uncommitted = self._check_git_status()
        if not is_clean:
            raise GitSafetyError(
                f"Safety Guard Activated: Target repository has uncommitted user modifications: {uncommitted}. "
                f"Halting execution to avoid overwriting or clobbering uncommitted work."
            )

    def _check_git_status(self) -> tuple[bool, list[str]]:
        """Read-only check of `git status --porcelain` to detect uncommitted changes."""
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.target_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            output = res.stdout.strip()
            if not output:
                return True, []
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            return False, lines
        except (subprocess.SubprocessError, FileNotFoundError):
            return True, []

    def _get_current_branch(self) -> str | None:
        """Read-only check of active Git branch."""
        try:
            res = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.target_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return res.stdout.strip() or None
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    def _get_recent_commits(self, limit: int = 5) -> list[str]:
        """Read-only check of recent git commit history."""
        try:
            res = subprocess.run(
                ["git", "log", f"-n{limit}", "--oneline"],
                cwd=self.target_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        except (subprocess.SubprocessError, FileNotFoundError):
            return []

    def _scan_files(self) -> tuple[list[str], list[str], list[str]]:
        """Scans all non-ignored files within the target repository."""
        all_files: list[str] = []
        py_modules: list[str] = []
        test_files: list[str] = []

        ignore_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "venv",
            "env",
            "build",
            "dist",
        }

        for path in self.target_path.rglob("*"):
            if path.is_file():
                # Check if any parent part is in ignore_dirs
                parts = set(path.relative_to(self.target_path).parts)
                if parts & ignore_dirs:
                    continue

                rel_str = path.relative_to(self.target_path).as_posix()
                all_files.append(rel_str)

                if path.suffix == ".py":
                    py_modules.append(rel_str)
                    if path.name.startswith("test_") or path.name.endswith("_test.py"):
                        test_files.append(rel_str)

        return sorted(all_files), sorted(py_modules), sorted(test_files)

    def _extract_symbols(self, py_modules: list[str]) -> dict[str, list[str]]:
        """Parses Python ASTs in read-only mode to extract declared functions and classes."""
        symbols: dict[str, list[str]] = {}

        for rel_mod in py_modules:
            abs_path = self.target_path / rel_mod
            if not abs_path.is_file():
                continue

            try:
                with open(abs_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(abs_path))

                declared: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        declared.append(node.name)

                symbols[rel_mod] = sorted(declared)
            except (SyntaxError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to parse AST for '{rel_mod}': {e}")

        return symbols
