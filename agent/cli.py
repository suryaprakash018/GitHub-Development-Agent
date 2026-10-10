"""Command-Line Interface (CLI) for Autonomous AI + Data Development Agent."""

import argparse
import subprocess
import sys

from agent.core.config import load_config
from agent.core.exceptions import (
    ConfigurationError,
    DecoupledTargetViolationError,
    SecurityError,
)
from agent.git.analyzer import TargetRepoAnalyzer
from agent.roadmap.engine import RoadmapEngine
from agent.selection.selector import TaskSelector
from agent.state.manager import StateManager
from agent.utils.logger import setup_logger
from agent.validation.health import SystemHealthChecker

logger = setup_logger("agent.cli")


def cmd_verify(args: argparse.Namespace) -> int:
    """Deterministically verifies system environment, API keys, Git CLI, and decoupled repository status."""
    print("=" * 70)
    print(" SYSTEM ENVIRONMENT & DECOUPLED SAFETY VERIFICATION")
    print("=" * 70)

    all_passed = True

    # Check 1: Python Version
    py_version = sys.version.split()[0]
    if (sys.version_info.major, sys.version_info.minor) >= (3, 11):
        print(f" [PASS] Python Version: {py_version} (>= 3.11 requirement satisfied)")
    else:
        print(f" [FAIL] Python Version: {py_version} (Requires Python 3.11 or newer)")
        all_passed = False

    # Check 2: Git CLI
    try:
        git_res = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        print(f" [PASS] Git CLI: {git_res.stdout.strip()}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print(" [FAIL] Git CLI: Not installed or not available on system PATH")
        all_passed = False

    # Check 3: Configuration & Target Repository Decoupled Security
    try:
        config = load_config()
        print(" [PASS] Configuration: Successfully loaded config.yaml and environment")
    except Exception as e:
        print(f" [FAIL] Configuration Error: {e}")
        return 1

    try:
        target_path = config.get_validated_target_path()
        print(f" [PASS] Decoupled Target Repository Path: '{target_path}'")

        if not target_path.exists():
            print(" [WARN] Target directory does not exist yet. It will be created when needed.")
        elif not (target_path / ".git").is_dir():
            print(" [WARN] Target path exists but is not yet initialized as a Git repository.")
        else:
            print(" [PASS] Target directory is an initialized Git repository.")

    except DecoupledTargetViolationError as e:
        print(f" [FAIL] Decoupled Safety Violation: {e}")
        all_passed = False
    except (ConfigurationError, SecurityError) as e:
        print(f" [WARN] Target Repository: {e}")

    # Check 4: Git Remote Accessibility
    checker = SystemHealthChecker(config=config)
    remote_res = checker.check_git_remote_accessibility()
    print(f" [{remote_res.status}] Git Remote Accessibility: {remote_res.message}")

    # Check 5: Active AI Provider Key & Connectivity
    active_provider = config.ai.provider.lower()
    if active_provider == "groq":
        groq_key = config.ai.groq_api_key
        if groq_key and groq_key.strip() != "":
            print(" [PASS] Groq API Key: Configured")
            conn_res = checker.check_groq_api_connectivity()
            print(f" [{conn_res.status}] Groq API Connectivity: {conn_res.message}")
        else:
            print(" [WARN] Groq API Key: Not set in .env or environment (GROQ_API_KEY)")
    else:
        gemini_key = config.ai.gemini_api_key
        if gemini_key and gemini_key.strip() != "":
            print(" [PASS] Gemini API Key: Configured")
            conn_res = checker.check_gemini_api_connectivity()
            print(f" [{conn_res.status}] Gemini API Connectivity: {conn_res.message}")
        else:
            print(" [WARN] Gemini API Key: Not set in .env or environment (GEMINI_API_KEY)")

    # Check 6: Operational Safety Flags
    print("-" * 70)
    print(f" Operational Safety: DRY_RUN={config.operational_mode.dry_run}")
    print(f" Operational Safety: AUTO_COMMIT={config.operational_mode.auto_commit}")
    print(f" Operational Safety: AUTO_PUSH={config.operational_mode.auto_push}")
    print("=" * 70)

    if all_passed:
        print("RESULT: Core environment and decoupled safety checks PASSED.")
        return 0
    else:
        print("RESULT: Critical environment or safety checks FAILED.")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Displays current roadmap progress, portfolio metrics, and system state."""
    config = load_config()
    state_manager = StateManager()
    state = state_manager.load_state()
    roadmap_engine = RoadmapEngine()
    summary = roadmap_engine.get_roadmap_summary(state)

    print("=" * 70)
    print(" AUTONOMOUS DEVELOPMENT AGENT STATUS")
    print("=" * 70)
    print(f" Last Run:              {state.last_run_timestamp or 'Never executed'}")
    print(
        f" Roadmap Progress:      {summary['completed_milestones']}/{summary['total_milestones']} milestones ({summary['completion_percentage']}%)"
    )
    print(f" Total Tasks Completed: {state.metrics.total_tasks_completed}")
    print(f" Total Commits Created: {state.metrics.total_commits_created}")
    print(f" Total Tests Written:   {state.metrics.total_tests_written}")
    print("-" * 70)
    print(f" Target Repository:     {config.repository.target_path or '[Not Configured]'}")
    print(f" Target Branch:         {config.repository.default_branch}")
    print(f" Operational Mode:      DRY_RUN={config.operational_mode.dry_run}")
    print("=" * 70)

    # If target repo configured, show next planned task
    if config.repository.target_path:
        try:
            target_path = config.get_validated_target_path()
            analyzer = TargetRepoAnalyzer(target_path=target_path)
            analysis = analyzer.analyze()
            selector = TaskSelector(roadmap_engine=roadmap_engine)
            next_task = selector.select_next_task(analysis, state)
            if next_task:
                print(f" Next Up: [{next_task.task_id}] {next_task.title}")
                print(f" Type:    {next_task.task_type}")
                print(f" Value:   Portfolio score {next_task.portfolio_value_score}")
                print("=" * 70)
        except Exception:
            pass

    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Inspects the target repository and displays the exact next task selected by the TaskSelector."""
    config = load_config()
    state_manager = StateManager()
    state = state_manager.load_state()

    try:
        target_path = config.get_validated_target_path()
    except Exception as e:
        print(f"Cannot generate plan: {e}")
        return 1

    analyzer = TargetRepoAnalyzer(target_path=target_path)
    analysis = analyzer.analyze()
    selector = TaskSelector()
    candidate = selector.select_next_task(analysis, state)

    print("=" * 70)
    print(" PROPOSED DEVELOPMENT TASK PLAN")
    print("=" * 70)
    if not candidate:
        print("No task available: All roadmap milestones are complete or blocked.")
        return 0

    print(f" Task ID:         {candidate.task_id}")
    print(f" Track / Project: {candidate.track_id} -> {candidate.project_id}")
    print(f" Milestone:       {candidate.milestone_id}")
    print(f" Title:           {candidate.title}")
    print(f" Type:            {candidate.task_type}")
    print(f" Value Score:     {candidate.portfolio_value_score}")
    print(f" Target Files:    {candidate.target_files}")
    print(f" Test Files:      {candidate.test_files}")
    print(f" Rationale:       {candidate.rationale}")
    print("=" * 70)
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    """Manages the Windows Task Scheduler registration for unattended daily runs."""
    from agent.scheduler.windows import WindowsSchedulerManager

    mgr = WindowsSchedulerManager()
    if args.action == "status":
        status = mgr.get_status()
        print("=" * 70)
        print(" WINDOWS TASK SCHEDULER STATUS")
        print("=" * 70)
        print(f" Task Name:      {status.task_name}")
        print(f" Registered:     {status.is_registered}")
        print(f" Status:         {status.status}")
        print(f" Next Run Time:  {status.next_run_time or 'N/A'}")
        print(f" Last Run Time:  {status.last_run_time or 'Never'}")
        print(f" Last Result:    {status.last_result or 'N/A'}")
        print("=" * 70)
        return 0
    elif args.action == "register":
        time_to_set = getattr(args, "time", None) or "22:00"
        success, msg = mgr.register_task(preferred_time=time_to_set)
        if success:
            print(f"[SUCCESS] Registered Windows task '{mgr.task_name}' daily at {time_to_set}.")
            print(
                "Note: The task has NOT been executed yet. It will trigger at the scheduled time."
            )
            return 0
        else:
            print(f"[FAIL] Could not register task: {msg}")
            return 1
    elif args.action == "unregister":
        success, msg = mgr.unregister_task()
        if success:
            print(f"[SUCCESS] Unregistered task '{mgr.task_name}'.")
            return 0
        else:
            print(f"[FAIL] Could not unregister task: {msg}")
            return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Runs the autonomous development cycle using AutonomousDevelopmentOrchestrator."""
    config = load_config()
    if args.dry_run:
        config.operational_mode.dry_run = True

    from agent.orchestrator.pipeline import AutonomousDevelopmentOrchestrator

    orchestrator = AutonomousDevelopmentOrchestrator(config=config)
    result = orchestrator.run(scheduled=args.scheduled, force_dry_run=args.dry_run)

    print("\n" + "=" * 70)
    print(f" EXECUTION CYCLE RESULT: {result.status}")
    print("=" * 70)
    if result.candidate:
        print(f" Task:           [{result.candidate.task_id}] {result.candidate.title}")
    print(f" Explanation:    {result.explanation}")
    print(f" Quality Gate:   {'PASSED' if result.quality_gate_passed else 'FAILED/SKIPPED'}")
    print(f" Committed:      {result.committed}")
    print(f" Pushed:         {result.pushed}")
    if result.telemetry_file:
        print(f" Telemetry Audit: {result.telemetry_file}")
    if result.commit_message:
        print("-" * 70)
        print(" PROPOSED CONVENTIONAL COMMIT (Simulated in DRY_RUN, not committed):")
        print(result.commit_message)
    print("=" * 70 + "\n")

    return (
        0
        if result.status in {"DRY_RUN_APPROVED", "SUCCESS", "NO_TASK_AVAILABLE", "AI_UNAVAILABLE"}
        else 1
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dev-agent",
        description="Autonomous AI + Data Development Agent for GitHub",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # verify
    subparsers.add_parser("verify", help="Verify environment, Git, API keys, and safety boundaries")

    # status
    subparsers.add_parser("status", help="Show roadmap progress and system state")

    # plan
    subparsers.add_parser("plan", help="Inspect target repository and show proposed task plan")

    # schedule
    sched_parser = subparsers.add_parser(
        "schedule", help="Manage Windows Task Scheduler unattended execution"
    )
    sched_parser.add_argument(
        "action",
        choices=["status", "register", "unregister"],
        help="Action to perform",
    )
    sched_parser.add_argument(
        "--time",
        default="22:00",
        help="Daily execution time in HH:MM 24h format (default: 22:00)",
    )

    # run
    run_parser = subparsers.add_parser("run", help="Execute an autonomous development cycle")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Simulate execution without committing or pushing",
    )
    run_parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Flag indicating run was initiated by Windows Task Scheduler",
    )
    run_parser.add_argument(
        "--run-now",
        action="store_true",
        help="Trigger immediate execution",
    )

    args = parser.parse_args()

    if args.command == "verify":
        sys.exit(cmd_verify(args))
    elif args.command == "status":
        sys.exit(cmd_status(args))
    elif args.command == "plan":
        sys.exit(cmd_plan(args))
    elif args.command == "schedule":
        sys.exit(cmd_schedule(args))
    elif args.command == "run":
        sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
