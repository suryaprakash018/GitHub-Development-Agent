# Autonomous AI + Data Development Agent — Operations Manual

This operations manual provides comprehensive administrative and operational procedures for running, monitoring, troubleshooting, and transitioning the **Autonomous AI + Data Development Agent**.

---

## 1. System Architecture & Safety Model

The system operates under a strict **Decoupled Architecture**:
* **Agent Engine Repository (`GitHub-Development-Agent`)**: Houses the orchestration pipeline, AST repository analyzer, DAG roadmap engine, test harness, self-remediator, Git client, scheduler manager, and unit test suites.
* **Target Portfolio Repository (External)**: An entirely independent repository where all portfolio code, unit tests, and documentation are synthesized.

```
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│     Agent Engine Workspace      │           │    External Target Portfolio    │
│  (GitHub-Development-Agent)     │           │    (Separate Git Repository)    │
│                                 │           │                                 │
│  • Orchestrator & Task DAG      │  AST Scan │  • src/data_science/...         │
│  • Concurrency Lock (.lock)     │ ────────> │  • tests/data_science/...       │
│  • Gemini AI Synthesis          │           │  • conftest.py, pyproject.toml  │
│  • Automated Pytest & Ruff      │ Rollback/ │                                 │
│  • Structured Telemetry Runs    │ Commit    │  • Pristine Git Worktree        │
└─────────────────────────────────┘ ────────> └─────────────────────────────────┘
```

### Immutable Safety Guardrails
1. **Anti-Self-Targeting**: The agent refuses to execute if configured to target itself or any subpath within its root.
2. **Clean Worktree Guarantee**: Execution aborts if unexpected or untracked changes exist in the target repository before execution.
3. **Single-Instance Concurrency**: `ExecutionLock` prevents multiple processes from modifying the target repository concurrently.
4. **Deterministic Quality Gate**: Reject code failing unit tests, violating lint standards, modifying out-of-scope files, or introducing trivial churn.
5. **DRY_RUN Rollback Guarantee**: While `DRY_RUN=true`, all synthesized files and bytecode are cleaned up after testing, leaving the target working tree pristine.

---

## 2. Normal Operations & CLI Reference

All routines are accessed via the `dev-agent` CLI entrypoint:

### 2.1 System Health & Connectivity Verification
Inspects Python runtime, Git CLI, decoupled boundaries, remote accessibility, and Gemini API connectivity:
```powershell
python -m agent.cli verify
```

### 2.2 Inspecting Roadmap Progress & Status
Displays completed milestones, total tests written, commits created, and the next planned milestone:
```powershell
python -m agent.cli status
```

### 2.3 Previewing Next Milestone Plan
Performs a read-only AST inspection of the target repository and displays the exact next task candidate:
```powershell
python -m agent.cli plan
```

### 2.4 Executing an Autonomous Cycle (Dry-Run Mode)
Runs code generation, pytest validation, ruff linting, self-remediation, quality gate scoring, and generates a conventional commit message without modifying Git:
```powershell
python -m agent.cli run --dry-run
```

---

## 3. Scheduled Execution (Windows Task Scheduler)

The agent supports unattended daily execution using the native Windows Task Scheduler (`schtasks.exe`).

### 3.1 Managing the Schedule
* **Check Scheduler Status**:
  ```powershell
  python -m agent.cli schedule status
  ```
* **Register Daily Task (Configurable Time, Default 22:00)**:
  ```powershell
  python -m agent.cli schedule register --time 22:00
  ```
* **Unregister / Disable Task**:
  ```powershell
  python -m agent.cli schedule unregister
  ```

### 3.2 PowerShell Automation Script
Alternatively, use the automated PowerShell runner:
```powershell
# Register for 23:30 daily
powershell -ExecutionPolicy Bypass -File .\scripts\setup_scheduler.ps1 -Action register -Time "23:30"

# Check status
powershell -ExecutionPolicy Bypass -File .\scripts\setup_scheduler.ps1 -Action status

# Remove task
powershell -ExecutionPolicy Bypass -File .\scripts\setup_scheduler.ps1 -Action unregister
```

---

## 4. Telemetry & Audit Trails

Every execution cycle produces a timestamped JSON audit log in `state/runs/`:
```text
state/runs/run_<timestamp>_<run_id>_<task_id>.json
```

### Telemetry Content
* **Run Identification**: UTC timestamp, duration, execution mode (`dry_run`, `auto_commit`, `auto_push`).
* **Task Scope**: Track ID, milestone ID, task title, target files, test files.
* **AST Metrics**: Python module count, test file count, symbol counts before and after synthesis.
* **Remediation Details**: Total attempts, diagnoses, diff adjustments.
* **Validation Outcome**: Pytest pass/fail, test count, failure trace, Ruff clean status.
* **Quality Gate Decision**: Overall score, pass/fail status, rejection reasons (if any).
* **Git Operations**: Conventional commit subject, commit hash, push status.
* **Secret Redaction**: All API keys, passwords, bearer tokens, and credentials are automatically scrubbed and replaced with `[REDACTED_CREDENTIAL]`.

---

## 5. Failure Recovery & Troubleshooting

### Scenario A: Concurrency Lock Error (`LOCK_BUSY`)
* **Symptom**: CLI reports `Cannot run: another agent instance is currently executing`.
* **Resolution**:
  1. Check Windows Task Manager for any hanging Python processes.
  2. If no process is running, the lock file `state/execution.lock` may be stale from an abrupt shutdown.
  3. Delete the stale lock file:
     ```powershell
     Remove-Item -Path .\state\execution.lock -Force
     ```

### Scenario B: Dirty Worktree Abort (`GIT_SAFETY_ERROR`)
* **Symptom**: `Target repository has uncommitted modifications`.
* **Resolution**: The agent will never overwrite your manual work. Navigate to the target repository and either commit, stash, or revert your changes:
  ```powershell
  cd C:\path\to\target-repo
  git status
  git stash  # or git commit -m "WIP"
  ```

### Scenario C: Quality Gate Rejection
* **Symptom**: `Task rejected by quality gate (Score < 0.80)`.
* **Resolution**:
  1. Review the generated audit log in `state/runs/`.
  2. Check rejection reasons (e.g. failing tests, lines of code below 15, missing type hints).
  3. The agent will automatically re-attempt the task with a clean slate on the next cycle, incorporating failure context into its remediation prompt.

---

## 6. Controlled Transition to Live Mode

Follow this step-by-step checklist to transition from safe dry-run mode to live unattended commits:

### Step 1: Pre-Flight Verification
Run the diagnostic suite:
```powershell
python -m agent.cli verify
```
Confirm:
- [x] Target repository path is valid and clean.
- [x] Target repository branch is `main`.
- [x] Remote `origin` is configured and accessible (`git remote -v`).
- [x] Gemini API key is valid and connectivity verified.

### Step 2: Validate With a Clean Dry-Run
Execute a simulated cycle and inspect the proposed code diff and commit:
```powershell
python -m agent.cli run --dry-run
```
Confirm that:
- Quality gate passes (`PASSED`).
- Diff preview contains substantive, tested AI/Data Science code.
- Target repository remains completely clean afterwards (`git status`).

### Step 3: Update Operational Configuration in `.env`
Edit `.env` (never commit this file):
```env
# Enable live commit and push
DRY_RUN=false
AUTO_COMMIT=true
AUTO_PUSH=true
```

### Step 4: Execute First Live Contribution
Trigger a single controlled live run:
```powershell
python -m agent.cli run --run-now
```
Inspect the target repository:
```powershell
cd C:\path\to\target-repo
git log -n 1
git status
```
Verify that:
1. Exactly the approved task files were staged and committed.
2. The commit message follows Conventional Commits format.
3. The commit was pushed to your remote GitHub repository.

### Step 5: Enable Windows Scheduled Task
Once verified, register the scheduled task:
```powershell
python -m agent.cli schedule register --time 22:00
```

---

## 7. Emergency Stop Procedures

In the event of an unexpected behavior, anomaly, or required maintenance:

1. **Unregister Scheduled Task**:
   ```powershell
   python -m agent.cli schedule unregister
   ```
2. **Revert Operational Mode to Safe Mode**:
   Set in `.env`:
   ```env
   DRY_RUN=true
   AUTO_COMMIT=false
   AUTO_PUSH=false
   ```
3. **Terminate Any Running Agent Process**:
   ```powershell
   Get-Process -Name python | Where-Object { $_.CommandLine -like "*agent.cli*" } | Stop-Process -Force
   ```
4. **Remove Any Lingering Lockfile**:
   ```powershell
   Remove-Item -Path .\state\execution.lock -Force -ErrorAction SilentlyContinue
   ```
5. **Inspect Target Repository**:
   ```powershell
   cd C:\path\to\target-repo
   git status
   git reset --hard HEAD  # Only if uncommitted temporary files remained
   ```
