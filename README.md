# Autonomous AI + Data Development Agent

An autonomous, long-term development agent built to continuously advance an external target repository in the **AI & Data Engineering** domain.

This system is engineered for **genuine technical development**, completely decoupled from vanity metrics, whitespace churn, or artificial contribution bots.

---

## Architectural Principles

1. **Decoupled Architecture**: This repository (`GitHub-Development-Agent`) contains only the orchestration engine, safety guardrails, state manager, validation harness, and tests. It operates strictly upon a separate, configurable target portfolio repository.
2. **Deterministic-First with AI Synthesis**: Deterministic code manages Git operations, AST inspection, test execution, path validation, and lockfiles. Google Gemini is utilized strictly for technical planning, code generation, error remediation, and semantic utility scoring.
3. **Directed Acyclic Graph (DAG) Roadmap**: Progresses logically from fundamental scientific computing (NumPy/Pandas/Statistics) to Machine Learning, Deep Learning, Generative AI/RAG, Data Pipelines, and Production MLOps.
4. **Hard Safety Gates**:
   - Prevention of self-modification (the agent cannot target itself).
   - Clean working tree verification before any modifications.
   - Comprehensive unit testing & linting before commit consideration.
   - Initial `DRY_RUN=true` enforcement.
   - Prevention of destructive Git operations (`reset --hard`, `push --force`).

---

## Quick Start & Verification

### 1. Prerequisites
- Python 3.11+
- Git CLI configured with your credentials

### 2. Configuration
Copy `.env.example` to `.env` and configure your settings:
```powershell
cp .env.example .env
```
Edit `.env`:
- `GEMINI_API_KEY`: Your Google Gemini API key.
- `TARGET_REPOSITORY_PATH`: Absolute path to your separate target portfolio repository.

### 3. Verify Health & Environment
Run the deterministic verification check:
```powershell
python -m agent.cli verify
```

### 4. Inspect Status
Check current roadmap stage and target repository metrics:
```powershell
python -m agent.cli status
```

### 5. Run Dry-Run Cycle
Execute a complete planning, code generation, and test validation cycle without committing or pushing:
```powershell
python -m agent.cli run --dry-run
```
