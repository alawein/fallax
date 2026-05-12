---
type: canonical
source: none
sync: none
sla: none
---

# Phase 0: Fix Dashboard Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get all 372 tests passing by installing the missing `dashboard` extra into the project virtual environment.

**Architecture:** The `dashboard` extra (`fastapi>=0.136.1`, `uvicorn>=0.46.0`) is declared in `pyproject.toml` but was never synced into `.venv`. When pytest runs, it cannot find `fastapi` in `.venv` and falls back to a system Python 3.14 installation that has a corrupt mixed-version FastAPI (applications.py and routing.py from different sub-versions). Installing the extra gives pytest a consistent, correct FastAPI. The `dashboard/api.py` code requires no changes — it does not use the deprecated `on_startup`/`on_shutdown` kwargs.

**Tech Stack:** Python 3.12+, uv, pytest, FastAPI 0.136.1+

---

## File Map

| Action | Path | What changes |
|---|---|---|
| No code changes | `dashboard/api.py` | Confirmed compatible — no changes needed |
| No code changes | `pyproject.toml` | Already correctly pins `fastapi>=0.136.1` in `dashboard` extra |

All work in this plan is environment installation and verification — no source files are modified.

---

## Task 1: Install the dashboard extra and verify tests pass

**Files:**
- No files modified — environment change only

- [ ] **Step 1: Confirm the current failure**

Run: `uv run pytest tests/test_dashboard.py -q --no-cov`

Expected output (approximately):
```
2 failed, 12 errors
```

This confirms the baseline before the fix.

- [ ] **Step 2: Install the dashboard extra**

Run: `uv sync --extra dashboard`

Expected output: uv resolves and installs `fastapi>=0.136.1` and `uvicorn>=0.46.0` into `.venv`. You will see lines like:
```
Resolved ... packages in ...
Installed fastapi-0.136.x ...
Installed uvicorn-0.46.x ...
```

If uv reports "Nothing to install" or a version below 0.136.1, run `uv sync --extra dashboard --upgrade` to force a fresh resolution.

- [ ] **Step 3: Verify the installed FastAPI version**

Run: `uv run python -c "import fastapi; print(fastapi.__version__)"`

Expected: a version string `>=0.136.1`, e.g. `0.136.1`.

If this prints a version below `0.136.1`, the system Python is still shadowing `.venv`. In that case always prefix test runs with `uv run`.

- [ ] **Step 4: Run dashboard tests only**

Run: `uv run pytest tests/test_dashboard.py -v --no-cov`

Expected: all 14 tests PASS with output like:
```
tests/test_dashboard.py::TestListExperiments::test_returns_experiments PASSED
tests/test_dashboard.py::TestListExperiments::test_empty_dir PASSED
tests/test_dashboard.py::TestListExperiments::test_nonexistent_dir PASSED
tests/test_dashboard.py::TestGetReport::test_returns_report PASSED
tests/test_dashboard.py::TestGetReport::test_not_found PASSED
tests/test_dashboard.py::TestGetResults::test_returns_all_results PASSED
tests/test_dashboard.py::TestGetResults::test_filter_by_min_score PASSED
tests/test_dashboard.py::TestGetResults::test_filter_by_round PASSED
tests/test_dashboard.py::TestGetResults::test_nonexistent_round PASSED
tests/test_dashboard.py::TestGetResults::test_not_found PASSED
tests/test_dashboard.py::TestGetSummary::test_returns_summary PASSED
tests/test_dashboard.py::TestGetSummary::test_severity_breakdown PASSED
tests/test_dashboard.py::TestModelComparison::test_returns_model_stats PASSED
tests/test_dashboard.py::TestModelComparison::test_not_found PASSED
14 passed
```

If any test fails with a new error (not the original `on_startup` TypeError), read the error carefully. Likely causes:
- `starlette.testclient` import error → run `uv sync --extra dev` to ensure `httpx` is installed
- Route not found (404 when 200 expected) → check `dashboard/api.py` route paths haven't changed

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest tests/ -q --no-cov`

Expected: `372 passed` (358 previously passing + 14 dashboard tests), 0 failed, 0 errors.

If the count is different, note which tests are failing and fix them before continuing.

- [ ] **Step 6: Commit the lock file update**

The only file changed on disk is `uv.lock` (dependency lock file updated to include fastapi and uvicorn).

```bash
git add uv.lock
git commit -m "chore(deps): install dashboard extra (fastapi>=0.136.1, uvicorn) — fixes test_dashboard"
```

---

## Self-Review

**Spec coverage:** Phase 0 requires all tests green and FastAPI pin consistent. This plan delivers both. ✓

**Placeholder scan:** No placeholders — all steps have exact commands and expected output. ✓

**Type consistency:** No code changes; no type drift possible. ✓
