---
type: canonical
source: none
sync: none
sla: none
---

# Baseline Subcommand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `reasonbench baseline capture|compare|status` subcommands that close the capture → compare → regress-check loop currently missing from the CLI.

**Architecture:** Three new `_cmd_baseline_*` functions in `reasonbench/__main__.py` wired into a nested `baseline` subparser. `ModelBaseline` gains a `captured_at` field. `BenchmarkSuite.save_baselines` already exists and handles all disk I/O. The `ci-smoke.yml` no-op is replaced with a real `baseline status` call.

**Tech Stack:** Python 3.12+, argparse (nested subparsers), pydantic v2, pytest + `monkeypatch` + `capsys`, `unittest.mock.patch`

---

## File Map

| Action | Path | What changes |
|--------|------|--------------|
| Modify | `reasonbench/benchmark.py` | Add `captured_at: str = ""` field to `ModelBaseline` |
| Modify | `reasonbench/__main__.py` | Add `_cmd_baseline_capture`, `_cmd_baseline_compare`, `_cmd_baseline_status`; wire nested `baseline` subparser into `main()` |
| Create | `tests/test_baseline.py` | 7 test functions covering all three commands |
| Modify | `.github/workflows/ci-smoke.yml` | Replace no-op with `baseline status --version v1`; fix branch filter `master`→`main` |

---

### Task 1: Add `captured_at` to `ModelBaseline`

**Files:**
- Modify: `reasonbench/benchmark.py:28-38`
- Test: `tests/test_benchmark.py` — append one test to `TestBenchmarkModels`

- [ ] **Step 1: Write the failing test**

Open `tests/test_benchmark.py` and append this test inside `class TestBenchmarkModels`:

```python
def test_model_baseline_captured_at_roundtrip(self):
    b = ModelBaseline(
        model_name="m",
        overall_score=3.5,
        failure_rate=0.2,
        captured_at="2026-04-18T12:00:00+00:00",
    )
    data = b.model_dump_json()
    loaded = ModelBaseline.model_validate_json(data)
    assert loaded.captured_at == "2026-04-18T12:00:00+00:00"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:/Users/mesha/Desktop/Dropbox/GitHub/alawein/fallax
python -m pytest tests/test_benchmark.py::TestBenchmarkModels::test_model_baseline_captured_at_roundtrip -v --no-cov
```

Expected: FAIL — `ModelBaseline` has no `captured_at` field, pydantic raises `ValidationError`.

- [ ] **Step 3: Add `captured_at` to `ModelBaseline` in `benchmark.py`**

Find `class ModelBaseline(BaseModel):` (line 28). The current class ends after `runs: int`. Add one field after `runs`:

```python
class ModelBaseline(BaseModel):
    """Baseline scores for a single model on a benchmark."""

    model_name: str
    overall_score: float
    failure_rate: float
    category_scores: dict[str, float] = Field(default_factory=dict)
    type_scores: dict[str, float] = Field(default_factory=dict)
    assumption_density: float = 0.0
    runs: int = Field(ge=1, default=1)
    captured_at: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_benchmark.py::TestBenchmarkModels::test_model_baseline_captured_at_roundtrip -v --no-cov
```

Expected: PASS.

- [ ] **Step 5: Run full test suite to verify no regressions**

```bash
python -m pytest tests/test_benchmark.py -v --no-cov
```

Expected: all existing tests PASS (adding a field with a default is backward-compatible).

- [ ] **Step 6: Commit**

```bash
git add reasonbench/benchmark.py tests/test_benchmark.py
git commit -m "feat(benchmark): add captured_at field to ModelBaseline"
```

---

### Task 2: Add `baseline status` command

**Files:**
- Modify: `reasonbench/__main__.py`
- Create: `tests/test_baseline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_baseline.py` with this content:

```python
"""Tests for baseline subcommands."""

from __future__ import annotations

import functools
import json

import pytest

from reasonbench.__main__ import main
from reasonbench.benchmark import BenchmarkSuite


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def benchmark_dir(tmp_path):
    """Temp benchmark dir with a v1 suite that has one baseline entry."""
    v1 = tmp_path / "v1"
    v1.mkdir()

    # minimal prompts.jsonl so load_prompts() doesn't raise
    from reasonbench.models import Prompt
    from reasonbench.taxonomy import FailureType
    prompt = Prompt(
        prompt_id="p1",
        failure_type=FailureType.UNSTATED_ASSUMPTION,
        prompt_text="Test prompt",
        difficulty=5,
        template_id="implicit_assumption_trap",
    )
    (v1 / "prompts.jsonl").write_text(prompt.model_dump_json() + "\n", encoding="utf-8")

    baselines = {
        "version": "v1",
        "models": [
            {
                "model_name": "base-model",
                "overall_score": 4.0,
                "failure_rate": 0.2,
                "runs": 1,
                "assumption_density": 0.0,
                "captured_at": "2026-04-18T10:00:00+00:00",
            }
        ],
    }
    (v1 / "baselines.json").write_text(json.dumps(baselines), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def patch_suite(benchmark_dir, monkeypatch):
    """Patch BenchmarkSuite in __main__ to use benchmark_dir."""
    monkeypatch.setattr(
        "reasonbench.__main__.BenchmarkSuite",
        functools.partial(BenchmarkSuite, benchmarks_dir=benchmark_dir),
    )
    return benchmark_dir


# ---------------------------------------------------------------------------
# baseline status
# ---------------------------------------------------------------------------

class TestBaselineStatus:
    def test_status_prints_model_name(self, patch_suite, capsys):
        code = main(["baseline", "status", "--version", "v1"])
        assert code == 0
        captured = capsys.readouterr()
        assert "base-model" in captured.out

    def test_status_empty_baselines(self, patch_suite, benchmark_dir, capsys):
        (benchmark_dir / "v1" / "baselines.json").write_text(
            json.dumps({"version": "v1", "models": []}), encoding="utf-8"
        )
        code = main(["baseline", "status", "--version", "v1"])
        assert code == 0
        captured = capsys.readouterr()
        assert "No baselines" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_baseline.py -v --no-cov
```

Expected: FAIL — `baseline` subcommand doesn't exist yet.

- [ ] **Step 3: Add the `baseline` subparser and `_cmd_baseline_status` to `__main__.py`**

**3a.** Add the import for `BenchmarkSuite` near the top of `__main__.py`. Find the existing imports block (lines 11-13):

```python
from .client import AnthropicClient
from .clients import create_client
from .pipeline import Pipeline
```

Add one line:

```python
from .client import AnthropicClient
from .benchmark import BenchmarkSuite
from .clients import create_client
from .pipeline import Pipeline
```

**3b.** Add `_cmd_baseline_status` before the `main()` function definition. Insert it after `_cmd_benchmark` (around line 320):

```python
def _cmd_baseline_status(args: argparse.Namespace) -> int:
    """Show recorded baselines for a benchmark version."""
    suite = BenchmarkSuite()
    baselines = suite.load_baselines(args.version)

    if not baselines.models:
        print(f"No baselines recorded for {args.version}.")
        return 0

    print(f"\nBaselines ({args.version})")
    print(f"  {'Model':<35} {'Score':>7} {'Fail%':>7} {'Captured':>25}")
    print(f"  {'-'*78}")
    for m in baselines.models:
        print(
            f"  {m.model_name:<35} {m.overall_score:>7.2f} "
            f"{m.failure_rate:>7.1%} {m.captured_at:>25}"
        )
    return 0
```

**3c.** Inside `main()`, find the block of `subparsers.add_parser(...)` calls (starting around line 333). Add the nested `baseline` subparser **after** the `bench_p` block and **before** `args = parser.parse_args(argv)`:

```python
    # -- baseline --
    baseline_p = subparsers.add_parser(
        "baseline", help="Manage benchmark baselines"
    )
    baseline_sub = baseline_p.add_subparsers(dest="baseline_command")

    # baseline status
    stat_p = baseline_sub.add_parser("status", help="Show recorded baselines")
    stat_p.add_argument("--version", default="v1", help="Benchmark version")
```

**3d.** In the dispatch block at the bottom of `main()`, add handling after the `benchmark` block:

```python
    if args.command == "baseline":
        if args.baseline_command == "capture":
            return _cmd_baseline_capture(args)
        if args.baseline_command == "compare":
            return _cmd_baseline_compare(args)
        if args.baseline_command == "status":
            return _cmd_baseline_status(args)
        baseline_p.print_help()
        return 1
```

Note: `_cmd_baseline_capture` and `_cmd_baseline_compare` don't exist yet — that's fine, they won't be called until Task 3 and 4.

- [ ] **Step 4: Run the status tests to verify they pass**

```bash
python -m pytest tests/test_baseline.py::TestBaselineStatus -v --no-cov
```

Expected: both PASS.

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest tests/ -v --no-cov -x
```

Expected: all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add reasonbench/__main__.py tests/test_baseline.py
git commit -m "feat(cli): add baseline status subcommand"
```

---

### Task 3: Add `baseline capture` command

**Files:**
- Modify: `reasonbench/__main__.py`
- Modify: `tests/test_baseline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baseline.py`:

```python
# ---------------------------------------------------------------------------
# baseline capture
# ---------------------------------------------------------------------------

class TestBaselineCapture:
    def _fake_results(self):
        """Two fake EvaluationResults for deterministic scoring."""
        from tests.conftest import make_result
        return [make_result(score=3), make_result(score=5)]

    def test_capture_writes_entry(self, patch_suite, benchmark_dir, monkeypatch, capsys):
        from unittest.mock import patch as mock_patch
        fake = self._fake_results()
        with mock_patch("reasonbench.__main__.Pipeline") as MockPipeline:
            MockPipeline.return_value.run_prompts.return_value = fake
            code = main([
                "baseline", "capture",
                "--version", "v1",
                "--model", "new-model",
                "--judge", "judge-model",
            ])
        assert code == 0
        # reload baselines from disk
        baselines_path = benchmark_dir / "v1" / "baselines.json"
        data = json.loads(baselines_path.read_text(encoding="utf-8"))
        names = [m["model_name"] for m in data["models"]]
        assert "new-model" in names

    def test_capture_replaces_existing_entry(self, patch_suite, benchmark_dir, monkeypatch):
        from unittest.mock import patch as mock_patch
        fake = self._fake_results()
        # capture for "base-model" which already has an entry
        with mock_patch("reasonbench.__main__.Pipeline") as MockPipeline:
            MockPipeline.return_value.run_prompts.return_value = fake
            main([
                "baseline", "capture",
                "--version", "v1",
                "--model", "base-model",
                "--judge", "judge-model",
            ])
        data = json.loads(
            (benchmark_dir / "v1" / "baselines.json").read_text(encoding="utf-8")
        )
        # still exactly one entry for base-model (not duplicated)
        assert sum(1 for m in data["models"] if m["model_name"] == "base-model") == 1

    def test_capture_missing_version_returns_1(self, patch_suite, capsys):
        code = main([
            "baseline", "capture",
            "--version", "v99",
            "--model", "m",
            "--judge", "j",
        ])
        assert code == 1
        assert "not found" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_baseline.py::TestBaselineCapture -v --no-cov
```

Expected: FAIL — `_cmd_baseline_capture` doesn't exist yet.

- [ ] **Step 3: Add `_cmd_baseline_capture` to `__main__.py`**

Insert before `_cmd_baseline_status` (which you added in Task 2):

```python
def _cmd_baseline_capture(args: argparse.Namespace) -> int:
    """Capture baseline scores for one model on a benchmark version."""
    from datetime import UTC, datetime

    suite = BenchmarkSuite()
    try:
        prompts = suite.load_prompts(args.version)
    except FileNotFoundError:
        print(f"Error: benchmark {args.version} not found", file=sys.stderr)
        return 1

    client = _make_client(args)
    pipeline = Pipeline(
        client=client,
        models=[args.model],
        judge_model=args.judge,
        output_path=Path(args.output),
        seed=42,
    )
    results = pipeline.run_prompts(prompts)
    scores = suite.score_results(results)

    from .benchmark import ModelBaseline
    baselines = suite.load_baselines(args.version)
    entry = ModelBaseline(
        model_name=args.model,
        overall_score=scores["overall_score"],
        failure_rate=scores["failure_rate"],
        category_scores=scores["category_scores"],
        type_scores=scores["type_scores"],
        captured_at=datetime.now(UTC).isoformat(),
    )
    baselines.models = [m for m in baselines.models if m.model_name != args.model]
    baselines.models.append(entry)
    path = suite.save_baselines(baselines)

    print(f"\nBaseline captured ({args.version} / {args.model})")
    print(f"  Overall score:  {scores['overall_score']:.2f}")
    print(f"  Failure rate:   {scores['failure_rate']:.1%}")
    print(f"  Prompts scored: {scores['total']}")
    print(f"  Saved to:       {path}")
    return 0
```

- [ ] **Step 4: Add the `capture` argparse block to `main()`**

Inside the `baseline_sub` block you started in Task 2 (after `stat_p` definition), add:

```python
    # baseline capture
    cap_p = baseline_sub.add_parser("capture", help="Capture baseline for a model")
    cap_p.add_argument("--version", default="v1", help="Benchmark version")
    cap_p.add_argument("--model", required=True, help="Model to evaluate")
    cap_p.add_argument(
        "--judge",
        required=not bool(default_judge),
        default=default_judge or None,
        help="Judge model (or set REASONBENCH_JUDGE_MODEL)",
    )
    cap_p.add_argument("--output", default="baseline_run.jsonl")
    cap_p.add_argument("--provider", default="anthropic", help="LLM provider")
```

- [ ] **Step 5: Run capture tests**

```bash
python -m pytest tests/test_baseline.py::TestBaselineCapture -v --no-cov
```

Expected: all 3 PASS.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ --no-cov -x -q
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add reasonbench/__main__.py tests/test_baseline.py
git commit -m "feat(cli): add baseline capture subcommand"
```

---

### Task 4: Add `baseline compare` command

**Files:**
- Modify: `reasonbench/__main__.py`
- Modify: `tests/test_baseline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baseline.py`:

```python
# ---------------------------------------------------------------------------
# baseline compare
# ---------------------------------------------------------------------------

class TestBaselineCompare:
    def _fake_results_with_score(self, avg_score: float):
        """Return results whose average score equals avg_score."""
        from tests.conftest import make_result
        # Two results averaging to avg_score: use avg_score for both
        score_int = round(avg_score)
        return [make_result(score=score_int), make_result(score=score_int)]

    def test_compare_exits_zero_within_threshold(self, patch_suite, monkeypatch):
        """4.0 baseline, 3.9 current, threshold 0.05 → no regression → exit 0."""
        from unittest.mock import patch as mock_patch
        fake = self._fake_results_with_score(3.9)
        with mock_patch("reasonbench.__main__.Pipeline") as MockPipeline:
            MockPipeline.return_value.run_prompts.return_value = fake
            code = main([
                "baseline", "compare",
                "--version", "v1",
                "--model", "base-model",
                "--judge", "judge-model",
                "--threshold", "0.5",
            ])
        assert code == 0

    def test_compare_exits_two_on_regression(self, patch_suite, monkeypatch, capsys):
        """4.0 baseline, 1.0 current, threshold 0.05 → regression → exit 2."""
        from unittest.mock import patch as mock_patch
        fake = self._fake_results_with_score(1.0)
        with mock_patch("reasonbench.__main__.Pipeline") as MockPipeline:
            MockPipeline.return_value.run_prompts.return_value = fake
            code = main([
                "baseline", "compare",
                "--version", "v1",
                "--model", "base-model",
                "--judge", "judge-model",
                "--threshold", "0.05",
            ])
        assert code == 2
        assert "REGRESSION" in capsys.readouterr().out

    def test_compare_exits_one_on_missing_baseline(self, patch_suite, capsys):
        """No baseline for 'unknown-model' → exit 1."""
        code = main([
            "baseline", "compare",
            "--version", "v1",
            "--model", "unknown-model",
            "--judge", "judge-model",
        ])
        assert code == 1
        assert "no baseline" in capsys.readouterr().err.lower()

    def test_compare_exits_one_on_missing_version(self, patch_suite, capsys):
        code = main([
            "baseline", "compare",
            "--version", "v99",
            "--model", "base-model",
            "--judge", "judge-model",
        ])
        assert code == 1
        assert "not found" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_baseline.py::TestBaselineCompare -v --no-cov
```

Expected: FAIL — `_cmd_baseline_compare` doesn't exist yet.

- [ ] **Step 3: Add `_cmd_baseline_compare` to `__main__.py`**

Insert before `_cmd_baseline_capture`:

```python
def _cmd_baseline_compare(args: argparse.Namespace) -> int:
    """Compare a model run against its recorded baseline."""
    suite = BenchmarkSuite()
    try:
        prompts = suite.load_prompts(args.version)
    except FileNotFoundError:
        print(f"Error: benchmark {args.version} not found", file=sys.stderr)
        return 1

    baselines = suite.load_baselines(args.version)
    recorded = next(
        (m for m in baselines.models if m.model_name == args.model), None
    )
    if recorded is None:
        print(
            f"Error: no baseline for '{args.model}' — run 'baseline capture' first",
            file=sys.stderr,
        )
        return 1

    client = _make_client(args)
    pipeline = Pipeline(
        client=client,
        models=[args.model],
        judge_model=args.judge,
        output_path=Path(args.output),
        seed=42,
    )
    results = pipeline.run_prompts(prompts)
    scores = suite.score_results(results)

    delta = scores["overall_score"] - recorded.overall_score
    fr_delta = scores["failure_rate"] - recorded.failure_rate
    regressed = delta < -args.threshold

    print(f"\nBaseline comparison ({args.version} / {args.model})")
    print(f"  {'Metric':<20} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
    print(f"  {'-' * 54}")
    print(
        f"  {'overall_score':<20} {recorded.overall_score:>10.2f} "
        f"{scores['overall_score']:>10.2f} {delta:>+10.2f}"
    )
    print(
        f"  {'failure_rate':<20} {recorded.failure_rate:>10.1%} "
        f"{scores['failure_rate']:>10.1%} {fr_delta:>+10.1%}"
    )

    if regressed:
        print(
            f"\nREGRESSION: {args.model} overall_score dropped "
            f"{abs(delta):.2f} points (threshold: {args.threshold})"
        )
        return 2

    print(f"\nPASS: within threshold ({args.threshold})")
    return 0
```

- [ ] **Step 4: Add the `compare` argparse block to `main()`**

Inside the `baseline_sub` block (after the `cap_p` block you added in Task 3), add:

```python
    # baseline compare
    cmp_p = baseline_sub.add_parser("compare", help="Compare run against baseline")
    cmp_p.add_argument("--version", default="v1", help="Benchmark version")
    cmp_p.add_argument("--model", required=True, help="Model to compare")
    cmp_p.add_argument(
        "--judge",
        required=not bool(default_judge),
        default=default_judge or None,
        help="Judge model",
    )
    cmp_p.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Allowed overall_score drop before regression (default: 0.05)",
    )
    cmp_p.add_argument("--output", default="compare_run.jsonl")
    cmp_p.add_argument("--provider", default="anthropic", help="LLM provider")
```

- [ ] **Step 5: Run compare tests**

```bash
python -m pytest tests/test_baseline.py::TestBaselineCompare -v --no-cov
```

Expected: all 4 PASS.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ --no-cov -x -q
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add reasonbench/__main__.py tests/test_baseline.py
git commit -m "feat(cli): add baseline compare subcommand"
```

---

### Task 5: Fix `ci-smoke.yml`

**Files:**
- Modify: `.github/workflows/ci-smoke.yml`

No tests needed — this is a CI workflow change. Correctness is verified by reading the file after editing.

- [ ] **Step 1: Replace the file contents**

Write `.github/workflows/ci-smoke.yml` with:

```yaml
name: ci-smoke

on:
  push:
    branches: [main]
    paths-ignore:
      - '**/*.md'
      - 'docs/**'
      - '.github/ISSUE_TEMPLATE/**'
      - 'LICENSE'
  pull_request:
    branches: [main]
    paths-ignore:
      - '**/*.md'
      - 'docs/**'
      - '.github/ISSUE_TEMPLATE/**'
      - 'LICENSE'

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.12"
      - name: Install
        run: uv sync --extra dashboard
      - name: Smoke — baseline status
        run: python -m reasonbench baseline status --version v1
```

- [ ] **Step 2: Verify the file**

```bash
cat C:/Users/mesha/Desktop/Dropbox/GitHub/alawein/fallax/.github/workflows/ci-smoke.yml
```

Confirm: `branches: [main]` appears twice (push and pull_request), and `python -m reasonbench baseline status --version v1` is the smoke step.

- [ ] **Step 3: Commit**

```bash
cd C:/Users/mesha/Desktop/Dropbox/GitHub/alawein/fallax
git add .github/workflows/ci-smoke.yml
git commit -m "ci: replace smoke no-op with baseline status; fix master→main"
```

---

## Self-Review

**Spec coverage:**
- `baseline status` — Task 2 ✓
- `baseline capture` — Task 3 ✓
- `baseline compare` — Task 4 ✓
- `captured_at` field — Task 1 ✓
- Replace/not-append on same model_name — Task 3, `test_capture_replaces_existing_entry` ✓
- Exit codes 0/1/2 — Task 4 ✓
- `ci-smoke.yml` fixed — Task 5 ✓
- Missing version → exit 1 — Task 3 step 1 `test_capture_missing_version_returns_1`, Task 4 step 1 `test_compare_exits_one_on_missing_version` ✓
- Missing baseline for model → exit 1 — Task 4 step 1 `test_compare_exits_one_on_missing_baseline` ✓

**Placeholder scan:** None — all steps have complete code.

**Type consistency:**
- `BenchmarkSuite()` called with no args throughout — consistent ✓
- `suite.load_baselines(args.version)` → `BenchmarkBaselines` — matches `benchmark.py:85` ✓
- `suite.save_baselines(baselines)` takes `BenchmarkBaselines` — matches `benchmark.py:93` ✓
- `ModelBaseline` imported from `.benchmark` inside the function — consistent across Task 1 and Task 3 ✓
- `pipeline.run_prompts(prompts)` — matches `_cmd_benchmark` usage at line 301 ✓
- `scores["overall_score"]`, `scores["failure_rate"]`, `scores["category_scores"]`, `scores["type_scores"]` — all present in `score_results()` return dict (`benchmark.py:177`) ✓
