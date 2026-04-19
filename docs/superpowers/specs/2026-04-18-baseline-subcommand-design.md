---
type: canonical
source: none
sync: none
sla: none
---

# Baseline Subcommand Design

**Goal:** Add a `reasonbench baseline` subcommand group that closes the capture → compare → regress-check loop missing from the existing benchmark CLI.

**Approach:** Option B — new `baseline` subcommand group with three focused commands: `capture`, `compare`, `status`. Each does one thing. No changes to the existing `benchmark` subcommand behavior.

---

## Commands

### `reasonbench baseline capture`

```
reasonbench baseline capture \
  --version v1 \
  --models claude-sonnet-4-6 \
  --judge claude-sonnet-4-6 \
  [--provider anthropic] \
  [--output benchmark_results.jsonl]
```

Runs the benchmark pipeline against `benchmarks/v1/` prompts, scores the results via `BenchmarkSuite.score_results()`, then writes a model entry into `benchmarks/v1/baselines.json`. If an entry for the same `model_name` already exists, it is replaced. Exits 0 on success.

### `reasonbench baseline compare`

```
reasonbench baseline compare \
  --version v1 \
  --models claude-sonnet-4-6 \
  --judge claude-sonnet-4-6 \
  [--threshold 0.05] \
  [--provider anthropic] \
  [--output benchmark_results.jsonl]
```

Runs the same pipeline and scoring as `capture`. Diffs `overall_score` against the recorded baseline for each model. Prints a delta table to stdout. Exit codes:
- `0` — all models within threshold
- `1` — missing baseline for a requested model, or other configuration error
- `2` — at least one model regressed beyond threshold (CI-distinguishable from misconfiguration)

Default threshold: `0.05` (5 percentage points on `overall_score`).

### `reasonbench baseline status`

```
reasonbench baseline status [--version v1]
```

Reads `benchmarks/<version>/baselines.json` and prints a formatted table of recorded models, their scores, failure rates, and `captured_at` timestamps. Makes zero API calls. Safe to run in CI with no credentials.

---

## Data Shape

`benchmarks/v1/baselines.json` gains structured model entries:

```json
{
  "version": "v1",
  "models": [
    {
      "model_name": "claude-sonnet-4-6",
      "overall_score": 0.74,
      "failure_rate": 0.18,
      "category_scores": {
        "logical_deduction": 0.71,
        "causal_inference": 0.77
      },
      "captured_at": "2026-04-18T12:00:00Z"
    }
  ]
}
```

The `captured_at` field is an ISO 8601 UTC timestamp written at capture time.

---

## File Changes

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `reasonbench/benchmark.py` | Add `save_baselines(version, entry)` to `BenchmarkSuite` |
| Modify | `reasonbench/__main__.py` | Add `_cmd_baseline_capture`, `_cmd_baseline_compare`, `_cmd_baseline_status`; wire into `main()` |
| Modify | `tests/test_benchmark.py` | Add 4 new test functions (or new `tests/test_baseline.py` if it keeps the file focused) |
| Modify | `.github/workflows/ci-smoke.yml` | Replace no-op with `reasonbench baseline status --version v1` |

No new files in `reasonbench/` — all logic lives in existing modules.

---

## BenchmarkSuite.save_baselines

New method on the existing `BenchmarkSuite` class in `reasonbench/benchmark.py`:

```python
def save_baselines(self, version: str, entry: dict) -> None:
    """Write or replace a model entry in baselines.json."""
```

- Loads the existing `baselines.json` for the version (or starts from `{"version": version, "models": []}` if absent).
- Removes any existing entry with the same `model_name`.
- Appends the new entry.
- Writes back atomically (write to `.tmp`, rename).

---

## Error Handling

| Condition | Command | Exit | Output |
|-----------|---------|------|--------|
| Benchmark version not found | all | 1 | `Error: benchmark <version> not found` |
| No baseline recorded for model | compare | 1 | `Error: no baseline for '<model>' — run capture first` |
| Score regression beyond threshold | compare | 2 | delta table + `REGRESSION: <model> dropped <N> points` |
| API error during pipeline | capture, compare | propagates | exception message to stderr |

---

## Testing

All tests use `tmp_path` for baselines file writes. No test touches `benchmarks/v1/baselines.json`. Pipeline and scoring calls are monkeypatched — no API keys needed.

| Test | File | What it verifies |
|------|------|-----------------|
| `test_baseline_capture_writes_entry` | `test_baseline.py` | `main(["baseline", "capture", ...])` writes correct entry to baselines JSON |
| `test_baseline_capture_replaces_existing_entry` | `test_baseline.py` | Second capture for same model replaces, not appends |
| `test_baseline_compare_exits_zero_on_pass` | `test_baseline.py` | Score 0.78 vs baseline 0.80, threshold 0.05 → exit 0 |
| `test_baseline_compare_exits_two_on_regression` | `test_baseline.py` | Score 0.70 vs baseline 0.80, threshold 0.05 → exit 2 |
| `test_baseline_compare_exits_one_on_missing_baseline` | `test_baseline.py` | No entry in baselines for requested model → exit 1 |
| `test_baseline_status_prints_table` | `test_baseline.py` | Reads fixture baselines.json, asserts model name in stdout |

---

## CI Smoke Test

`ci-smoke.yml` replaces its current no-op body with:

```yaml
- name: Install
  run: uv sync --extra dashboard
- name: Smoke — baseline status
  run: python -m reasonbench baseline status --version v1
```

Trigger: push/PR to `main`. No credentials required.

---

## Out of Scope

- Cross-repo benchmark consistency schema (X-T3) — separate spec
- Scheduled GitHub Actions to capture baselines automatically — follow-on after this lands
- Dashboard visualization of baseline history — follow-on
