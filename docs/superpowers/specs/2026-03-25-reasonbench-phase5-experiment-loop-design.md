---
type: canonical
source: none
sync: none
sla: none
---

# ReasonBench Phase 5: Experiment Loop & Reporting — Design Spec

## Goal

Wire the full feedback loop together: run → analyze → evolve → re-run → report. An `Experiment` orchestrator chains `Pipeline` → `Analyzer` → `PromptEvolver` → `Pipeline` into a multi-round evaluation, with per-round tracking and a final structured report.

## Problem

Phases 1-4 built every piece of the adversarial reasoning loop as independent modules. Users must manually chain CLI commands (`run` → `analyze` → `evolve` → `run` again), passing JSONL files between steps. There is no way to:

- Run multiple rounds automatically
- Track how prompts evolve across rounds
- See whether evolved prompts are actually harder
- Get a summary report of an entire experiment

## Architecture

Two new modules, one new data model, one new CLI subcommand.

### Data Model: `ExperimentRound`

Added to `reasonbench/models.py`:

```python
class ExperimentRound(BaseModel):
    """Metadata for one round of an experiment."""
    round_number: int = Field(ge=1)
    prompts_evaluated: int = Field(ge=0)
    avg_score: float
    failure_rate: float
    hard_case_count: int = Field(ge=0)
    evolved_count: int = Field(ge=0)
    repair_success_rate: float | None = None
```

Captures per-round stats. Intentionally lightweight — no nested result data, just aggregate metrics.

### Module: `experiment.py`

```python
class Experiment:
    def __init__(
        self, client: LLMClient, models: list[str], judge_model: str,
        output_dir: Path, evolve_model: str,
        params_dir: Path | None = None, seed: int | None = None,
    ) -> None: ...

    def run(
        self, initial_count: int = 10, rounds: int = 3, min_score: int = 6
    ) -> dict: ...
```

**Round logic:**

- **Round 1:** `PromptGenerator.generate_batch(initial_count)` → pipeline evaluates prompts → `Analyzer` extracts hard cases (score >= `min_score`) → `PromptEvolver` produces next-round prompts
- **Rounds 2+:** Pipeline evaluates evolved prompts directly (no generator) → analyze → evolve
- **Final round:** After evaluation, `RootCauseExtractor` runs on all results combined, `SelfRepairTester` runs on final-round failures only

Each round writes results to `output_dir/round_{n}.jsonl`.

**Return value:**

```python
{
    "rounds": list[ExperimentRound],
    "root_cause_patterns": list[RootCausePattern],
    "repair_results": list[RepairResult],
    "total_prompts": int,
    "total_failures": int,
}
```

**Key design decisions:**

- Pipeline is instantiated per-round with a round-specific output path, not reused
- Evolved prompts from round N become the input for round N+1
- If evolution produces zero prompts (no hard cases), the experiment terminates early
- The `Experiment` class does NOT subclass or wrap `Pipeline` — it composes it

### Module: `report.py`

```python
class ReportBuilder:
    def __init__(self, experiment_data: dict) -> None: ...
    def build(self) -> dict: ...
    def to_markdown(self) -> str: ...
```

**`build()` output:**

```python
{
    "total_rounds": int,
    "total_prompts": int,
    "total_failures": int,
    "score_trend": list[float],       # avg score per round
    "failure_trend": list[float],     # failure rate per round
    "score_delta": float,             # round N avg - round 1 avg
    "failure_delta": float,           # round N rate - round 1 rate
    "hardening_rate": float,          # % of evolved prompts that stayed hard
    "repair_success_rate": float | None,
    "top_patterns": list[dict],       # top 5 root cause patterns as dicts
}
```

**`to_markdown()` sections:**

1. Overview (total rounds, prompts, failures)
2. Per-round table (round, prompts, avg score, failure rate, evolved count)
3. Trends (score/failure deltas, whether prompts got harder)
4. Top root cause patterns
5. Repair success summary

### CLI: `experiment` subcommand

```
reasonbench experiment \
    --models m1 m2 \
    --judge j \
    --evolve-model e \
    --rounds 3 \
    --count 10 \
    --min-score 6 \
    --output-dir ./experiment/
```

All model args required (no defaults). Writes:

- `experiment/round_1.jsonl`, `round_2.jsonl`, ... (per-round results)
- `experiment/report.json` (structured report from `ReportBuilder.build()`)
- `experiment/report.md` (human-readable from `ReportBuilder.to_markdown()`)

### Public API additions to `__init__.py`

New exports: `Experiment`, `ExperimentRound`, `ReportBuilder`

## Data Flow

```
                    ┌──────────────────────────────────────────────┐
                    │                Experiment.run()               │
                    │                                              │
Round 1:            │  Generator → Pipeline → Analyzer → Evolver  │
                    │      ↓                                ↓     │
                    │  round_1.jsonl              evolved prompts  │
                    │                                   ↓         │
Round 2:            │              Pipeline → Analyzer → Evolver  │
                    │      ↓                                ↓     │
                    │  round_2.jsonl              evolved prompts  │
                    │                                   ↓         │
Round N:            │              Pipeline → Analyzer             │
                    │      ↓            ↓                          │
                    │  round_N.jsonl  RootCauseExtractor           │
                    │                 SelfRepairTester             │
                    │                       ↓                      │
                    │               ReportBuilder                  │
                    │            ↓              ↓                  │
                    │      report.json    report.md                │
                    └──────────────────────────────────────────────┘
```

## Dependencies

- No new external dependencies
- Composes existing modules: Pipeline, Analyzer, PromptEvolver, SelfRepairTester, RootCauseExtractor
- Reuses existing LLMClient protocol and MockClient test fixture

## Task Breakdown

| Task | Component | New/Modified Files | Tests |
|------|-----------|-------------------|-------|
| 1 | ExperimentRound model | models.py, test_models.py | 3 |
| 2 | Experiment runner | experiment.py, test_experiment.py | 10 |
| 3 | ReportBuilder | report.py, test_report.py | 8 |
| 4 | CLI experiment subcommand | __main__.py, test_cli.py | 3 |
| 5 | Init update + full suite | __init__.py | 0 |
| **Total** | | **~8 files** | **~24 tests** |

## Testing Strategy

- All tests use `MockClient` — no real API calls
- `Experiment` tests create a mock client that returns predictable responses, run 2 rounds, and assert round metadata and file outputs
- `ReportBuilder` tests use hand-crafted experiment data dicts, asserting report structure and markdown output
- CLI test uses `patch("reasonbench.__main__.AnthropicClient")` pattern established in Phases 2-4
