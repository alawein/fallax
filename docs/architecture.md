---
type: canonical
source: none
sync: none
sla: none
---

# Architecture

## Components

- `reasonbench/` — core evaluation engine, task loaders, scorers, and configuration.
- `benchmarks/` — curated benchmark bundles with tasks, gold answers, and configs.
- `dashboard/` — visualization layer for runs and regressions.
- `website/` — marketing/docs site.
- `tests/` — pytest suites (unit + integration).

## Data Model

- **Task** — multi-step prompt/instruction with expected intermediate states.
- **Run** — model invocation with recorded responses and parsed steps.
- **Score** — structured scoring record (step correctness, final correctness, metadata).

## Execution Flow

1. Select benchmark bundle (config).
2. Load tasks into `reasonbench`.
3. Run model adapter to produce responses.
4. Parse and score responses step-by-step.
5. Emit results to JSON; dashboard consumes these artifacts.

## Extensibility

- Add a domain: create config + tasks under `benchmarks/<domain>/`.
- Add a model: implement adapter conforming to the interface in `reasonbench/models`.
- Add metrics: extend scorer to emit new dimensions; update dashboard schemas accordingly.

## Reproducibility

- Pin datasets and configs; version benchmark bundles.
- Use seeds for any sampling; record seeds in result metadata.
- Keep runs and scores under a dedicated `outputs/` directory (gitignored).
