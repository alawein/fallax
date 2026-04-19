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

## Module layout

```text
reasonbench/
  models.py          # Pydantic data models
  templates.py       # Adversarial prompt templates (25 patterns)
  generator.py       # Prompt generation from templates
  evolver.py         # Template evolution and mutation
  client.py          # Multi-provider LLM client abstraction
  clients/           # Provider-specific implementations
  evaluator.py       # Response evaluation pipeline
  scoring.py         # Scoring framework (6 dimensions)
  analyzer.py        # Statistical analysis
  clusterer.py       # Response clustering (scikit-learn)
  root_cause.py      # Root cause diagnostics
  predictor.py       # Failure prediction
  repair.py          # Prompt repair suggestions
  pipeline.py        # End-to-end orchestration
  runner.py          # CLI runner
  experiment.py      # Experiment management
  benchmark.py       # Versioned benchmark suite
  report.py          # Report generation
  storage.py         # Result persistence
  taxonomy.py        # Failure taxonomy
  validators.py      # Input validation
  data/              # Static data (taxonomies, datasets)
```

## Key concepts

- **Templates** — adversarial reasoning patterns (syllogistic, temporal, modal, etc.).
- **Evolution** — templates mutate via LLM-driven rewriting for diversity.
- **Scoring** — 6-dimensional evaluation (logical validity, premise accuracy, and related axes).
- **Clustering** — groups failure modes via scikit-learn for pattern detection.
- **Benchmarks** — versioned datasets under `benchmarks/` for reproducibility.

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
