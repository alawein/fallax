---
type: canonical
source: none
sync: none
sla: none
authority: canonical
audience: [contributors, maintainers, agents]
last-verified: 2026-04-16
---

# SSOT

## Current state

- `fallax` is a Python-first reasoning-evaluation repo centered on the
  `reasonbench` package.
- The runtime surface includes the evaluation engine, benchmark suites, a
  dashboard, and a small project site.
- Docs and repo surfaces are now aligned with the benchmark and scoring
  semantics described in the current README and docs lane.

## Active boundaries

- `reasonbench/` is the canonical evaluation-engine surface.
- `benchmarks/` holds reproducibility and regression material.
- `dashboard/` and `website/` are presentation layers, not the scoring source
  of truth.
- `docs/` carries architecture, deployment, roadmap, and troubleshooting notes.

## Active decisions

- Step-level correctness matters more than single final-answer scoring.
- Benchmark semantics must stay explicit and reproducible.
- New domains or task packs should land with docs and tests, not as silent
  config drift.

## Validation baseline

```bash
pytest tests/
ruff check .
mypy reasonbench
```
