---
type: canonical
source: none
sync: none
sla: none
authority: canonical
audience: [agents, contributors, maintainers]
last-verified: 2026-04-16
---

# AGENTS — Fallax

## Workspace identity

Fallax is a Python reasoning-evaluation toolkit built around `reasonbench/`.

## Directory structure

- `reasonbench/`: primary source
- `benchmarks/`: benchmark definitions and datasets
- `dashboard/`: results UI surface
- `tests/`: required verification
- `docs/`: repo-local documentation

## Governance rules

1. Use `uv` as the primary environment workflow.
2. Keep public evaluation schemas stable unless explicitly versioned.
3. Maintain deterministic benchmark behavior.
4. Do not commit transient benchmark artifacts or secrets.
5. Comments should explain scoring and taxonomy behavior clearly.

## Code conventions

- Type hints and accurate docstrings on public surfaces
- Conventional commits only
- Add tests when evaluation behavior changes

## Build and test commands

```bash
uv sync --all-extras
python -m pytest tests/
python -m ruff check reasonbench/ tests/
python -m mypy reasonbench/
```
