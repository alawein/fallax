---
type: canonical
source: none
sync: none
sla: none
authority: canonical
audience: [agents, contributors, maintainers]
last-verified: 2026-09-06
last_updated: 2026-09-06
---

# AGENTS · Fallax

## Workspace identity

Fallax is a Python reasoning-evaluation toolkit built around `fallax/`.

## Directory structure

- `fallax/`: primary source
- `benchmarks/`: benchmark definitions and datasets
- `dashboard/`: results UI surface
- `tests/`: required verification
- `docs/`: repo-local documentation

- `website/`: public-facing project surface

## Governance rules

1. Use `uv` as the primary environment workflow.
2. Keep public evaluation schemas stable unless explicitly versioned.
3. Maintain deterministic benchmark behavior.
4. Do not commit transient benchmark artifacts or secrets.
5. Comments should explain scoring and taxonomy behavior clearly.

6. Provider API keys live in environment variables only.

7. Keep the scoring, clustering, and taxonomy surfaces legible instead of collapsing everything into a single opaque score.

## Simplicity defaults

- Make the smallest change that satisfies the acceptance criteria.
- Prefer direct functions and plain data structures.
- No class when a function suffices. No framework for one implementation.
- No shared abstraction before real duplication exists.
- Prefer the standard library or an existing dependency.
- Avoid factories, registries, adapters, plugins, and config layers without multiple real consumers.
- Keep control flow direct. Use early returns when clearer. Keep errors explicit.
- Comments explain invariants, assumptions, and failure modes. Delete dead code instead of commenting it out.
- Keep pull requests single-purpose. Stop when tests and acceptance criteria pass. Do not rewrite adjacent working code without a stated need.

## Code conventions

- Type hints and accurate docstrings on public surfaces
- Conventional commits only
- Add tests when evaluation behavior changes

## Build and test commands

```bash
uv sync --all-extras
python -m pytest tests/
python -m ruff check fallax/ tests/
python -m mypy fallax/
```
