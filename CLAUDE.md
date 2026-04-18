---
type: canonical
source: none
sync: none
sla: none
authority: canonical
audience: [ai-agents, contributors]
last_updated: 2026-04-15
last-verified: 2026-04-15
---

# CLAUDE.md — Fallax

## Workspace identity

Fallax is a reasoning-evaluation repo. The work here is about exposing
multi-step failure modes in LLM outputs, not just recording final-answer
accuracy. Keep the evaluation logic inspectable at the step, taxonomy, and
benchmark level.

Shared voice and research-writing contract:

- <https://github.com/alawein/alawein/blob/main/docs/style/VOICE.md>
- <https://github.com/alawein/alawein/blob/main/prompt-kits/AGENT.md>

## Directory structure

- `reasonbench/`: canonical evaluation engine
- `benchmarks/`: benchmark definitions and datasets
- `dashboard/`: visualization and operator-facing results surface
- `website/`: public-facing project surface
- `tests/`: required verification
- `docs/`: repo-local documentation

## Governance rules

1. Use `uv` as the primary environment and dependency workflow.
2. Preserve schema stability for public evaluation outputs unless the change is
   explicitly versioned.
3. Keep benchmark behavior deterministic under the pinned environment.
4. Provider API keys live in environment variables only.
5. Do not commit transient benchmark artifacts or scratch result stores.
6. Keep the scoring, clustering, and taxonomy surfaces legible instead of
   collapsing everything into a single opaque score.

## Code conventions

- Public Python behavior lives under `reasonbench/`.
- Comments explain benchmark semantics, scoring dimensions, or failure-taxonomy
  logic.
- Prefer explicit benchmark contracts over convenience wrappers that hide what
  is being measured.

## Build and test commands

```bash
uv sync --all-extras
python -m pytest tests/
python -m ruff check reasonbench/ tests/
python -m mypy reasonbench/
```

Ruff selects `E, F, I, UP, B, SIM`; line length 88. See `docs/architecture.md` for the `reasonbench/` module layout and key concepts.

## Gotchas

- Provider API keys must be set as environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) — never commit or paste into chat.
- The `all-providers` extra installs all LLM client dependencies; individual provider extras are also available.
- Dashboard requires the `dashboard` extra (`uv sync --extra dashboard`).
- `.coverage` and `.benchmarks/` are generated artifacts — keep them out of commits.
