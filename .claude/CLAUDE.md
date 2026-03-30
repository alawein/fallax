---
type: derived
source: org/governance-templates
sync: manual
sla: on-change
authority: canonical
audience: [agents, contributors]
last-verified: 2026-03-26
---

# reasonbench — Claude Code Configuration

## Project Context

Fallax — LLM Adversarial Reasoning Evaluation System. Multi-provider adversarial prompt generation, evaluation, clustering, and diagnostics.

## Quick Links

- Governance: [AGENTS.md](AGENTS.md)
- Shared governance guides: [../../../docs/shared/](../../../docs/shared/)

## Session Bootstrap

Before working:
1. Run `git log --oneline -5` to see recent work
2. Read `ROADMAP.md` for planned phases
3. Run `python -m pytest tests/` to verify current state

## Work Style

- Execute, do not plan. When asked to do something, do it.
- One change at a time. Make the smallest complete change, verify, then move to next.
- If stuck for >2 tool calls, stop and ask.

## Test Gates

After modifying code, run relevant tests before proceeding:
```bash
python -m pytest tests/
python -m mypy reasonbench/
python -m ruff check reasonbench/ tests/
```

## Environment

- Python 3.12+ (use `python`, not `python3`)
- Dependencies via `uv` (not pip)
- No credentials in chat; use env vars for API keys
