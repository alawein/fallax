---
type: canonical
source: none
sync: none
sla: none
authority: canonical
audience: [ai-agents, contributors]
last_updated: 2026-09-06
last-verified: 2026-09-06
---

# CLAUDE.md · Fallax

Universal agent rules and simplicity defaults live in [AGENTS.md](AGENTS.md). Read that first.

## Claude-specific deltas

Shared voice and research-writing contract:

- <https://github.com/alawein/alawein/blob/main/docs/style/VOICE.md>
- <https://github.com/alawein/alawein/blob/main/prompt-kits/AGENT.md>

### Gotchas

- Provider API keys must be set as environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.); never commit or paste into chat.
- The `all-providers` extra installs all LLM client dependencies; individual provider extras are also available.
- Dashboard requires the `dashboard` extra (`uv sync --extra dashboard`).
- `.coverage` and `.benchmarks/` are generated artifacts; keep them out of commits.

Ruff selects `E, F, I, UP, B, SIM`; line length 88. See `docs/architecture.md` for module layout.
