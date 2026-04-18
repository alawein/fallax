---
type: derived
source: org/governance-templates
sync: script
sla: on-change
authority: canonical
audience: [agents, contributors]
last-verified: 2026-04-16
---

# fallax — local Claude bootstrap

## Project Context

Fallax is a reasoning-evaluation repo. The work here is about exposing multi-step failure modes in LLM outputs, not just recording final-answer accuracy. Keep the evaluation logic inspectable at the step, taxonomy, and benchmark level.

## Authority

- Root [CLAUDE.md](CLAUDE.md) is authoritative for repo context and repo-specific constraints.
- Root [AGENTS.md](AGENTS.md) is authoritative for repo rules and operating boundaries.
- Shared voice contract: <https://github.com/alawein/alawein/blob/main/docs/style/VOICE.md>
- Workspace prompt kit: <https://github.com/alawein/alawein/blob/main/prompt-kits/AGENT.md>

## Before You Touch Code

1. Run `git log --oneline -5` to see recent work.
2. Read root `CLAUDE.md` for project-specific context.
3. Read root `AGENTS.md` if the task changes structure, process, tooling, or docs policy.
4. Read the shared voice contract and use the repo overlay that matches this surface.
5. Run the smallest relevant verification command before widening the change.

## Working Rules

- Execute on the smallest complete surface.
- Verify immediately after each meaningful change.
- If missing context blocks the work after two tool moves, stop and ask.
- Keep GitHub-facing `README.md` and `docs/README.md` frontmatter-free.
- Match the shared Alawein voice contract for docs, prompts, naming, comments, and math writing.
- Do not add secrets or hand-edit generated output to silence a failing check.

## Test Gates

After modifying code, run the relevant verification path before ending the session.

## Environment

- Git configured for LF (not CRLF).
- Python: use `python` (not `python3`).
- No credentials in chat; use `gh secret set` or `vercel env add` instead.
