---
type: canonical
source: none
sync: none
sla: none
---

<!-- Template: python-research-tooling v1.0.0 -->
<!-- Generated for governance parity across workspace repos. -->
---
type: normative
authority: canonical
audience: [agents, contributors, maintainers]
last-verified: 2026-03-30
---

# AGENTS — reasonbench

> **Status: Normative.** Do not modify without maintainer review.

This repository is a Python benchmark and reasoning-evaluation toolkit (`reasonbench`).

## Repository Scope

| Directory | Purpose | Governance level |
|-----------|---------|------------------|
| `reasonbench/` | Core library package | Primary |
| `dashboard/` | Optional API/UI service wiring | Stable |
| `tests/` | Test suite and regression fixtures | Primary |
| `benchmarks/` | Benchmark definitions and dataset wrappers | Secondary |
| `docs/` | Additional documentation | Secondary |

## Invariants (Must Always Hold)

1. `pytest` must pass before proposing behavior-impacting merges.
2. Public evaluation outputs should remain schema-stable unless explicitly versioned.
3. Requests/session handling changes must preserve timeout and error-path coverage.
4. No secrets or API keys in repository files.
5. Keep package dependencies minimal and pinned to avoid reproducibility drift.
6. Benchmark artifacts should remain deterministic under pinned environments.

## Agent Rules

- Read `AGENTS.md` and `CLAUDE.md` before changing files.
- Prefer minimal, targeted edits with explicit test evidence.
- Preserve Python runtime compatibility for `>=3.12` unless justified.
- Run `pytest` for touched modules; add regression tests where confidence is non-trivial.
- Run `ruff check .` for touched Python packages.
- Avoid changing benchmark fixture payloads unless required by evidence-backed behavior changes.
- Do not edit `uv.lock` without dependency intent and review comment.

## Test / Verification Requirements

- Baseline: `pytest` (uses project default testpaths).
- Type safety on touched modules: run `mypy reasonbench` where practical.
- If dashboard routes change, validate local run path documented in README.

## Repo-Specific Notes

- This repo does not currently include `.claude/AGENTS.md`; this file is authoritative.
- For release or behavior-surface changes, update `CHANGELOG.md` and `README.md`.
