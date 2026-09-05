# Fallax

Status:      active
Category:    lab
Owner:       alawein
Visibility:  public
Purpose:     LLM adversarial reasoning evaluation harness; live at https://fallax.online.
Next action: continue

## Purpose

Fallax is a CLI harness that scores language models on step-level reasoning
correctness, not final-answer accuracy, across 25 adversarial templates in six
failure categories. It is for researchers comparing reasoning-failure modes
across model releases. Unlike accuracy-only benchmarks, its judge model scores
each intermediate step and classifies the failure by type. It does not measure
general model capability or replace task-specific evaluation suites.

- Lifecycle: active
- Verification date: 2026-08-28
- Scope: CLI evaluation harness, versioned benchmark prompts, and baseline capture/compare tooling
- Live: https://fallax.online

## Install

```bash
git clone https://github.com/alawein/fallax.git
cd fallax
uv sync                         # core + dev deps
uv sync --extra openai          # OpenAI or OpenRouter provider
uv sync --extra gemini          # Google Gemini provider
uv sync --extra dashboard       # FastAPI results explorer
```

Provider API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or
`GOOGLE_API_KEY` as needed. Ollama runs locally with no key.

## Commands

The commands below are run from the repository root. Validate the current
checkout with `uv run pytest -q`; test totals intentionally are not recorded
here because they change as coverage grows. `run`, `baseline capture`, and
`baseline compare` need a provider API key. `analyze` can run entirely offline
against the committed demonstration fixture.

### Offline end-to-end demonstration

No credentials or network access are needed to read and analyze the committed
example results:

```bash
uv run python -m fallax analyze examples/fixtures/offline-results.jsonl
```

The command exercises Fallax's JSONL loading, result validation, and analysis
reporting. The fixture is intentionally small and illustrative; it is not a
benchmark result or a model-quality claim.

```bash
uv run pytest tests/ -q
uv run ruff check fallax/ tests/

uv run python -m fallax run \
  --models claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001 \
  --output results.jsonl

uv run python -m fallax baseline capture \
  --version v1 \
  --model claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001

uv run python -m fallax baseline compare \
  --version v1 \
  --model claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001

uv run python -m fallax analyze results.jsonl
```

Benchmark v1 holds 100 curated prompts in `benchmarks/v1/`; baseline scores live in
`benchmarks/v1/baselines.json`.

## Architecture

```
fallax/
├── fallax/              # evaluation engine (taxonomy, templates, scoring, pipeline)
│   └── clients/         # Anthropic, OpenAI, Gemini, Ollama adapters
├── benchmarks/v1/       # frozen prompts, baselines, metadata
├── dashboard/           # FastAPI results explorer
├── tests/               # pytest suite
├── website/             # project site
└── docs/                # architecture, deployment, roadmap
```

See [docs/architecture/topology.md](docs/architecture/topology.md) for on-disk layout and [docs/architecture.md](docs/architecture.md) for module boundaries and data flow.

## Docs map

- [docs/README.md](docs/README.md)
- [docs/architecture.md](docs/architecture.md)
- [SSOT.md](SSOT.md)
- [LESSONS.md](LESSONS.md)

## Consumers

- Internal model evaluation and regression checks before release
- Used by the alawein research workflows through the CLI
- Benchmark v1 baselines referenced in cross-model comparison reports

## Release and versioning

- Current package version: `0.1.0` in `pyproject.toml`; this is a pre-1.0 package state.
- Publish mode: public GitHub repository; no PyPI publication or GitHub Release is currently published.
- Historical tag: [`v1.0.0`](https://github.com/alawein/fallax/tree/v1.0.0) is an annotated Git tag, not a current GitHub Release. It predates current `main`; do not infer package or release status from the tag alone.
- Benchmark sets are versioned under `benchmarks/v1/`; prompt changes require a new version directory.
- Citation metadata: [CITATION.cff](CITATION.cff).
- Changelog: [CHANGELOG.md](CHANGELOG.md).

Before publishing a release, reconcile the package version, release notes, and
the historical tag's documented `served_model` provenance limitation.
