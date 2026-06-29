# Fallax

Status:      active
Category:    research
Owner:       alawein
Visibility:  private
Purpose:     LLM adversarial reasoning evaluation system and benchmarking surface.
Next action: continue

## Purpose

Fallax evaluates language models on structured, multi-step reasoning tasks:
logical deduction, mathematical proof, causal inference, and compositional planning.
It scores step-level correctness (not final-answer accuracy) across 25 adversarial
templates in six failure categories. Internal research and model-comparison workflows
consume the harness, frozen benchmark sets, and baseline capture tooling.

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
- Alembiq and portfolio research workflows comparing reasoning failure modes
- Benchmark v1 baselines referenced in cross-model comparison reports

## Release and versioning

- Version source: `pyproject.toml` (`fallax 0.1.0`)
- Publish mode: private GitHub repo; PyPI publish not configured
- Benchmark sets are versioned under `benchmarks/v1/`; prompt changes require a new version directory
- Changelog: [CHANGELOG.md](CHANGELOG.md)
