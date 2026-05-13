# Fallax

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

**Fallax** evaluates language models on structured, multi-step reasoning tasks: logical deduction, mathematical proof, causal inference, and compositional planning. It surfaces failure modes that single-turn benchmarks miss by measuring step-level correctness, not just final answers.

## Benchmark v1 Results

100 curated adversarial prompts across 25 reasoning failure templates.

| Model | Overall Score | Failure Rate |
|---|---|---|
| claude-sonnet-4-6 | pending | pending |
| gpt-4o-mini | pending | pending |

Baselines pending; run `fallax baseline capture --version v1 --model <model> --judge <judge>` to populate. See `benchmarks/v1/` for the frozen prompt set and metadata.

## Why Fallax

- Measures step-level correctness, not just final answers.
- 25 adversarial templates across 6 failure categories (logic errors, assumption errors, constraint violations, generalization errors, ambiguity failures, multi-step breaks).
- Reproducible harness: seed-fixed prompt generation, versioned benchmark sets, deterministic scoring.
- Multi-provider: Anthropic, OpenAI, Gemini, and local models via Ollama.

## Features

- **Multi-step evaluation:** tasks requiring chained reasoning, not pattern matching
- **Structured scoring:** 6-dimensional step-level correctness (not final-answer accuracy)
- **Failure taxonomy:** 6 categories, 10 types, 4 severity levels
- **Extensible harness:** add reasoning domains via config
- **Benchmark versioning:** immutable prompt sets for reproducible cross-model comparison
- **Baseline tracking:** capture, compare, and regress-check model scores over time

## Providers

| Provider | Extra | Env var |
|---|---|---|
| Anthropic (default) | `uv sync` | `ANTHROPIC_API_KEY` |
| OpenAI | `uv sync --extra openai` | `OPENAI_API_KEY` |
| Google Gemini | `uv sync --extra gemini` | `GOOGLE_API_KEY` |
| Ollama (local) | `uv sync` (uses `requests`) | none, needs Ollama running |

## Tech Stack

- **Language:** Python 3.12+
- **Build:** `pyproject.toml` (`fallax 0.1.0`)
- **Testing:** pytest
- **Linting:** ruff, mypy

## Quick Start

```bash
# Install
uv sync                         # core + dev deps
uv sync --extra openai          # add OpenAI provider
uv sync --extra dashboard       # add dashboard server

# Run tests
uv run pytest tests/ -q

# Evaluate a model
uv run python -m fallax run \
  --models claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001 \
  --output results.jsonl

# Benchmark against v1
uv run python -m fallax baseline capture \
  --version v1 \
  --model claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001

# Compare against baseline
uv run python -m fallax baseline compare \
  --version v1 \
  --model claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001

# Analyze results
uv run python -m fallax analyze results.jsonl
```

## Project Structure

```text
fallax/
├── fallax/          # core evaluation engine (taxonomy, templates, scoring, pipeline)
├── fallax/clients/  # provider-specific LLM clients (anthropic, openai, gemini, ollama)
├── benchmarks/v1/   # frozen benchmark: prompts.jsonl, baselines.json, metadata.json
├── dashboard/       # FastAPI results explorer
├── tests/           # 372-test pytest suite
├── website/         # project site
└── pyproject.toml   # package config
```

## Roadmap

- v1.1: Capture baselines for claude-sonnet-4-6 and gpt-4o-mini; tag v1.0.0
- v1.2: Reproducibility dashboard (web UI for visualizing experiment results)
- v2.0: Causal graph and program synthesis reasoning domains

## License

[MIT](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `uv run pytest tests/ -q` and `uv run ruff check fallax/ tests/` before submitting.

## Ownership

- **Maintainer:** @alawein
- **Support:** GitHub Issues on this repository
