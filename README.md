# Fallax

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

**Fallax** evaluates language models on structured, multi-step reasoning tasks: logical deduction, mathematical proof, causal inference, and compositional planning. It surfaces failure modes that single-turn benchmarks miss by measuring step-level correctness, not just final answers.

## Benchmark v1 Results

100 curated adversarial prompts across 25 reasoning failure templates. Both baselines captured via OpenRouter and judged by `anthropic/claude-haiku-4.5` for cross-family grading consistency. Scores are on a 0-10 step-failure scale; **lower is better**. Failure rate is the fraction of prompts scoring at or above 4.

| Model | Overall Score | Failure Rate |
|---|---|---|
| `anthropic/claude-sonnet-4.6` | 6.77 | 82.0% |
| `openai/gpt-4o-mini` | 8.14 | 91.0% |

Both models fail at high absolute rates by design; the v1 prompts are adversarial and target reasoning failure modes that single-turn accuracy hides. Sonnet 4.6 outperforms gpt-4o-mini on this benchmark by ~14% on overall score and 9 percentage points on failure rate. See `benchmarks/v1/baselines.json` for per-category and per-failure-type breakdowns.

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
| OpenRouter | `uv sync --extra openai` | `OPENROUTER_API_KEY` |
| Google Gemini | `uv sync --extra gemini` | `GOOGLE_API_KEY` |
| Ollama (local) | `uv sync` (uses `requests`) | none, needs Ollama running |

OpenRouter is an OpenAI-API-compatible gateway: one key unlocks Claude, GPT, Gemini, and many open-weight models. Use provider-prefixed model slugs, for example `--provider openrouter --model anthropic/claude-sonnet-4.6 --judge anthropic/claude-haiku-4.5`. Baselines captured through a gateway record the resolved model identifier in `served_model` for provenance; direct-provider baselines record the same string as `model_name`. (Known limitation in v1.0.0: `served_model` currently records the last model the client called, which is the judge rather than the model under test. The `model_name` field is authoritative for the model that was actually evaluated; the `served_model` field will be corrected in v1.0.1.)

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
├── tests/           # pytest suite
├── website/         # project site
└── pyproject.toml   # package config
```

## Roadmap

- v1.0.1: Fix `served_model` provenance bug (currently records judge model, not model under test).
- v1.1: Add baselines for Gemini and DeepSeek R1; document cross-judge sensitivity.
- v1.2: Reproducibility dashboard (web UI for visualizing experiment results).
- v2.0: Causal graph and program synthesis reasoning domains.

## License

[MIT](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `uv run pytest tests/ -q` and `uv run ruff check fallax/ tests/` before submitting.

## Ownership

- **Maintainer:** @alawein
- **Support:** GitHub Issues on this repository
