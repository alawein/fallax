---
type: canonical
owner: platform-engineering
last-reviewed: 2026-03-31
---

# Deployment and Release · fallax

Fallax is a research and benchmarking tool, not a deployed service. There is
no server, no container, and no production environment to operate. All
evaluation runs execute locally against the provider APIs you configure.

## Running Evaluations Locally

Follow the Quick Start in the root [README](../README.md):

```bash
# Install deps (core + all extras)
uv sync --all-extras

# Run a model evaluation
uv run python -m fallax run \
  --models claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001 \
  --output results.jsonl

# Capture a baseline against the v1 benchmark
uv run python -m fallax baseline capture \
  --version v1 \
  --model claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001

# Compare against a captured baseline
uv run python -m fallax baseline compare \
  --version v1 \
  --model claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001
```

Required environment variables:

| Provider | Variable |
|---|---|
| Anthropic (default) | `ANTHROPIC_API_KEY` |
| OpenAI / OpenRouter | `OPENAI_API_KEY` or `OPENROUTER_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |

## Optional Dashboard

The FastAPI results explorer is available locally only:

```bash
uv sync --extra dashboard
uv run python -m fallax dashboard
```

It does not require any deployment infrastructure.

## Versioning

Fallax follows [Semantic Versioning](https://semver.org/). The current package
version is in `pyproject.toml`. Releases are tagged in git; there is no
release pipeline or package registry publication at this time.
