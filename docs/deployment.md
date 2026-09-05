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
version is `0.1.0` in `pyproject.toml`; treat the project as pre-1.0 until a
release is deliberately prepared. There is no release pipeline, package
registry publication, or GitHub Release at this time.

A historical annotated Git tag,
[`v1.0.0`](https://github.com/alawein/fallax/tree/v1.0.0), exists but is not a
GitHub Release and does not represent the current package version. Any future
release should reconcile that tag, the package version, its notes, and the
known `served_model` provenance limitation before publication.
