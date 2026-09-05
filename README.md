# Fallax

> Inspect reasoning steps across a fixed prompt taxonomy and keep records for
> comparing model releases.

## Purpose

Fallax evaluates intermediate reasoning as well as the final answer. Benchmark
v1 fixes 100 prompts from 25 adversarial templates, covering 10 failure types in
six broad categories. Each run stores judge assessments of answer correctness,
flawed steps, assumptions, and counterfactual validity. Its composite score also
uses answer-string disagreement.

This is not another leaderboard. Scores depend on the benchmark, provider, and
judge model. They show how a model fails on this prompt set; they do not rank
general model capability.

Fallax is for ML researchers and engineers comparing reasoning failures across
model releases. It is maintained by Meshal Alawein.

### What it is

Fallax is a Python CLI for adversarial reasoning evaluation. It generates or
loads prompts, runs supported model providers, asks a judge model to apply five
validators, and writes structured results for analysis and comparison.

Providers include Anthropic, OpenAI, OpenRouter, Gemini, and Ollama. Remote
providers require their API key in the environment; Ollama runs locally.

### What it is not

Fallax does not measure general model capability, prove that a model's reasoning
is faithful, or remove judge-model bias. A score is meaningful only with its
benchmark version, provider, served model, and judge provenance.

## Install

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/alawein/fallax.git
cd fallax
uv sync
uv run python -m fallax baseline status --version v1
```

The last command reads the recorded v1 baselines without calling a model
provider. It prints one row each for `anthropic/claude-sonnet-4.6` and
`openai/gpt-4o-mini`.

## Commands

### Analyze the offline example

```bash
uv run python -m fallax analyze examples/fixtures/offline-results.jsonl
```

This command reads and analyzes the committed demonstration fixture without
credentials or network access. It exercises JSONL loading, result validation,
and analysis reporting. The fixture is illustrative, not a benchmark result or
a model-quality claim.

### Run a provider benchmark

Install the provider extra you need, then name both the evaluated model and the
judge:

```bash
uv sync --extra openai
uv run python -m fallax benchmark \
  --version v1 \
  --models openai/gpt-4o-mini \
  --judge anthropic/claude-4.5-haiku-20251001 \
  --provider openrouter \
  --output benchmark_results.jsonl
```

This example requires `OPENROUTER_API_KEY`. For another provider, set its matching
environment variable before the run:
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or
`GOOGLE_API_KEY`.

## Benchmark versions

| Version | Prompts | Templates | Failure types | Categories | Prompt digest |
| --- | ---: | ---: | ---: | ---: | --- |
| v1 | 100 | 25 | 10 | 6 | `58abd983...d1e8ab` |

`benchmarks/v1/metadata.json` records seed 42 and the full SHA-256 digest.
Changes to a frozen prompt set require a new version directory.

## Baseline provenance

The repository records one v1 run for each of two evaluated models. Both were
captured through OpenRouter and judged by
`anthropic/claude-4.5-haiku-20251001` on 2026-05-13. The stored composite score
is a failure-severity score, so higher values mean a more severe failure.

| Evaluated model | Composite score | Failure rate |
| --- | ---: | ---: |
| `anthropic/claude-sonnet-4.6` | 6.77 | 82% |
| `openai/gpt-4o-mini` | 8.14 | 91% |

These are recorded baselines, not a current model comparison. Re-run the same
benchmark with explicit provider and served-model provenance before drawing a
new conclusion.

## Architecture

See [docs/architecture.md](docs/architecture.md) for module boundaries and data
flow, and [docs/architecture/topology.md](docs/architecture/topology.md) for
on-disk layout.

## Reproducibility

Benchmark metadata stores the prompt count, generation parameters, failure
taxonomy, and prompt-file digest. Baselines store the evaluated model, served
judge model, provider, capture time, category scores, and failure-type scores.

Run `uv run pytest --cov=fallax --cov=dashboard --cov-fail-under=90` to
check the current checkout. A local test run does not constitute a new provider
evaluation.

## Consumers

Used by ML researchers and engineers comparing reasoning failures across model
releases via the CLI.

## Release and versioning

- Version source: `pyproject.toml` (`fallax` 0.1.0, pre-1.0 package state)
- The historical `v1.0.0` git tag does not determine the current package version
  or GitHub Release status; see [docs/deployment.md](docs/deployment.md) before
  publishing a release, and preserve the documented `served_model` provenance
  limits.
- Benchmark sets are versioned under `benchmarks/v1/`; changes to a frozen
  prompt set require a new version directory.
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## Docs map

- [Architecture](docs/architecture.md)
- [Repository topology](docs/architecture/topology.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Citation metadata](CITATION.cff)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)

## License

Fallax is maintained by Meshal Alawein and released under the [MIT License](LICENSE).
