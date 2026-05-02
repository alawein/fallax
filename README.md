# Fallax

## Public value

Fallax is a review-tier research repo until its core purpose, active workflow,
and publication posture are explicit. If promoted publicly, frame it around a
clear technical question, the reproducible implementation surface, and the
evidence it produces.

## Publication boundaries

- Confirm whether this is active research, a prototype, or an archive.
- Add or verify license and ownership before public promotion.
- Keep generated outputs, local data, secrets, and unpublished notes out of
  public examples unless intentionally sanitized.
- Prefer a small reproducible demo over broad undocumented scope.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Fallax** evaluates language models on structured, multi-step reasoning tasks — logical deduction, mathematical proof, causal inference, and compositional planning. It surfaces failure modes that single-turn benchmarks miss.

## Why Fallax

- Measures step-level correctness, not just final answers.
- Configurable domains and task packs to mirror real workloads.
- Reproducible harness with dashboards for regression tracking.

## Features

- **Multi-step evaluation** — Tasks requiring chained reasoning, not pattern matching
- **Structured scoring** — Step-level correctness, not just final-answer accuracy
- **Extensible harness** — Add new reasoning domains via config
- **Dashboard** — Visual results explorer
- **Benchmarks** — Performance regression tracking

## Tech Stack

- **Language:** Python 3.12+
- **Build:** pyproject.toml (`reasonbench`)
- **Testing:** pytest
- **Linting:** ruff, mypy

## Quick Start

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e .
pytest tests/
```

### Alternative installs

```bash
# Using uv
uv venv
uv pip install -e .
uv run pytest tests/

# Optional GPU extras (if available)
pip install -e .[gpu]

# Lint/type
ruff check .
mypy reasonbench
```

## Make/Task Shortcuts

```bash
make test          # fast suite
make test-all      # full suite
make lint          # ruff + mypy
make format        # formatters
```

## Project Structure

```text
fallax/
├── reasonbench/     # core evaluation engine
├── benchmarks/      # performance benchmarks
├── dashboard/       # results visualization
├── tests/           # pytest suite
├── website/         # project site
├── docs/            # documentation
└── pyproject.toml   # package config (reasonbench 0.1.0)
```

## Roadmap

- **Near term:** deterministic scoring fixtures, dataset versioning, richer domain configs
- **Mid term:** new benchmarks (causal graphs, program synthesis), reproducibility dashboard

## TODO

- [ ] Publish task schema examples in `docs/`
- [ ] Add seed-based determinism toggle for all evaluators
- [ ] Benchmark harness against public LLM baselines

## License

[MIT](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Keep tests green and run ruff/mypy before submitting.

## Ownership

- **Maintainer:** @alawein
- **Support:** GitHub Issues on this repository
