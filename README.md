---
type: canonical
source: none
sync: none
sla: none
---

# Fallax

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

> *Multi-step reasoning evaluation and benchmark suite for LLMs.*

**Fallax** evaluates language models on structured, multi-step reasoning tasks — logical deduction, mathematical proof, causal inference, and compositional planning. Designed to surface failure modes that single-turn benchmarks miss.

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

## Installation

```bash
pip install -e .
```

## License

[MIT](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Ownership

- **Maintainer:** @alawein
- **Support:** GitHub Issues on this repository
