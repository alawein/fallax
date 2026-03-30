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

## Installation

```bash
pip install -e .
```

## License

[MIT](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.