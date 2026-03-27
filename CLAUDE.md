---
type: canonical
source: none
sync: none
sla: none
---

<!-- Template: research-library v1.0.0 -->
<!-- Generated from _pkos governance templates. Do not edit the template sections -->
<!-- directly in consuming projects — update the template and re-sync instead.    -->

# CLAUDE.md — ReasonBench

## Repository Context

**Name:** ReasonBench
**Type:** research-library
**Purpose:** LLM Adversarial Reasoning Evaluation System. Generates adversarial
reasoning prompts, evaluates LLM responses across multiple providers (Anthropic,
OpenAI, Gemini, Ollama), and produces structured analysis with clustering,
scoring, and root-cause diagnostics.

---

## Build and Test Commands

```bash
# Install dependencies (use uv, not pip)
uv sync --all-extras

# Run tests
python -m pytest tests/

# Run single test
python -m pytest tests/test_specific.py -k "test_name"

# Type checking
python -m mypy reasonbench/

# Linting
python -m ruff check reasonbench/ tests/

# Format
python -m ruff format reasonbench/ tests/
```

---

## Architecture

```text
reasonbench/
  models.py          # Pydantic data models
  templates.py       # Adversarial prompt templates (25 patterns)
  generator.py       # Prompt generation from templates
  evolver.py         # Template evolution and mutation
  client.py          # Multi-provider LLM client abstraction
  clients/           # Provider-specific implementations
  evaluator.py       # Response evaluation pipeline
  scoring.py         # Scoring framework (6 dimensions)
  analyzer.py        # Statistical analysis
  clusterer.py       # Response clustering (scikit-learn)
  root_cause.py      # Root cause diagnostics
  predictor.py       # Failure prediction
  repair.py          # Prompt repair suggestions
  pipeline.py        # End-to-end orchestration
  runner.py          # CLI runner
  experiment.py      # Experiment management
  benchmark.py       # Versioned benchmark suite
  report.py          # Report generation
  storage.py         # Result persistence
  taxonomy.py        # Failure taxonomy
  validators.py      # Input validation
  data/              # Static data (taxonomies, datasets)
benchmarks/          # Versioned benchmark datasets
dashboard/           # FastAPI web UI for visualization
tests/               # Test suite
docs/                # Documentation
```

---

## Key Concepts

- **Templates:** Adversarial reasoning patterns (syllogistic, temporal, modal, etc.)
- **Evolution:** Templates mutate via LLM-driven rewriting for diversity
- **Scoring:** 6-dimensional evaluation (logical validity, premise accuracy, etc.)
- **Clustering:** Groups failure modes via scikit-learn for pattern detection
- **Benchmarks:** Versioned datasets in `benchmarks/` for reproducibility

---

## Conventions

- **Python version:** 3.12+
- **Dependency management:** `uv` (not pip)
- **Type annotations:** Required on all public functions (mypy strict)
- **Linting:** ruff with select rules (E, F, I, UP, B, SIM)
- **Line length:** 88 characters
- **Test framework:** pytest with coverage
- **File naming:** `snake_case.py`

---

## Gotchas

- Provider API keys must be set as environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
- The `all-providers` extra installs all LLM client dependencies
- Dashboard requires the `dashboard` extra (`uv sync --extra dashboard`)
- `.coverage` and `.benchmarks/` are generated artifacts — don't commit them
