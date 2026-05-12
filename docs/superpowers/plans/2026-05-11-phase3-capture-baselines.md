---
type: canonical
source: none
sync: none
sla: none
---

# Phase 3 + 4: Capture Baselines and Publication Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture benchmark v1 baseline scores for `claude-sonnet-4-6` and `gpt-4o-mini`, then update README, CHANGELOG, and website to make a concrete research claim backed by evidence.

**Architecture:** `benchmarks/v1/prompts.jsonl` already contains 100 curated adversarial prompts. `benchmarks/v1/baselines.json` exists but has an empty `models` list. The `fallax baseline capture` CLI command runs the 100 prompts through the target model, scores via a judge model, and appends a `ModelBaseline` entry to `baselines.json`. This plan runs capture twice (once per provider), then updates three publication surfaces: README, CHANGELOG, and `website/index.html`.

**Prerequisites:**
- Phase 0 complete (all 372 tests passing, `uv run pytest tests/ -q` exits 0).
- `ANTHROPIC_API_KEY` environment variable set.
- `OPENAI_API_KEY` environment variable set.
- `openai` extra installed: `uv sync --extra openai` (needed for OpenAI client).

**Cost estimate:** 100 prompts × 2 LLM calls (model + judge) × 2 providers ≈ 400 API calls total. At small model pricing this is typically under $2 USD for both captures combined.

**Tech Stack:** Python 3.12+, fallax CLI, uv

---

## File Map

| Action | Path | What changes |
|---|---|---|
| Written by CLI | `benchmarks/v1/baselines.json` | `models` list populated with 2 entries |
| Gitignored artifact | `baseline_run.jsonl` | Raw evaluation output from capture run (do not commit) |
| Modify | `README.md` | Replace "review-tier" hedge; add benchmark results table and Providers section |
| Modify | `CHANGELOG.md` | Add v1.0.0 release entry |
| Modify | `website/index.html` | Update stats numbers to match real baseline scores |

---

## Task 1: Capture Anthropic baseline

**Files:**
- Written by CLI: `benchmarks/v1/baselines.json`

- [ ] **Step 1: Verify prerequisites**

Run: `uv run python -m fallax baseline status --version v1`

Expected:
```
No baselines recorded for v1.
```

Also confirm API key is set:
```bash
uv run python -c "import os; key=os.environ.get('ANTHROPIC_API_KEY',''); print('key set' if key else 'MISSING')"
```

Expected: `key set`. If `MISSING`, set the key before continuing:
```bash
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell
```

Also install the openai extra for the next task:
```bash
uv sync --extra openai
```

- [ ] **Step 2: Run baseline capture for Anthropic**

This will run 100 prompts through `claude-sonnet-4-6` as the evaluated model, with `claude-haiku-4-5-20251001` as the judge (faster and cheaper for judging):

```bash
uv run python -m fallax baseline capture \
  --version v1 \
  --model claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001 \
  --provider anthropic \
  --output baseline_anthropic.jsonl
```

Expected output (approximate — exact scores will vary):
```
Baseline captured (v1 / claude-sonnet-4-6)
  Overall score:  3.42
  Failure rate:   34.2%
  Prompts scored: 100
  Saved to:       benchmarks/v1/baselines.json
```

This takes 5–15 minutes depending on API rate limits.

If you see `FileNotFoundError: benchmark v1 not found`, the benchmarks directory path is wrong. Confirm `benchmarks/v1/prompts.jsonl` exists: `ls benchmarks/v1/`.

If you see an authentication error, re-check `ANTHROPIC_API_KEY`.

- [ ] **Step 3: Verify the entry was written**

Run: `uv run python -m fallax baseline status --version v1`

Expected:
```
Baselines (v1)
  Model                               Score   Fail%                  Captured
  ------------------------------------------------------------------------------
  claude-sonnet-4-6                    3.42   34.2%   2026-05-11T...
```

(Exact numbers will differ; what matters is that the row appears.)

- [ ] **Step 4: Check baselines.json directly**

Run: `python -c "import json; d=json.load(open('benchmarks/v1/baselines.json')); print(len(d['models']), 'models')`

Expected: `1 models`

- [ ] **Step 5: Add baseline_*.jsonl to .gitignore**

The raw run output files should not be committed. Open `.gitignore` and confirm (or add) this line:

```
baseline_*.jsonl
```

If it's not there, add it to the end of `.gitignore`.

- [ ] **Step 6: Commit the Anthropic baseline**

```bash
git add benchmarks/v1/baselines.json .gitignore
git commit -m "feat(benchmark): capture claude-sonnet-4-6 baseline for v1"
```

---

## Task 2: Capture OpenAI baseline

**Files:**
- Updated by CLI: `benchmarks/v1/baselines.json`

- [ ] **Step 1: Verify OpenAI API key**

```bash
uv run python -c "import os; key=os.environ.get('OPENAI_API_KEY',''); print('key set' if key else 'MISSING')"
```

Expected: `key set`. If `MISSING`:
```bash
$env:OPENAI_API_KEY = "sk-..."   # PowerShell
```

- [ ] **Step 2: Run baseline capture for OpenAI**

```bash
uv run python -m fallax baseline capture \
  --version v1 \
  --model gpt-4o-mini \
  --judge gpt-4o-mini \
  --provider openai \
  --output baseline_openai.jsonl
```

Expected output (approximate):
```
Baseline captured (v1 / gpt-4o-mini)
  Overall score:  4.17
  Failure rate:   41.7%
  Prompts scored: 100
  Saved to:       benchmarks/v1/baselines.json
```

- [ ] **Step 3: Verify both entries exist**

Run: `uv run python -m fallax baseline status --version v1`

Expected: two rows — one for `claude-sonnet-4-6`, one for `gpt-4o-mini`.

- [ ] **Step 4: Commit the OpenAI baseline**

```bash
git add benchmarks/v1/baselines.json
git commit -m "feat(benchmark): capture gpt-4o-mini baseline for v1"
```

---

## Task 3: Update README with research claim and results

**Files:**
- Modify: `README.md`

Replace the entire README content with the version below. Fill in the `<SCORE>` and `<FAIL%>` placeholders with the actual numbers from `fallax baseline status --version v1` before committing.

- [ ] **Step 1: Read actual baseline scores**

Run: `uv run python -m fallax baseline status --version v1`

Note the `Score` and `Fail%` columns for both models. You will substitute these into the README.

- [ ] **Step 2: Write the updated README**

Replace the full content of `README.md` with:

```markdown
# Fallax

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

**Fallax** evaluates language models on structured, multi-step reasoning tasks — logical
deduction, mathematical proof, causal inference, and compositional planning. It surfaces
failure modes that single-turn benchmarks miss by measuring step-level correctness, not
just final answers.

## Benchmark v1 Results

100 curated adversarial prompts across 25 reasoning failure templates.

| Model | Overall Score | Failure Rate |
|---|---|---|
| claude-sonnet-4-6 | <SCORE_ANTHROPIC> / 10 | <FAIL%_ANTHROPIC> |
| gpt-4o-mini | <SCORE_OPENAI> / 10 | <FAIL%_OPENAI> |

Higher score = more severe reasoning failures detected. See `benchmarks/v1/` for the
frozen prompt set, baseline data, and metadata.

## Why Fallax

- Measures step-level correctness, not just final answers.
- 25 adversarial templates across 6 failure categories (logic errors, assumption errors,
  constraint violations, generalization errors, ambiguity failures, multi-step breaks).
- Reproducible harness: seed-fixed prompt generation, versioned benchmark sets,
  deterministic scoring.
- Multi-provider: Anthropic, OpenAI, Gemini, and local models via Ollama.

## Features

- **Multi-step evaluation** — Tasks requiring chained reasoning, not pattern matching
- **Structured scoring** — 6-dimensional step-level correctness (not final-answer accuracy)
- **Failure taxonomy** — 6 categories, 10 types, 4 severity levels
- **Extensible harness** — Add reasoning domains via config
- **Benchmark versioning** — Immutable prompt sets for reproducible cross-model comparison
- **Baseline tracking** — Capture, compare, and regress-check model scores over time

## Providers

| Provider | Extra | Env var |
|---|---|---|
| Anthropic (default) | `uv sync` | `ANTHROPIC_API_KEY` |
| OpenAI | `uv sync --extra openai` | `OPENAI_API_KEY` |
| Google Gemini | `uv sync --extra gemini` | `GOOGLE_API_KEY` |
| Ollama (local) | `uv sync` (uses `requests`) | none — needs Ollama running |

## Tech Stack

- **Language:** Python 3.12+
- **Build:** `pyproject.toml` (`fallax 0.1.0`)
- **Testing:** pytest (372 tests)
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

- v1.1: Additional benchmark versions with expanded template sets
- v1.2: Reproducibility dashboard (web UI for visualizing experiment results)
- v2.0: Causal graph and program synthesis reasoning domains

## License

[MIT](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `uv run pytest tests/ -q` and
`uv run ruff check fallax/ tests/` before submitting.

## Ownership

- **Maintainer:** @alawein
- **Support:** GitHub Issues on this repository
```

Substitute the actual score values from Step 1 into `<SCORE_ANTHROPIC>`, `<FAIL%_ANTHROPIC>`,
`<SCORE_OPENAI>`, `<FAIL%_OPENAI>`.

- [ ] **Step 3: Verify no placeholder strings remain**

Run: `grep -n "SCORE\|FAIL%\|<" README.md`

Expected: no output (all placeholders replaced). If any remain, go back and fill them in.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): replace review-tier hedge with v1 benchmark results and Providers table"
```

---

## Task 4: Add v1.0.0 CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Read the current CHANGELOG**

Open `CHANGELOG.md` and note the format in use (likely Keep a Changelog / conventional).

- [ ] **Step 2: Add the v1.0.0 entry at the top of the changelog body**

Insert the following block immediately after the `# Changelog` heading (or equivalent top section), before any existing entries:

```markdown
## [1.0.0] — 2026-05-11

### Added

- **Benchmark v1** — 100 curated adversarial reasoning prompts across 25 templates.
  Captured baselines for `claude-sonnet-4-6` and `gpt-4o-mini`.
- **25 adversarial templates** — Original 10 plus 15 new patterns: temporal ordering,
  negation scope, base rate neglect, survivorship bias, modus tollens, scope creep,
  anchoring, false dichotomy, composition fallacy, conjunction fallacy, regression to mean,
  conditional probability, vacuous truth, infinite regress, and equivocation traps.
- **Multi-provider clients** — OpenAI, Gemini, and Ollama alongside the existing Anthropic
  client. `create_client(provider, ...)` factory for CLI usage.
- **Baseline CLI** — `fallax baseline capture|compare|status` subcommands for capturing
  model scores, detecting regressions, and tracking scores over time.
- **Experiment loop** — Multi-round `Experiment` orchestrator with structured reporting.
- **Analytics and intelligence** — `Analyzer`, `FailurePredictor`, `FailureClusterer`,
  `RootCauseExtractor`, and `SelfRepairTester` modules.
- **372 tests** across all components (up from 72 in Phase 1).
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): add v1.0.0 release entry"
```

---

## Task 5: Update website stats

**Files:**
- Modify: `website/index.html`

- [ ] **Step 1: Read the actual baseline scores (if not still open)**

Run: `uv run python -m fallax baseline status --version v1`

Note the scores for both models.

- [ ] **Step 2: Find the stats section in website/index.html**

Search for the stats numbers in the file:

```bash
grep -n "stat\|num\|372\|25\|100\|score" website/index.html | head -20
```

This will show which lines contain the stats numbers.

- [ ] **Step 3: Update the stats numbers**

The stats section in the HTML typically looks like:

```html
<div class="stats">
  <div class="stat"><div class="num">25</div><div class="label">Templates</div></div>
  <div class="stat"><div class="num">100</div><div class="label">Benchmark Prompts</div></div>
  <div class="stat"><div class="num">372</div><div class="label">Tests</div></div>
  <div class="stat"><div class="num">2</div><div class="label">Model Baselines</div></div>
</div>
```

Verify these numbers are accurate:
- Templates: 25 ✓
- Benchmark Prompts: 100 ✓
- Tests: 372 (confirm with `uv run pytest tests/ -q --co 2>&1 | tail -1`)
- Model Baselines: 2 ✓

If any number in the HTML differs from reality, correct it.

- [ ] **Step 4: Commit**

```bash
git add website/index.html
git commit -m "docs(website): sync stats with v1 benchmark results"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run the full test suite one last time**

Run: `uv run pytest tests/ -q --no-cov`

Expected: 372 passed, 0 failed, 0 errors.

- [ ] **Step 2: Confirm baseline status**

Run: `uv run python -m fallax baseline status --version v1`

Expected: two model rows with real scores.

- [ ] **Step 3: Check ruff and mypy are still clean**

Run: `uv run ruff check fallax/ tests/ && uv run mypy fallax/`

Expected: `All checks passed!` followed by `Success: no issues found in 26 source files`.

- [ ] **Step 4: Tag v1.0.0**

```bash
git tag v1.0.0
git push origin main --tags
```

---

## Self-Review

**Spec coverage:**
- Capture baselines for 2+ models (claude-sonnet-4-6, gpt-4o-mini) — Tasks 1 and 2 ✓
- README replaces "review-tier" with research claim and Providers table — Task 3 ✓
- CHANGELOG v1.0 entry — Task 4 ✓
- Website stats match baselines — Task 5 ✓
- Final gate: all tests green, ruff/mypy clean — Task 6 ✓

**Placeholder scan:** Task 3 Step 2 has `<SCORE_ANTHROPIC>` etc. — these are explicit
fill-in-from-Step-1 instructions, not forgotten placeholders. Step 3 has a grep to verify
all are replaced before commit. ✓

**Type consistency:** No code changes in this plan; no type drift possible. ✓

**Spec requirement check:**
- `benchmarks/v1/metadata.json` — already exists and correct; no task needed ✓
- `.gitignore` for `baseline_*.jsonl` — Task 1 Step 5 ✓
- Git tag v1.0.0 — Task 6 Step 4 ✓
