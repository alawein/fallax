# ReasonBench Roadmap

Planned enhancements beyond the core 5-phase implementation.

---

## Phase 6: Multi-Provider LLM Support

**Goal:** Evaluate models from any provider, not just Anthropic.

**Approach:** Implement the existing `LLMClient` protocol for additional providers. The protocol requires a single method: `complete(prompt: str, *, model: str) -> str`.

**Planned clients:**

| Client | Provider | Models |
|--------|----------|--------|
| `OpenAIClient` | OpenAI | GPT-4o, GPT-4.1, o3, o4-mini |
| `GeminiClient` | Google | Gemini 2.5 Pro/Flash |
| `OllamaClient` | Local | Llama, Mistral, Qwen via Ollama |

**Key decisions:**
- Each client lives in its own module (`reasonbench/clients/openai.py`, etc.)
- `AnthropicClient` moves to `reasonbench/clients/anthropic.py` (re-export from `client.py` for backwards compat)
- Factory function `create_client(provider, **kwargs) -> LLMClient` for CLI usage
- CLI `--provider` flag alongside `--models` to select client

**Dependencies:** `openai`, `google-genai` (optional extras in `pyproject.toml`)

---

## Phase 7: Dashboard UI

**Goal:** Web interface to visualize experiment reports — score trends, failure clusters, root cause patterns.

**Approach:** Lightweight read-only dashboard that consumes the JSON/JSONL output files already produced by the experiment loop.

**Components:**
- **Backend:** FastAPI serving experiment data from JSONL/JSON files
- **Frontend:** React + Recharts (or similar) for interactive charts
- **Views:**
  - Experiment overview — rounds table, score/failure trend lines
  - Failure explorer — filterable table of `EvaluationResult` records
  - Cluster visualization — 2D scatter plot of reasoning trace clusters (from `FailureClusterer`)
  - Root cause patterns — bar chart of top patterns by frequency
  - Model comparison — side-by-side accuracy across models

**Key decisions:**
- Read-only (no writes back to result files)
- Standalone app, not embedded in the reasonbench package
- Lives in `dashboard/` directory
- Can load any experiment output directory

**Dependencies:** `fastapi`, `uvicorn`, `react`, `recharts`

---

## Phase 8: CI/CD Pipeline

**Goal:** Automated test, lint, and type-check on every push via GitHub Actions.

**Workflow: `.github/workflows/ci.yml`**

| Job | Tool | Purpose |
|-----|------|---------|
| `test` | `pytest` | Run full test suite (258+ tests) |
| `lint` | `ruff` | Lint and format check |
| `typecheck` | `mypy` | Static type checking |
| `coverage` | `pytest-cov` | Coverage report (target: 90%+) |

**Matrix:** Python 3.12, 3.13 on ubuntu-latest

**Key decisions:**
- Use `uv` for fast dependency installation in CI
- Cache `.venv` between runs
- Run on push to `master` and on PRs
- Add status badges to README

---

## Phase 9: Expanded Template Library

**Goal:** Grow from 10 to 25+ adversarial prompt templates covering more reasoning failure modes.

**New template categories:**

| Template ID | Failure Target | Description |
|-------------|---------------|-------------|
| `temporal_ordering_trap` | `multi_step_break` | Events with implicit time dependencies |
| `negation_scope_trap` | `invalid_inference` | Ambiguous negation scope ("not all X are Y") |
| `base_rate_neglect` | `unjustified_assumption` | Statistical reasoning with misleading priors |
| `survivorship_bias_trap` | `overgeneralization` | Conclusions from non-representative samples |
| `modus_tollens_break` | `contradiction` | Failure to apply contrapositive correctly |
| `scope_creep_trap` | `partial_satisfaction` | Gradually expanding problem scope in multi-step |
| `anchoring_trap` | `unjustified_assumption` | Irrelevant numbers biasing quantitative reasoning |
| `false_dichotomy_trap` | `ignored_constraint` | Presenting only 2 options when more exist |
| `composition_fallacy` | `overgeneralization` | Assuming parts' properties apply to whole |
| `conjunction_fallacy` | `invalid_inference` | P(A and B) rated higher than P(A) |
| `regression_to_mean_trap` | `pattern_misapplication` | Mistaking regression for causal effect |
| `conditional_probability_trap` | `unjustified_assumption` | Confusing P(A\|B) with P(B\|A) |
| `vacuous_truth_trap` | `contradiction` | Mishandling conditionals with false antecedents |
| `infinite_regress_trap` | `multi_step_break` | Recursive definitions without grounding |
| `equivocation_trap` | `ambiguity_failure` | Same word used with different meanings |

**Key decisions:**
- Each new template gets 5 parameter sets in `data/params/`
- Update `DISTRIBUTION` weights to keep the mix balanced
- New templates must target existing `FailureType` values (no taxonomy changes)
- Add templates incrementally — each batch is independently testable

---

## Phase 10: Benchmark Dataset

**Goal:** Curate a fixed, versioned set of adversarial prompts for reproducible cross-model comparison.

**Approach:**
- Run large-scale experiments (500+ prompts) across multiple models
- Select prompts that reliably discriminate between models (high variance in scores)
- Freeze the selected set as a versioned benchmark

**Deliverables:**
- `benchmarks/v1/prompts.jsonl` — fixed prompt set (100-200 prompts)
- `benchmarks/v1/baselines.json` — reference scores per model
- `benchmarks/v1/metadata.json` — version, date, generation parameters, model list
- CLI command: `reasonbench benchmark --models ... --version v1`

**Scoring dimensions:**
- Overall score (0-10 composite)
- Per-category breakdown (6 failure categories)
- Per-type breakdown (10 failure types)
- Assumption density
- Self-repair success rate

**Key decisions:**
- Benchmark versions are immutable once published
- New versions can add prompts but never remove or modify existing ones
- Baselines include confidence intervals from multiple runs
- Results are comparable only within the same benchmark version

---

## Implementation Order

Recommended sequence based on dependencies and value:

```
Phase 8 (CI/CD) ──> Phase 6 (Multi-Provider) ──> Phase 9 (Templates)
                                                        │
                                                        v
                                              Phase 10 (Benchmark)
                                                        │
                                                        v
                                              Phase 7 (Dashboard)
```

1. **Phase 8 (CI/CD)** — first, to protect all subsequent work with automated testing
2. **Phase 6 (Multi-Provider)** — enables cross-model evaluation
3. **Phase 9 (Templates)** — expands adversarial coverage
4. **Phase 10 (Benchmark)** — requires multi-provider + expanded templates
5. **Phase 7 (Dashboard)** — last, as it consumes data from all other phases
