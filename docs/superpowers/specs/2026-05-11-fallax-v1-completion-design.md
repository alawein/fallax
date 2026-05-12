---
type: canonical
source: none
sync: none
sla: none
---

# Fallax v1 Completion Design

**Date:** 2026-05-11
**Approach:** Research publication posture

---

## Project Goal

Fallax evaluates language models on structured, multi-step reasoning tasks and surfaces failure
modes that single-turn benchmarks miss. The v1.0 milestone delivers a reproducible benchmark
suite with comparative results across at least two LLM providers.

---

## Definition of Done

Fallax v1.0 is complete when:

1. All tests pass (`pytest tests/ -q` exits 0, zero errors).
2. Multi-provider clients (OpenAI, Gemini, Ollama) have test coverage and type-check clean.
3. 25 adversarial templates are implemented and tested.
4. `benchmarks/v1/` contains a frozen prompt set, captured baselines for 2+ models, and a metadata file.
5. README makes a concrete research claim backed by the v1 baseline results.
6. CI is green on every push (test, lint, typecheck, smoke).

---

## Current State

### Completed (Phases 1–5 + baseline subcommand)

| Module | Status | Tests |
|---|---|---|
| taxonomy, models, templates (10), validators | Done | ~72 tests |
| scoring, storage | Done | ~21 tests |
| client (Anthropic), generator, runner, evaluator, pipeline, CLI | Done | ~75 tests |
| analyzer, predictor, clusterer | Done | ~35 tests |
| evolver, repair, root_cause | Done | ~60 tests |
| experiment, report | Done | ~35 tests |
| benchmark suite, baseline subcommands | Done | ~52 tests |
| **Total** | | **~360 tests passing** |

### Active Blocker

`tests/test_dashboard.py` — 2 failures, 12 errors.
- Root cause: `dashboard/api.py` uses deprecated `on_startup`/`on_shutdown` kwargs removed in
  the installed FastAPI version. FastAPI now requires a `lifespan` async context manager.

### Partial Work

`fallax/clients/` contains `openai.py`, `gemini.py`, `ollama.py` — added in Phase 6 — but
there are no test files specifically covering these modules.

### Not Started

- Phase 9: 15 additional adversarial templates (10→25).
- Phase 10: Versioned benchmark dataset with frozen prompts and captured baselines.

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| FastAPI version conflict may affect dashboard runtime, not just tests | Medium | Pin FastAPI version in `pyproject.toml` dashboard extra after fix |
| Multi-provider clients may have drifted from `LLMClient` protocol | Medium | `mypy fallax/clients/` + protocol compliance tests |
| Phase 10 requires real LLM API calls (cost, keys needed) | Medium | Use smallest capable model per provider; document cost estimate |
| Phase 9 templates must precede Phase 10 (sequentially dependent) | Low | Phases 2 and 3 are explicitly ordered |
| Template distribution weights must still sum to 100 after adding 15 templates | Low | Enforce with existing test `test_distribution_sums_to_100` |

---

## Phased Completion Roadmap

### Phase 0: Fix and Stabilize (1–2 days)

**Goal:** All tests green, CI clean.

| Task | Files | Definition of Done |
|---|---|---|
| Fix `dashboard/api.py` — replace deprecated `on_startup`/`on_shutdown` with `lifespan` context manager | `dashboard/api.py` | `pytest tests/test_dashboard.py` fully passes |
| Audit and pin FastAPI version in `pyproject.toml` dashboard extra | `pyproject.toml` | No version drift between `.venv` and declared deps |
| Confirm multi-provider client test coverage exists or create minimal stubs | `tests/test_clients.py` (new) | `create_client()` factory tested; each provider stub tested |
| Run `ruff check fallax/ tests/` and `mypy fallax/` clean | — | Zero lint errors, zero mypy errors |

**Success gate:** `pytest tests/ -q` exits 0 with no failures or errors.

---

### Phase 1: Multi-Provider Validation (2–3 days)

**Goal:** OpenAI, Gemini, Ollama clients are tested and `LLMClient` protocol is enforced.

**Architecture:** Each client module (`openai.py`, `gemini.py`, `ollama.py`) implements the
`LLMClient` protocol defined in `fallax/client.py`. The `create_client(provider, **kwargs)`
factory function in `fallax/clients/__init__.py` dispatches to the correct implementation.

| Task | Files | Definition of Done |
|---|---|---|
| Add `tests/test_clients.py` | `tests/test_clients.py` (new) | `create_client()` dispatch tested; each client's `complete()` tested with mocked HTTP; protocol compliance verified; 15+ tests |
| Type-check all clients | `fallax/clients/*.py` | `mypy fallax/clients/` exits 0 |
| Add "Providers" table to README | `README.md` | One row per provider showing env var and install extra |

**Success gate:** `pytest tests/test_clients.py -q` exits 0; `mypy fallax/clients/` clean.

---

### Phase 2: Template Expansion (3–5 days)

**Goal:** 25 adversarial templates (add all 15 from the Phase 9 roadmap spec).

**New templates to add (from `docs/roadmap.md`):**

| Template ID | Failure Target |
|---|---|
| `temporal_ordering_trap` | `multi_step_break` |
| `negation_scope_trap` | `invalid_inference` |
| `base_rate_neglect` | `unjustified_assumption` |
| `survivorship_bias_trap` | `overgeneralization` |
| `modus_tollens_break` | `contradiction` |
| `scope_creep_trap` | `partial_satisfaction` |
| `anchoring_trap` | `unjustified_assumption` |
| `false_dichotomy_trap` | `ignored_constraint` |
| `composition_fallacy` | `overgeneralization` |
| `conjunction_fallacy` | `invalid_inference` |
| `regression_to_mean_trap` | `pattern_misapplication` |
| `conditional_probability_trap` | `unjustified_assumption` |
| `vacuous_truth_trap` | `contradiction` |
| `infinite_regress_trap` | `multi_step_break` |
| `equivocation_trap` | `ambiguity_failure` |

**Constraints (from roadmap):**
- Each template targets an existing `FailureType` — no taxonomy changes.
- Each new template gets 5 parameter sets in `fallax/data/params/<template_id>.json`.
- `DISTRIBUTION` weights must still sum to 100 after redistribution.

| Task | Files | Definition of Done |
|---|---|---|
| Add 15 templates to `fallax/templates.py` | `fallax/templates.py` | `len(TEMPLATES) == 25`; all templates have non-empty parameters; all placeholders present in template text |
| Add parameter bank JSON files | `fallax/data/params/<id>.json` (15 new) | Each new template renders successfully from its parameter bank |
| Update `DISTRIBUTION` weights | `fallax/templates.py` | `sum(DISTRIBUTION.values()) == 100` |
| Update `test_templates.py` count assertions | `tests/test_templates.py` | All tests pass with 25 templates |

**Success gate:** `pytest tests/test_templates.py -q` exits 0; `len(TEMPLATES) == 25`.

---

### Phase 3: Benchmark v1 Dataset (3–5 days + API cost)

**Goal:** `benchmarks/v1/` has a frozen, versioned prompt set and baseline scores.

**Architecture:** Uses `fallax run` → `fallax baseline capture` workflow. The frozen benchmark
is version-immutable once committed.

| Task | Files | Definition of Done |
|---|---|---|
| Generate 500+ candidate prompts using all 25 templates | `outputs/candidates.jsonl` (gitignored) | 500 rows covering all 25 templates |
| Select 100–200 high-variance prompts | `benchmarks/v1/prompts.jsonl` | Fixed prompt file committed; prompts selected for model discrimination |
| Capture baselines for 2+ models | `benchmarks/v1/baselines.json` | At minimum: one Anthropic model + one OpenAI model |
| Write `benchmarks/v1/metadata.json` | `benchmarks/v1/metadata.json` | Fields: version, date, model list, generation params, seed, prompt count |
| Update README | `README.md` | Replace "review-tier" hedge; add benchmark results table with model names, overall scores, failure rates |

**Models for v1 baseline (minimum):**
- `claude-sonnet-4-6` (Anthropic)
- `gpt-4o-mini` or `gpt-4.1-mini` (OpenAI — lowest cost option)

**Success gate:** `fallax baseline status --version v1` prints 2+ model rows; `benchmarks/v1/` fully committed.

---

### Phase 4: Publication Cleanup (1–2 days)

**Goal:** Repo is citable and publicly promotable.

| Task | Files | Definition of Done |
|---|---|---|
| Add v1.0 entry to `CHANGELOG.md` | `CHANGELOG.md` | Summarizes Phases 1–5, Phase 6 (multi-provider), Phase 9 (templates), Phase 10 (benchmark) |
| Verify website copy matches baseline results | `website/index.html` | Stats section reflects actual v1 scores; no placeholder numbers |
| Confirm CI green on main | `.github/workflows/ci.yml`, `ci-smoke.yml` | All jobs pass on a clean push |

**Success gate:** All CI jobs green; README makes a concrete, evidence-backed research claim.

---

## Deferred Items (Post v1.0)

- Phase 7 (React dashboard frontend) — requires JS build pipeline; defer until benchmark data is stable and the visualization has real data to show.
- Additional benchmark versions (v2, v3) with new templates.
- External leaderboard or submission pipeline.
- Additional LLM providers beyond Anthropic + OpenAI.

---

## Prioritized Backlog

| Priority | Task | Phase | Effort |
|---|---|---|---|
| P0 | Fix `dashboard/api.py` FastAPI compat | 0 | Small |
| P0 | Verify/add `test_clients.py` | 0/1 | Small |
| P0 | `ruff` + `mypy` clean | 0 | Small |
| P1 | Add 15 templates to `templates.py` | 2 | Medium |
| P1 | Add parameter banks for new templates | 2 | Medium |
| P1 | Type-check clients; add protocol tests | 1 | Small |
| P1 | Generate candidate prompts (500+) | 3 | Small (automated) |
| P1 | Freeze `benchmarks/v1/prompts.jsonl` | 3 | Medium |
| P2 | Capture baselines (2+ models) | 3 | Medium (needs API keys + cost) |
| P2 | Write `benchmarks/v1/metadata.json` | 3 | Small |
| P2 | Update README with research claim | 4 | Small |
| P3 | Add v1.0 CHANGELOG entry | 4 | Small |
| P3 | Verify website stats | 4 | Small |

---

## Next 10 Immediate Actions

1. Fix `dashboard/api.py` — replace `on_startup`/`on_shutdown` with `lifespan` context manager.
2. Confirm pinned FastAPI version in `pyproject.toml` dashboard extra is consistent with the fix.
3. Run `pytest tests/test_dashboard.py -q` to confirm all 14 tests pass.
4. Run `pytest tests/ -q` to confirm full suite is green.
5. Check if `tests/test_clients.py` exists; if not, create minimal tests for `create_client()` factory.
6. Run `mypy fallax/clients/` and fix any type errors.
7. Run `ruff check fallax/ tests/` and fix any issues.
8. Add 15 templates to `fallax/templates.py` with parameter lists and template text.
9. Add `fallax/data/params/<template_id>.json` for each new template.
10. Update `DISTRIBUTION` weights and run `pytest tests/test_templates.py -q`.
