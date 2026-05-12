---
type: canonical
source: none
sync: none
sla: none
---

# Changelog

All notable changes to **Fallax** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] — 2026-05-11

### Added

- **Benchmark v1** — 100 curated adversarial reasoning prompts across 25 templates. Baselines for `claude-sonnet-4-6` and `gpt-4o-mini` pending capture.
- **25 adversarial templates** — Original 10 plus 15 new patterns: temporal ordering, negation scope, base rate neglect, survivorship bias, modus tollens, scope creep, anchoring, false dichotomy, composition fallacy, conjunction fallacy, regression to mean, conditional probability, vacuous truth, infinite regress, and equivocation traps.
- **Multi-provider clients** — OpenAI, Gemini, and Ollama alongside the existing Anthropic client. `create_client(provider, ...)` factory for CLI usage.
- **Baseline CLI** — `fallax baseline capture|compare|status` subcommands for capturing model scores, detecting regressions, and tracking scores over time.
- **Experiment loop** — Multi-round `Experiment` orchestrator with structured reporting.
- **Analytics and intelligence** — `Analyzer`, `FailurePredictor`, `FailureClusterer`, `RootCauseExtractor`, and `SelfRepairTester` modules.
- **372 tests** across all components (up from 72 in Phase 1).

## [Unreleased]

- Initial public release