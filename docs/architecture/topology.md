---
type: canonical
last_updated: 2026-06-29
---

# Repository topology

Archetype: `python-agent-service` (fleet topology canon). `type=tooling` in catalog.

On-disk layout as of 2026-06-29. Evaluation engine, frozen benchmarks, and results explorer.

## Tree

```text
fallax/
├── fallax/                      # evaluation engine (python -m fallax)
│   ├── clients/                 # Anthropic, OpenAI, Gemini, Ollama adapters
│   ├── data/                    # static adversarial trap JSON (25 templates)
│   ├── templates.py generator.py evolver.py   # prompt taxonomy and mutation
│   ├── evaluator.py scoring.py  # step-level evaluation pipeline
│   ├── pipeline.py runner.py      # CLI orchestration
│   ├── benchmark.py storage.py  # versioned benchmark loading and persistence
│   ├── analyzer.py clusterer.py root_cause.py  # post-run analysis
│   └── models.py taxonomy.py validators.py     # types and failure taxonomy
├── benchmarks/
│   └── v1/                      # frozen benchmark set
│       ├── prompts.jsonl        # 100 curated prompts
│       ├── baselines.json       # captured baseline scores
│       └── metadata.json        # version metadata
├── dashboard/                   # FastAPI results explorer (optional extra)
│   ├── api.py
│   └── static/index.html
├── tests/                       # pytest unit and integration suite
├── website/                     # project marketing/docs site
└── docs/                        # architecture, deployment, roadmap
```

## Surfaces

| Path | Role |
|------|------|
| `fallax/` | Core harness: generate prompts, run models, score steps, emit JSONL |
| `fallax/clients/` | Thin provider adapters behind `client.py` |
| `fallax/data/` | Static trap definitions; not runtime-generated prompts |
| `benchmarks/v1/` | Frozen prompt set and baselines; changes require a new version dir |
| `dashboard/` | Optional FastAPI UI over persisted run artifacts |
| `tests/` | Regression checks on scoring, pipeline, and benchmark loading |

## Rules

- Benchmark sets are versioned under `benchmarks/<version>/`; never mutate v1 in place.
- Provider API keys stay in environment variables, not committed configs.
- Run outputs belong in gitignored paths; dashboard reads exported JSONL.

## Related docs

- [architecture.md](../architecture.md) for module boundaries and execution flow
