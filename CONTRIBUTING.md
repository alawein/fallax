---
type: canonical
source: _devkit/templates
sync: propagated
sla: none
---

# Contributing to fallax

LLM adversarial reasoning evaluation system -- multi-provider prompt generation and diagnostics.

This project follows the [alawein org contributing standards](https://github.com/alawein/alawein/blob/main/CONTRIBUTING.md).

## Getting Started

```bash
git clone https://github.com/alawein/fallax.git
cd fallax
uv sync --all-extras
```

## Development Workflow

1. Branch off `master` using prefix: `feat/`, `fix/`, `docs/`, `chore/`, `test/`
2. Make your changes — keep PRs focused on a single concern
3. Run `python -m pytest tests/` to validate your changes before committing
4. Commit using [Conventional Commits](https://www.conventionalcommits.org/) — `type(scope): subject`
5. Open a Pull Request to `master`

## Code Standards

- Python 3.12+, uv for dependencies, ruff + mypy strict
- Type annotations required on all public functions
- Versioned benchmarks in `benchmarks/` are immutable
- API keys via environment variables only

## Pull Request Checklist

- [ ] CI passes (no failing checks)
- [ ] Tests added or updated for new functionality
- [ ] `python -m ruff check reasonbench/ tests/ && python -m mypy reasonbench/ && python -m pytest tests/` passes
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No breaking changes without a version bump plan

## Reporting Issues

Open an issue on the [GitHub repository](https://github.com/alawein/fallax/issues) with steps to reproduce and relevant context.

## License

By contributing, you agree that your contributions will be licensed under [MIT](LICENSE).
