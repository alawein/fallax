---
type: canonical
audience: [contributors]
source: none
sync: none
sla: none
---

# Coding Guidelines

## Naming Conventions

**Files:**
- Python: `snake_case.py`
- TypeScript components: `PascalCase.tsx`
- Config files: `kebab-case.json` or `kebab-case.md`

**Functions/Methods:**
- Verb-noun pattern: `fetchData()`, `buildWidget()`
- Consistent tense: present for operations (`validate`), past for state (`loaded`)

**Git Branches:**
- Feature: `feat/description`
- Fix: `fix/issue-number`
- Docs: `docs/topic`
- Chore: `chore/task`

**Commits:**
- Conventional commits: `feat(scope): subject`, `fix(scope): subject`
- Subject: lowercase, imperative, <50 chars
- Body: explain why, not what (if needed)

## Code Style

Style configuration: See project linter and formatter configs.

## Testing

- Write tests before code (TDD)
- Run tests before committing

## Documentation

- README.md: Project overview, quick start, installation
- Code comments: Why, not what. Comments explain intent.
