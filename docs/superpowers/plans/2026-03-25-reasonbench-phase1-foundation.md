---
type: canonical
source: none
sync: none
sla: none
---

# Fallax Phase 1: Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational data models, prompt templates, validators, scoring engine, and JSONL storage for an LLM adversarial reasoning evaluation system.

**Architecture:** Domain-driven flat package with pure functions and Pydantic models. No LLM calls in Phase 1 — all components produce/consume structured data. The taxonomy defines failure types, models define the data contracts, templates generate adversarial prompts, validators build evaluation prompts, scoring computes composite failure scores, and storage persists results as JSONL.

**Tech Stack:** Python 3.12+, Pydantic v2, pytest, uv (dependency management)

**Scope:** Phase 1 of 5. This plan covers the foundation layer only. Phases 2–5 (multi-agent orchestration, failure prediction, prompt evolution, API/UI) will be separate plans built on top of these stable interfaces.

---

## File Structure

```
reasonbench/                  # repo root
├── .gitignore
├── pyproject.toml            # project metadata, deps, tool config
├── reasonbench/              # package
│   ├── __init__.py           # public API re-exports
│   ├── taxonomy.py           # FailureCategory, FailureType, Severity enums + category map
│   ├── models.py             # Pydantic models: Prompt, ModelResponse, ValidationResult, etc.
│   ├── templates.py          # 10 prompt templates + TemplateRegistry
│   ├── validators.py         # 5 validator prompt builders (ValidatorPack)
│   ├── scoring.py            # Score computation, severity mapping, hardness
│   └── storage.py            # JSONL append/read for EvaluationResult
├── tests/
│   ├── __init__.py
│   ├── test_taxonomy.py
│   ├── test_models.py
│   ├── test_templates.py
│   ├── test_validators.py
│   ├── test_scoring.py
│   ├── test_storage.py
│   └── test_integration.py   # end-to-end: template → render → score → store → read
└── docs/
    └── superpowers/
        └── plans/
            └── 2026-03-25-reasonbench-phase1-foundation.md  # this file
```

**Dependency order:** `taxonomy` → `models` → (`templates`, `validators`, `scoring`, `storage`) → `integration test`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `reasonbench/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd C:/Users/mesha/Desktop/GitHub/alawein/fallax
git init
```

- [ ] **Step 2: Create .gitignore**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
*.jsonl
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 3: Create pyproject.toml**

```toml
[project]
name = "reasonbench"
version = "0.1.0"
description = "LLM Adversarial Reasoning Evaluation System"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0,<3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["reasonbench"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create empty package and test init files**

Create `reasonbench/__init__.py`:
```python
"""Fallax — LLM Adversarial Reasoning Evaluation System."""
```

Create `tests/__init__.py` (empty file).

- [ ] **Step 5: Install dependencies with uv**

```bash
uv sync --dev
```

- [ ] **Step 6: Verify pytest runs (no tests yet)**

Run: `uv run pytest --co -q`
Expected: `no tests ran`

- [ ] **Step 7: Commit scaffolding**

```bash
git add .gitignore pyproject.toml reasonbench/__init__.py tests/__init__.py uv.lock
git commit -m "chore: scaffold reasonbench project with pyproject.toml and uv"
```

---

## Task 2: Failure Taxonomy

**Files:**
- Create: `reasonbench/taxonomy.py`
- Create: `tests/test_taxonomy.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_taxonomy.py`:

```python
from reasonbench.taxonomy import (
    FailureCategory,
    FailureType,
    Severity,
    FAILURE_CATEGORY_MAP,
    get_category,
)


class TestFailureCategory:
    def test_has_six_categories(self):
        assert len(FailureCategory) == 6

    def test_values(self):
        expected = {
            "logic_error",
            "assumption_error",
            "constraint_violation",
            "generalization_error",
            "ambiguity_failure",
            "multi_step_break",
        }
        assert {c.value for c in FailureCategory} == expected


class TestFailureType:
    def test_has_ten_types(self):
        assert len(FailureType) == 10

    def test_values(self):
        expected = {
            "contradiction",
            "invalid_inference",
            "unstated_assumption",
            "unjustified_assumption",
            "ignored_constraint",
            "partial_satisfaction",
            "overgeneralization",
            "pattern_misapplication",
            "ambiguity_failure",
            "multi_step_break",
        }
        assert {t.value for t in FailureType} == expected


class TestSeverity:
    def test_has_four_levels(self):
        assert len(Severity) == 4

    def test_ordering(self):
        levels = [s.value for s in Severity]
        assert levels == ["critical", "high", "medium", "low"]


class TestCategoryMap:
    def test_covers_all_failure_types(self):
        assert set(FAILURE_CATEGORY_MAP.keys()) == set(FailureType)

    def test_logic_error_subtypes(self):
        assert FAILURE_CATEGORY_MAP[FailureType.CONTRADICTION] == FailureCategory.LOGIC_ERROR
        assert FAILURE_CATEGORY_MAP[FailureType.INVALID_INFERENCE] == FailureCategory.LOGIC_ERROR

    def test_assumption_error_subtypes(self):
        assert FAILURE_CATEGORY_MAP[FailureType.UNSTATED_ASSUMPTION] == FailureCategory.ASSUMPTION_ERROR
        assert FAILURE_CATEGORY_MAP[FailureType.UNJUSTIFIED_ASSUMPTION] == FailureCategory.ASSUMPTION_ERROR

    def test_constraint_violation_subtypes(self):
        assert FAILURE_CATEGORY_MAP[FailureType.IGNORED_CONSTRAINT] == FailureCategory.CONSTRAINT_VIOLATION
        assert FAILURE_CATEGORY_MAP[FailureType.PARTIAL_SATISFACTION] == FailureCategory.CONSTRAINT_VIOLATION

    def test_generalization_error_subtypes(self):
        assert FAILURE_CATEGORY_MAP[FailureType.OVERGENERALIZATION] == FailureCategory.GENERALIZATION_ERROR
        assert FAILURE_CATEGORY_MAP[FailureType.PATTERN_MISAPPLICATION] == FailureCategory.GENERALIZATION_ERROR


class TestGetCategory:
    def test_returns_correct_category(self):
        assert get_category(FailureType.CONTRADICTION) == FailureCategory.LOGIC_ERROR

    def test_every_type_resolves(self):
        for ft in FailureType:
            result = get_category(ft)
            assert isinstance(result, FailureCategory)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_taxonomy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reasonbench.taxonomy'`

- [ ] **Step 3: Write implementation**

Create `reasonbench/taxonomy.py`:

```python
"""Failure taxonomy for adversarial reasoning evaluation.

Tier 1: Six broad failure categories.
Tier 2: Ten granular failure types mapped to Tier 1 categories.
"""

from enum import Enum


class FailureCategory(str, Enum):
    """Tier 1 failure categories."""

    LOGIC_ERROR = "logic_error"
    ASSUMPTION_ERROR = "assumption_error"
    CONSTRAINT_VIOLATION = "constraint_violation"
    GENERALIZATION_ERROR = "generalization_error"
    AMBIGUITY_FAILURE = "ambiguity_failure"
    MULTI_STEP_BREAK = "multi_step_break"


class FailureType(str, Enum):
    """Tier 2 granular failure types."""

    CONTRADICTION = "contradiction"
    INVALID_INFERENCE = "invalid_inference"
    UNSTATED_ASSUMPTION = "unstated_assumption"
    UNJUSTIFIED_ASSUMPTION = "unjustified_assumption"
    IGNORED_CONSTRAINT = "ignored_constraint"
    PARTIAL_SATISFACTION = "partial_satisfaction"
    OVERGENERALIZATION = "overgeneralization"
    PATTERN_MISAPPLICATION = "pattern_misapplication"
    AMBIGUITY_FAILURE = "ambiguity_failure"
    MULTI_STEP_BREAK = "multi_step_break"


class Severity(str, Enum):
    """Score-based severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


FAILURE_CATEGORY_MAP: dict[FailureType, FailureCategory] = {
    FailureType.CONTRADICTION: FailureCategory.LOGIC_ERROR,
    FailureType.INVALID_INFERENCE: FailureCategory.LOGIC_ERROR,
    FailureType.UNSTATED_ASSUMPTION: FailureCategory.ASSUMPTION_ERROR,
    FailureType.UNJUSTIFIED_ASSUMPTION: FailureCategory.ASSUMPTION_ERROR,
    FailureType.IGNORED_CONSTRAINT: FailureCategory.CONSTRAINT_VIOLATION,
    FailureType.PARTIAL_SATISFACTION: FailureCategory.CONSTRAINT_VIOLATION,
    FailureType.OVERGENERALIZATION: FailureCategory.GENERALIZATION_ERROR,
    FailureType.PATTERN_MISAPPLICATION: FailureCategory.GENERALIZATION_ERROR,
    FailureType.AMBIGUITY_FAILURE: FailureCategory.AMBIGUITY_FAILURE,
    FailureType.MULTI_STEP_BREAK: FailureCategory.MULTI_STEP_BREAK,
}


def get_category(failure_type: FailureType) -> FailureCategory:
    """Look up the Tier 1 category for a Tier 2 failure type."""
    return FAILURE_CATEGORY_MAP[failure_type]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_taxonomy.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/taxonomy.py tests/test_taxonomy.py
git commit -m "feat(taxonomy): add failure categories, types, severity enums"
```

---

## Task 3: Data Models

**Files:**
- Create: `reasonbench/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models.py`:

```python
import json

import pytest
from pydantic import ValidationError

from reasonbench.models import (
    Assumption,
    EvaluationResult,
    FailureRecord,
    ModelResponse,
    Prompt,
    ValidationResult,
)
from reasonbench.taxonomy import FailureType, Severity


class TestPrompt:
    def test_create_with_defaults(self):
        p = Prompt(
            failure_type=FailureType.CONTRADICTION,
            prompt_text="Test prompt",
            difficulty=5,
        )
        assert p.prompt_id  # auto-generated UUID
        assert p.failure_type == FailureType.CONTRADICTION
        assert p.prompt_text == "Test prompt"
        assert p.difficulty == 5
        assert p.template_id is None
        assert p.parameters == {}

    def test_create_with_all_fields(self):
        p = Prompt(
            prompt_id="custom-id",
            failure_type=FailureType.UNSTATED_ASSUMPTION,
            prompt_text="Full prompt",
            difficulty=8,
            template_id="implicit_assumption_trap",
            parameters={"rule_a": "x > 0", "rule_b": "x < 10"},
        )
        assert p.prompt_id == "custom-id"
        assert p.template_id == "implicit_assumption_trap"
        assert p.parameters["rule_a"] == "x > 0"

    def test_difficulty_bounds_low(self):
        with pytest.raises(ValidationError):
            Prompt(
                failure_type=FailureType.CONTRADICTION,
                prompt_text="Test",
                difficulty=0,
            )

    def test_difficulty_bounds_high(self):
        with pytest.raises(ValidationError):
            Prompt(
                failure_type=FailureType.CONTRADICTION,
                prompt_text="Test",
                difficulty=11,
            )

    def test_unique_ids(self):
        p1 = Prompt(failure_type=FailureType.CONTRADICTION, prompt_text="A", difficulty=1)
        p2 = Prompt(failure_type=FailureType.CONTRADICTION, prompt_text="B", difficulty=1)
        assert p1.prompt_id != p2.prompt_id


class TestModelResponse:
    def test_create_minimal(self):
        r = ModelResponse(
            model_name="gpt-4",
            answer="42",
            reasoning="Because math.",
        )
        assert r.model_name == "gpt-4"
        assert r.is_correct is None
        assert r.confidence is None

    def test_create_full(self):
        r = ModelResponse(
            model_name="claude",
            answer="yes",
            reasoning="Step 1...",
            is_correct=True,
            confidence=0.95,
        )
        assert r.confidence == 0.95

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ModelResponse(
                model_name="m", answer="a", reasoning="r", confidence=1.5
            )


class TestAssumption:
    def test_create(self):
        a = Assumption(text="x is positive", justified=False)
        assert a.text == "x is positive"
        assert a.justified is False


class TestValidationResult:
    def test_defaults(self):
        v = ValidationResult(reasoning_flawed=False)
        assert v.first_error_step is None
        assert v.assumptions == []
        assert v.counterfactual_fail is False
        assert v.adversarial_issues == []
        assert v.final_answer_correct is None

    def test_full(self):
        v = ValidationResult(
            reasoning_flawed=True,
            first_error_step=2,
            assumptions=[Assumption(text="monotonic", justified=False)],
            counterfactual_fail=True,
            adversarial_issues=["Step 2 assumes linearity"],
            final_answer_correct=False,
        )
        assert len(v.assumptions) == 1
        assert v.assumptions[0].justified is False


class TestFailureRecord:
    def test_create(self):
        f = FailureRecord(
            failure_type=FailureType.UNSTATED_ASSUMPTION,
            step_of_failure=2,
            reason="model assumed X without evidence",
            severity=Severity.HIGH,
        )
        assert f.severity == Severity.HIGH


class TestEvaluationResult:
    def test_create_and_roundtrip(self):
        er = EvaluationResult(
            prompt_id="test-001",
            failure_type=FailureType.CONTRADICTION,
            prompt_text="Test prompt",
            models={
                "gpt-4": ModelResponse(
                    model_name="gpt-4",
                    answer="yes",
                    reasoning="...",
                    is_correct=False,
                ),
            },
            validation=ValidationResult(reasoning_flawed=True, first_error_step=1),
            score=5,
            severity=Severity.HIGH,
        )
        # JSON round-trip
        json_str = er.model_dump_json()
        restored = EvaluationResult.model_validate_json(json_str)
        assert restored.prompt_id == "test-001"
        assert restored.models["gpt-4"].is_correct is False
        assert restored.validation.reasoning_flawed is True
        assert restored.score == 5
        assert restored.severity == Severity.HIGH
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/models.py`:

```python
"""Data models for prompts, responses, validation, and evaluation results."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from .taxonomy import FailureType, Severity


class Prompt(BaseModel):
    """A generated adversarial prompt."""

    prompt_id: str = Field(default_factory=lambda: str(uuid4()))
    failure_type: FailureType
    prompt_text: str
    difficulty: int = Field(ge=1, le=10)
    template_id: str | None = None
    parameters: dict[str, str] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """A model's response to a prompt."""

    model_name: str
    answer: str
    reasoning: str
    is_correct: bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Assumption(BaseModel):
    """An implicit assumption extracted from reasoning."""

    text: str
    justified: bool


class ValidationResult(BaseModel):
    """Combined output from all 5 validators."""

    reasoning_flawed: bool
    first_error_step: int | None = None
    assumptions: list[Assumption] = Field(default_factory=list)
    counterfactual_fail: bool = False
    adversarial_issues: list[str] = Field(default_factory=list)
    final_answer_correct: bool | None = None


class FailureRecord(BaseModel):
    """A single identified failure in a model's reasoning."""

    failure_type: FailureType
    step_of_failure: int | None = None
    reason: str
    severity: Severity


class EvaluationResult(BaseModel):
    """Complete evaluation result for one prompt across models."""

    prompt_id: str
    failure_type: FailureType
    prompt_text: str
    models: dict[str, ModelResponse]
    validation: ValidationResult
    score: int
    severity: Severity
    failures: list[FailureRecord] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/models.py tests/test_models.py
git commit -m "feat(models): add Pydantic data models for prompts, responses, validation, scoring"
```

---

## Task 4: Prompt Templates and Registry

**Files:**
- Create: `reasonbench/templates.py`
- Create: `tests/test_templates.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_templates.py`:

```python
import pytest

from reasonbench.taxonomy import FailureType
from reasonbench.templates import (
    DISTRIBUTION,
    TEMPLATES,
    PromptTemplate,
    TemplateRegistry,
)


class TestPromptTemplate:
    def test_is_frozen_dataclass(self):
        t = TEMPLATES[0]
        with pytest.raises(AttributeError):
            t.template_id = "changed"


class TestTemplateConstants:
    def test_ten_templates_defined(self):
        assert len(TEMPLATES) == 10

    def test_all_have_unique_ids(self):
        ids = [t.template_id for t in TEMPLATES]
        assert len(ids) == len(set(ids))

    def test_all_have_non_empty_parameters(self):
        for t in TEMPLATES:
            assert len(t.parameters) > 0, f"{t.template_id} has no parameters"

    def test_all_template_texts_contain_placeholders(self):
        for t in TEMPLATES:
            for param in t.parameters:
                assert f"{{{param}}}" in t.template_text, (
                    f"{t.template_id} missing placeholder {{{param}}}"
                )

    def test_distribution_sums_to_100(self):
        assert sum(DISTRIBUTION.values()) == 100

    def test_distribution_keys_match_template_ids(self):
        template_ids = {t.template_id for t in TEMPLATES}
        assert set(DISTRIBUTION.keys()) == template_ids


class TestTemplateRegistry:
    @pytest.fixture()
    def registry(self):
        return TemplateRegistry()

    def test_get_existing(self, registry):
        t = registry.get("implicit_assumption_trap")
        assert t.template_id == "implicit_assumption_trap"

    def test_get_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_all(self, registry):
        all_templates = registry.list_all()
        assert len(all_templates) == 10

    def test_list_by_failure_type(self, registry):
        results = registry.list_by_failure_type(FailureType.UNSTATED_ASSUMPTION)
        assert len(results) >= 1
        assert all(
            t.failure_target == FailureType.UNSTATED_ASSUMPTION for t in results
        )

    def test_render_success(self, registry):
        rendered = registry.render(
            "implicit_assumption_trap",
            rule_a="x > 0",
            rule_b="x < 10",
            edge_case_input="x = 0",
        )
        assert "x > 0" in rendered
        assert "x < 10" in rendered
        assert "x = 0" in rendered

    def test_render_missing_param_raises(self, registry):
        with pytest.raises(KeyError, match="Missing parameters"):
            registry.render("implicit_assumption_trap", rule_a="x > 0")

    def test_template_ids(self, registry):
        ids = registry.template_ids()
        assert len(ids) == 10
        assert "implicit_assumption_trap" in ids
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_templates.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/templates.py`:

```python
"""Adversarial prompt templates and registry."""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import FailureType


@dataclass(frozen=True)
class PromptTemplate:
    """A parameterizable adversarial prompt template."""

    template_id: str
    failure_target: FailureType
    template_text: str
    parameters: list[str]
    description: str


TEMPLATES: tuple[PromptTemplate, ...] = (
    PromptTemplate(
        template_id="implicit_assumption_trap",
        failure_target=FailureType.UNSTATED_ASSUMPTION,
        template_text=(
            "A system processes inputs under the following rules:\n"
            "- Rule A: {rule_a}\n"
            "- Rule B: {rule_b}\n\n"
            "No other rules are specified.\n\n"
            "Question: What happens when {edge_case_input}?\n"
            "Explain your reasoning step by step."
        ),
        parameters=["rule_a", "rule_b", "edge_case_input"],
        description="Model invents missing rules",
    ),
    PromptTemplate(
        template_id="contradictory_constraints",
        failure_target=FailureType.IGNORED_CONSTRAINT,
        template_text=(
            "A process must satisfy ALL of the following:\n"
            "1. {constraint_1}\n"
            "2. {constraint_2}\n"
            "3. {constraint_3}\n\n"
            "Is it possible? If yes, provide an example. If no, prove why not."
        ),
        parameters=["constraint_1", "constraint_2", "constraint_3"],
        description="Model ignores one constraint",
    ),
    PromptTemplate(
        template_id="false_analogy_trap",
        failure_target=FailureType.PATTERN_MISAPPLICATION,
        template_text=(
            "Problem: This scenario looks similar to {well_known_problem}, "
            "but differs in the following way: {key_difference}\n\n"
            "Solve the problem."
        ),
        parameters=["well_known_problem", "key_difference"],
        description="Model applies wrong pattern",
    ),
    PromptTemplate(
        template_id="recursive_definition_break",
        failure_target=FailureType.CONTRADICTION,
        template_text=(
            "Define a function:\n"
            "- Base case: {base_case}\n"
            "- Recursive rule: {recursive_rule}\n\n"
            "Question: Is this function well-defined for all inputs? Justify."
        ),
        parameters=["base_case", "recursive_rule"],
        description="Misses contradiction between base and recursion",
    ),
    PromptTemplate(
        template_id="multi_step_dependency",
        failure_target=FailureType.MULTI_STEP_BREAK,
        template_text=(
            "Given:\n"
            "Step 1: {step1}\n"
            "Step 2 depends on Step 1: {step2}\n"
            "Step 3 depends on Step 2: {step3}\n\n"
            "Question: What is the final result?"
        ),
        parameters=["step1", "step2", "step3"],
        description="Error propagation unnoticed",
    ),
    PromptTemplate(
        template_id="edge_case_inversion",
        failure_target=FailureType.OVERGENERALIZATION,
        template_text=(
            "A rule works for all typical cases: {general_rule}\n\n"
            "Test it on this boundary case: {edge_case}\n\n"
            "Does the rule still hold?"
        ),
        parameters=["general_rule", "edge_case"],
        description="Model assumes general rule holds at boundaries",
    ),
    PromptTemplate(
        template_id="ambiguous_spec_trap",
        failure_target=FailureType.AMBIGUITY_FAILURE,
        template_text=(
            "A system is described as: {ambiguous_description}\n\n"
            "Question: What is the output for {input}?\n"
            "If assumptions are required, state them explicitly."
        ),
        parameters=["ambiguous_description", "input"],
        description="Model doesn't declare assumptions",
    ),
    PromptTemplate(
        template_id="overconstrained_optimization",
        failure_target=FailureType.PARTIAL_SATISFACTION,
        template_text=(
            "Maximize {objective} subject to: {constraints}\n\n"
            "Is there a solution? If so, find it. If not, explain why."
        ),
        parameters=["objective", "constraints"],
        description="Returns impossible solution",
    ),
    PromptTemplate(
        template_id="hidden_variable_trap",
        failure_target=FailureType.UNJUSTIFIED_ASSUMPTION,
        template_text=(
            "A result depends on variables X and Y.\n"
            "You are given: {partial_info}\n\n"
            "Find the result."
        ),
        parameters=["partial_info"],
        description="Model assumes missing variable",
    ),
    PromptTemplate(
        template_id="self_consistency_trap",
        failure_target=FailureType.INVALID_INFERENCE,
        template_text=(
            "Here is a reasoning chain:\n"
            "{reasoning_chain}\n\n"
            "Question: Is this reasoning correct? "
            "If not, identify the first incorrect step."
        ),
        parameters=["reasoning_chain"],
        description="Model agrees with flawed reasoning",
    ),
)

DISTRIBUTION: dict[str, int] = {
    "implicit_assumption_trap": 15,
    "contradictory_constraints": 15,
    "edge_case_inversion": 10,
    "false_analogy_trap": 10,
    "multi_step_dependency": 10,
    "recursive_definition_break": 10,
    "ambiguous_spec_trap": 10,
    "overconstrained_optimization": 10,
    "hidden_variable_trap": 5,
    "self_consistency_trap": 5,
}


class TemplateRegistry:
    """Registry for looking up and rendering prompt templates."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {
            t.template_id: t for t in TEMPLATES
        }

    def get(self, template_id: str) -> PromptTemplate:
        """Get a template by ID. Raises KeyError if not found."""
        return self._templates[template_id]

    def list_all(self) -> list[PromptTemplate]:
        """Return all registered templates."""
        return list(self._templates.values())

    def list_by_failure_type(
        self, failure_type: FailureType
    ) -> list[PromptTemplate]:
        """Return templates targeting a specific failure type."""
        return [
            t
            for t in self._templates.values()
            if t.failure_target == failure_type
        ]

    def render(self, template_id: str, **params: str) -> str:
        """Render a template with given parameters.

        Raises KeyError if template not found or required parameter missing.
        """
        template = self.get(template_id)
        missing = set(template.parameters) - set(params.keys())
        if missing:
            raise KeyError(f"Missing parameters: {missing}")
        return template.template_text.format(**params)

    def template_ids(self) -> list[str]:
        """Return all template IDs."""
        return list(self._templates.keys())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_templates.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/templates.py tests/test_templates.py
git commit -m "feat(templates): add 10 adversarial prompt templates with registry"
```

---

## Task 5: Validator Prompt Pack

**Files:**
- Create: `reasonbench/validators.py`
- Create: `tests/test_validators.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_validators.py`:

```python
from reasonbench.validators import ValidatorPack


class TestReasoningCritic:
    def test_returns_string(self):
        result = ValidatorPack.reasoning_critic(
            prompt="What is 2+2?",
            answer="5",
            reasoning="2+2=5 because...",
        )
        assert isinstance(result, str)

    def test_includes_inputs(self):
        result = ValidatorPack.reasoning_critic(
            prompt="test prompt",
            answer="test answer",
            reasoning="test reasoning",
        )
        assert "test prompt" in result
        assert "test answer" in result
        assert "test reasoning" in result

    def test_includes_audit_instruction(self):
        result = ValidatorPack.reasoning_critic(
            prompt="p", answer="a", reasoning="r"
        )
        assert "FIRST step" in result


class TestAssumptionExtractor:
    def test_returns_string(self):
        result = ValidatorPack.assumption_extractor("some reasoning")
        assert isinstance(result, str)

    def test_includes_reasoning(self):
        result = ValidatorPack.assumption_extractor("step 1: assume X")
        assert "step 1: assume X" in result

    def test_asks_for_justification(self):
        result = ValidatorPack.assumption_extractor("r")
        assert "YES/NO" in result


class TestCounterfactualTest:
    def test_returns_string(self):
        result = ValidatorPack.counterfactual_test(
            reasoning="r", perturbation="change X to Y"
        )
        assert isinstance(result, str)

    def test_includes_both_inputs(self):
        result = ValidatorPack.counterfactual_test(
            reasoning="original reasoning",
            perturbation="flip the sign",
        )
        assert "original reasoning" in result
        assert "flip the sign" in result


class TestAdversarialChallenger:
    def test_returns_string(self):
        result = ValidatorPack.adversarial_challenger("some reasoning")
        assert isinstance(result, str)

    def test_includes_reasoning(self):
        result = ValidatorPack.adversarial_challenger("step 1: do X")
        assert "step 1: do X" in result

    def test_adversarial_intent(self):
        result = ValidatorPack.adversarial_challenger("r")
        assert "WRONG" in result


class TestTruthJudge:
    def test_returns_string(self):
        result = ValidatorPack.truth_judge(prompt="p", answer="a")
        assert isinstance(result, str)

    def test_includes_inputs(self):
        result = ValidatorPack.truth_judge(
            prompt="my prompt", answer="my answer"
        )
        assert "my prompt" in result
        assert "my answer" in result

    def test_yes_no_format(self):
        result = ValidatorPack.truth_judge(prompt="p", answer="a")
        assert "YES/NO" in result


class TestValidatorPackCompleteness:
    def test_has_five_validators(self):
        methods = [
            ValidatorPack.reasoning_critic,
            ValidatorPack.assumption_extractor,
            ValidatorPack.counterfactual_test,
            ValidatorPack.adversarial_challenger,
            ValidatorPack.truth_judge,
        ]
        assert len(methods) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validators.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/validators.py`:

```python
"""Validator prompt pack — 5 validators for reasoning evaluation."""


class ValidatorPack:
    """Builds validator prompts for evaluating model reasoning.

    Each method returns a fully-formed prompt string ready to send
    to a judge LLM. No LLM calls happen here — only prompt construction.
    """

    @staticmethod
    def reasoning_critic(prompt: str, answer: str, reasoning: str) -> str:
        """Build a reasoning critic prompt."""
        return (
            "You are a strict reasoning auditor.\n\n"
            f"Given:\nPROMPT: {prompt}\n"
            f"MODEL ANSWER: {answer}\n"
            f"REASONING: {reasoning}\n\n"
            "Task:\n"
            "1. Identify the FIRST step where reasoning may be invalid.\n"
            "2. Explain why it is invalid.\n"
            "3. If valid, explicitly confirm no issues.\n\n"
            "Be precise. Do not restate the solution."
        )

    @staticmethod
    def assumption_extractor(reasoning: str) -> str:
        """Build an assumption extractor prompt."""
        return (
            "Extract all implicit assumptions made in the reasoning.\n\n"
            f"REASONING: {reasoning}\n\n"
            "For each assumption:\n"
            "- State it clearly\n"
            "- Say whether it is justified by the prompt (YES/NO)"
        )

    @staticmethod
    def counterfactual_test(reasoning: str, perturbation: str) -> str:
        """Build a counterfactual stress test prompt."""
        return (
            "Given this reasoning:\n"
            f"{reasoning}\n\n"
            "Modify the input slightly:\n"
            f"{perturbation}\n\n"
            "Question: Does the reasoning still hold? "
            "If not, where does it break?"
        )

    @staticmethod
    def adversarial_challenger(reasoning: str) -> str:
        """Build an adversarial challenger prompt."""
        return (
            "Your goal is to prove the reasoning WRONG.\n\n"
            f"REASONING: {reasoning}\n\n"
            "Find any flaw, inconsistency, or unjustified step.\n\n"
            "If none exist, explain why the reasoning is robust."
        )

    @staticmethod
    def truth_judge(prompt: str, answer: str) -> str:
        """Build a final truth judge prompt."""
        return (
            "Given:\n"
            f"Prompt: {prompt}\n"
            f"Answer: {answer}\n\n"
            "Is the final answer correct? (YES/NO)\n\n"
            "Do NOT consider reasoning — only correctness."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validators.py -v`
Expected: All 14 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/validators.py tests/test_validators.py
git commit -m "feat(validators): add 5-validator prompt pack for reasoning evaluation"
```

---

## Task 6: Scoring Engine

**Files:**
- Create: `reasonbench/scoring.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_scoring.py`:

```python
import pytest

from reasonbench.scoring import Scorer
from reasonbench.taxonomy import Severity


class TestComputeScore:
    def test_all_good_returns_zero(self):
        score = Scorer.compute_score(
            is_correct=True,
            reasoning_flawed=False,
            assumption_errors=0,
            counterfactual_fail=False,
            model_disagreement=False,
        )
        assert score == 0

    def test_incorrect_adds_2(self):
        score = Scorer.compute_score(
            is_correct=False,
            reasoning_flawed=False,
            assumption_errors=0,
            counterfactual_fail=False,
            model_disagreement=False,
        )
        assert score == 2

    def test_reasoning_flawed_adds_3(self):
        score = Scorer.compute_score(
            is_correct=True,
            reasoning_flawed=True,
            assumption_errors=0,
            counterfactual_fail=False,
            model_disagreement=False,
        )
        assert score == 3

    def test_assumption_errors_add_per_error(self):
        score = Scorer.compute_score(
            is_correct=True,
            reasoning_flawed=False,
            assumption_errors=4,
            counterfactual_fail=False,
            model_disagreement=False,
        )
        assert score == 4

    def test_counterfactual_fail_adds_2(self):
        score = Scorer.compute_score(
            is_correct=True,
            reasoning_flawed=False,
            assumption_errors=0,
            counterfactual_fail=True,
            model_disagreement=False,
        )
        assert score == 2

    def test_model_disagreement_adds_1(self):
        score = Scorer.compute_score(
            is_correct=True,
            reasoning_flawed=False,
            assumption_errors=0,
            counterfactual_fail=False,
            model_disagreement=True,
        )
        assert score == 1

    def test_all_bad_maximum(self):
        score = Scorer.compute_score(
            is_correct=False,
            reasoning_flawed=True,
            assumption_errors=3,
            counterfactual_fail=True,
            model_disagreement=True,
        )
        # 2 + 3 + 3 + 2 + 1 = 11
        assert score == 11

    def test_critical_scenario(self):
        score = Scorer.compute_score(
            is_correct=False,
            reasoning_flawed=True,
            assumption_errors=1,
            counterfactual_fail=True,
            model_disagreement=False,
        )
        # 2 + 3 + 1 + 2 = 8
        assert score == 8


class TestSeverity:
    @pytest.mark.parametrize(
        "score, expected",
        [
            (0, Severity.LOW),
            (1, Severity.LOW),
            (2, Severity.MEDIUM),
            (3, Severity.MEDIUM),
            (4, Severity.HIGH),
            (5, Severity.HIGH),
            (6, Severity.CRITICAL),
            (10, Severity.CRITICAL),
        ],
    )
    def test_severity_thresholds(self, score, expected):
        assert Scorer.severity(score) == expected


class TestHardness:
    def test_zero(self):
        assert Scorer.hardness(
            wrong_models=0, reasoning_failures=0, repair_failures=0
        ) == 0

    def test_formula(self):
        result = Scorer.hardness(
            wrong_models=2, reasoning_failures=3, repair_failures=1
        )
        # 2*2 + 3*2 + 1*3 = 13
        assert result == 13

    def test_repair_failures_weighted_highest(self):
        a = Scorer.hardness(wrong_models=0, reasoning_failures=0, repair_failures=1)
        b = Scorer.hardness(wrong_models=1, reasoning_failures=0, repair_failures=0)
        assert a > b  # 3 > 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/scoring.py`:

```python
"""Scoring engine for evaluation results."""

from .taxonomy import Severity


class Scorer:
    """Computes composite failure scores, severity levels, and prompt hardness."""

    @staticmethod
    def compute_score(
        *,
        is_correct: bool,
        reasoning_flawed: bool,
        assumption_errors: int,
        counterfactual_fail: bool,
        model_disagreement: bool,
    ) -> int:
        """Compute composite failure score. Higher = more severe failure."""
        score = 0
        if not is_correct:
            score += 2
        if reasoning_flawed:
            score += 3
        score += assumption_errors
        if counterfactual_fail:
            score += 2
        if model_disagreement:
            score += 1
        return score

    @staticmethod
    def severity(score: int) -> Severity:
        """Map a composite score to a severity level."""
        if score >= 6:
            return Severity.CRITICAL
        if score >= 4:
            return Severity.HIGH
        if score >= 2:
            return Severity.MEDIUM
        return Severity.LOW

    @staticmethod
    def hardness(
        *,
        wrong_models: int,
        reasoning_failures: int,
        repair_failures: int,
    ) -> int:
        """Compute prompt hardness for ranking difficulty."""
        return (
            wrong_models * 2
            + reasoning_failures * 2
            + repair_failures * 3
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: All 14 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/scoring.py tests/test_scoring.py
git commit -m "feat(scoring): add composite score, severity mapping, and hardness computation"
```

---

## Task 7: JSONL Storage

**Files:**
- Create: `reasonbench/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_storage.py`:

```python
from pathlib import Path

import pytest

from reasonbench.models import (
    EvaluationResult,
    ModelResponse,
    ValidationResult,
)
from reasonbench.storage import JsonlStore
from reasonbench.taxonomy import FailureType, Severity


def _make_result(prompt_id: str, score: int) -> EvaluationResult:
    """Helper to build a minimal EvaluationResult."""
    return EvaluationResult(
        prompt_id=prompt_id,
        failure_type=FailureType.CONTRADICTION,
        prompt_text="test prompt",
        models={
            "test-model": ModelResponse(
                model_name="test-model",
                answer="42",
                reasoning="because",
                is_correct=False,
            ),
        },
        validation=ValidationResult(reasoning_flawed=True),
        score=score,
        severity=Severity.HIGH if score >= 4 else Severity.LOW,
    )


class TestJsonlStore:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> JsonlStore:
        return JsonlStore(tmp_path / "results.jsonl")

    def test_read_nonexistent_returns_empty(self, store):
        assert store.read_all() == []

    def test_count_nonexistent_returns_zero(self, store):
        assert store.count() == 0

    def test_append_and_read_single(self, store):
        result = _make_result("id-1", score=5)
        store.append(result)
        loaded = store.read_all()
        assert len(loaded) == 1
        assert loaded[0].prompt_id == "id-1"
        assert loaded[0].score == 5

    def test_append_multiple_and_count(self, store):
        store.append(_make_result("id-1", score=3))
        store.append(_make_result("id-2", score=7))
        store.append(_make_result("id-3", score=1))
        assert store.count() == 3

    def test_read_by_min_score(self, store):
        store.append(_make_result("low", score=1))
        store.append(_make_result("mid", score=4))
        store.append(_make_result("high", score=8))
        results = store.read_by_min_score(4)
        ids = {r.prompt_id for r in results}
        assert ids == {"mid", "high"}

    def test_roundtrip_preserves_nested_data(self, store):
        result = _make_result("nested", score=6)
        store.append(result)
        loaded = store.read_all()[0]
        assert loaded.models["test-model"].answer == "42"
        assert loaded.validation.reasoning_flawed is True

    def test_append_is_additive(self, store):
        store.append(_make_result("first", score=1))
        assert store.count() == 1
        store.append(_make_result("second", score=2))
        assert store.count() == 2
        # first result still present
        ids = {r.prompt_id for r in store.read_all()}
        assert "first" in ids
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/storage.py`:

```python
"""JSONL storage for evaluation results."""

from __future__ import annotations

from pathlib import Path

from .models import EvaluationResult


class JsonlStore:
    """Append-only JSONL storage for evaluation results."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, result: EvaluationResult) -> None:
        """Append a single result to the JSONL file."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

    def read_all(self) -> list[EvaluationResult]:
        """Read all results from the JSONL file."""
        if not self.path.exists():
            return []
        results: list[EvaluationResult] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(
                        EvaluationResult.model_validate_json(line)
                    )
        return results

    def count(self) -> int:
        """Count results in the file."""
        if not self.path.exists():
            return 0
        total = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
        return total

    def read_by_min_score(self, min_score: int) -> list[EvaluationResult]:
        """Read results with score >= min_score."""
        return [r for r in self.read_all() if r.score >= min_score]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_storage.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/storage.py tests/test_storage.py
git commit -m "feat(storage): add JSONL append-only store for evaluation results"
```

---

## Task 8: Package Init and Integration Test

**Files:**
- Modify: `reasonbench/__init__.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/test_integration.py`:

```python
"""End-to-end integration test: template → render → mock response → score → store → read."""

from pathlib import Path

import pytest

from reasonbench import (
    Assumption,
    EvaluationResult,
    FailureType,
    JsonlStore,
    ModelResponse,
    Prompt,
    Scorer,
    Severity,
    TemplateRegistry,
    ValidationResult,
    ValidatorPack,
    get_category,
    FailureCategory,
)


class TestEndToEnd:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> JsonlStore:
        return JsonlStore(tmp_path / "integration.jsonl")

    def test_full_pipeline(self, store):
        # 1. Render a prompt from a template
        registry = TemplateRegistry()
        prompt_text = registry.render(
            "implicit_assumption_trap",
            rule_a="inputs must be positive integers",
            rule_b="outputs are doubled",
            edge_case_input="the input is -3",
        )
        assert "positive integers" in prompt_text
        assert "-3" in prompt_text

        # 2. Create a Prompt object
        prompt = Prompt(
            failure_type=FailureType.UNSTATED_ASSUMPTION,
            prompt_text=prompt_text,
            difficulty=7,
            template_id="implicit_assumption_trap",
        )
        assert prompt.prompt_id  # UUID generated

        # 3. Verify category mapping
        category = get_category(prompt.failure_type)
        assert category == FailureCategory.ASSUMPTION_ERROR

        # 4. Mock two model responses (simulating dual-model eval)
        model_a = ModelResponse(
            model_name="model-a",
            answer="The system rejects -3",
            reasoning="Rule A says positive integers, so -3 is rejected",
            is_correct=True,
        )
        model_b = ModelResponse(
            model_name="model-b",
            answer="The output is -6",
            reasoning="Rule B doubles the input: -3 * 2 = -6",
            is_correct=False,
        )

        # 5. Build validator prompts (verify they produce strings)
        critic = ValidatorPack.reasoning_critic(
            prompt=prompt_text,
            answer=model_b.answer,
            reasoning=model_b.reasoning,
        )
        assert len(critic) > 0

        assumptions_prompt = ValidatorPack.assumption_extractor(
            model_b.reasoning
        )
        assert len(assumptions_prompt) > 0

        # 6. Create validation result (simulated validator outputs)
        validation = ValidationResult(
            reasoning_flawed=True,
            first_error_step=1,
            assumptions=[
                Assumption(
                    text="negative inputs are valid",
                    justified=False,
                ),
            ],
            counterfactual_fail=True,
            final_answer_correct=False,
        )

        # 7. Compute score
        score = Scorer.compute_score(
            is_correct=False,
            reasoning_flawed=True,
            assumption_errors=1,
            counterfactual_fail=True,
            model_disagreement=True,
        )
        # 2 + 3 + 1 + 2 + 1 = 9
        assert score == 9
        severity = Scorer.severity(score)
        assert severity == Severity.CRITICAL

        # 8. Assemble full evaluation result
        result = EvaluationResult(
            prompt_id=prompt.prompt_id,
            failure_type=prompt.failure_type,
            prompt_text=prompt.prompt_text,
            models={
                "model-a": model_a,
                "model-b": model_b,
            },
            validation=validation,
            score=score,
            severity=severity,
        )

        # 9. Store and read back
        store.append(result)
        loaded = store.read_all()
        assert len(loaded) == 1
        assert loaded[0].prompt_id == prompt.prompt_id
        assert loaded[0].score == 9
        assert loaded[0].severity == Severity.CRITICAL
        assert loaded[0].models["model-b"].is_correct is False
        assert loaded[0].validation.assumptions[0].justified is False

        # 10. Verify hard-case extraction
        hard_cases = store.read_by_min_score(6)
        assert len(hard_cases) == 1
        assert hard_cases[0].prompt_id == prompt.prompt_id
```

- [ ] **Step 2: Run integration test to verify it fails**

Run: `uv run pytest tests/test_integration.py -v`
Expected: FAIL with `ImportError` (package init doesn't re-export yet)

- [ ] **Step 3: Update package __init__.py with public API**

Modify `reasonbench/__init__.py`:

```python
"""Fallax — LLM Adversarial Reasoning Evaluation System."""

from .models import (
    Assumption,
    EvaluationResult,
    FailureRecord,
    ModelResponse,
    Prompt,
    ValidationResult,
)
from .scoring import Scorer
from .storage import JsonlStore
from .taxonomy import (
    FAILURE_CATEGORY_MAP,
    FailureCategory,
    FailureType,
    Severity,
    get_category,
)
from .templates import DISTRIBUTION, TemplateRegistry
from .validators import ValidatorPack

__all__ = [
    "Assumption",
    "DISTRIBUTION",
    "EvaluationResult",
    "FAILURE_CATEGORY_MAP",
    "FailureCategory",
    "FailureRecord",
    "FailureType",
    "JsonlStore",
    "ModelResponse",
    "Prompt",
    "Scorer",
    "Severity",
    "TemplateRegistry",
    "ValidationResult",
    "ValidatorPack",
    "get_category",
]
```

- [ ] **Step 4: Run integration test to verify it passes**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS (taxonomy: 11, models: 12, templates: 13, validators: 14, scoring: 14, storage: 7, integration: 1 = ~72 tests)

- [ ] **Step 6: Commit**

```bash
git add reasonbench/__init__.py tests/test_integration.py
git commit -m "feat: wire up public API and add end-to-end integration test"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | Scaffolding | 4 created | 0 |
| 2 | Taxonomy | 2 created | 11 |
| 3 | Models | 2 created | 12 |
| 4 | Templates | 2 created | 13 |
| 5 | Validators | 2 created | 14 |
| 6 | Scoring | 2 created | 14 |
| 7 | Storage | 2 created | 7 |
| 8 | Integration | 1 modified, 1 created | 1 |
| **Total** | | **16 files** | **~72 tests** |

**Phase 1 delivers:** A fully tested foundation with stable data contracts that Phase 2 (multi-agent orchestration, model runners, 100+ prompt generation) will build on top of. All interfaces are defined by Pydantic models — Phase 2 agents will consume and produce these types.
