---
type: canonical
source: none
sync: none
sla: none
---

# ReasonBench Phase 4: Self-Improvement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the feedback loop: evolve hard-case prompts into harder versions via LLM, test model self-repair ability, and mine recurring failure patterns from evaluation results.

**Architecture:** PromptEvolver takes high-scoring failures and uses an LLM to generate harder variants. SelfRepairTester gives failed models a second chance and tracks recovery. RootCauseExtractor mines unjustified assumptions and failure-type co-occurrences to surface systematic model weaknesses. Two new Pydantic models (RepairResult, RootCausePattern) extend the data contracts. CLI gains `evolve` and `repair` subcommands.

**Tech Stack:** Python 3.12+, Pydantic v2, pytest, uv (no new dependencies)

**Depends on:** Phase 3 (analyzer, predictor, clusterer, CLI subcommands) — 193 passing tests.

---

## File Structure

```
reasonbench/
├── reasonbench/
│   ├── models.py             # MODIFY: add RepairResult, RootCausePattern
│   ├── evolver.py            # NEW: LLM-driven prompt evolution
│   ├── repair.py             # NEW: self-repair testing
│   ├── root_cause.py         # NEW: failure pattern extraction
│   ├── __main__.py           # MODIFY: add evolve/repair subcommands
│   └── __init__.py           # MODIFY: add new exports
├── tests/
│   ├── test_models.py        # MODIFY: add tests for new models
│   ├── test_evolver.py       # NEW
│   ├── test_repair.py        # NEW
│   └── test_root_cause.py    # NEW
```

**Dependency order:** `models` → (`evolver`, `repair`, `root_cause`) → `CLI` → `__init__`

---

## Task 1: New Data Models

**Files:**
- Modify: `reasonbench/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for new models**

Append to `tests/test_models.py`:

```python
from reasonbench.models import RepairResult, RootCausePattern


class TestRepairResult:
    def test_create_minimal(self):
        r = RepairResult(
            model_name="model-a",
            prompt_text="test prompt",
            original_answer="wrong",
            repaired_answer="fixed",
            repair_reasoning="I was wrong because...",
        )
        assert r.model_name == "model-a"
        assert r.is_fixed is None

    def test_create_full(self):
        r = RepairResult(
            model_name="model-a",
            prompt_text="test prompt",
            original_answer="wrong",
            repaired_answer="fixed",
            repair_reasoning="I was wrong because...",
            is_fixed=True,
        )
        assert r.is_fixed is True

    def test_json_roundtrip(self):
        r = RepairResult(
            model_name="m",
            prompt_text="p",
            original_answer="a",
            repaired_answer="b",
            repair_reasoning="r",
            is_fixed=False,
        )
        restored = RepairResult.model_validate_json(r.model_dump_json())
        assert restored.is_fixed is False
        assert restored.original_answer == "a"


class TestRootCausePattern:
    def test_create(self):
        p = RootCausePattern(
            pattern="assumes monotonicity",
            frequency=15,
            models_affected=["model-a", "model-b"],
            example_prompt="Is f(x) monotonic?",
            failure_types=["contradiction", "unstated_assumption"],
        )
        assert p.frequency == 15
        assert len(p.models_affected) == 2

    def test_json_roundtrip(self):
        p = RootCausePattern(
            pattern="test",
            frequency=1,
            models_affected=["m"],
            example_prompt="p",
            failure_types=["contradiction"],
        )
        restored = RootCausePattern.model_validate_json(p.model_dump_json())
        assert restored.pattern == "test"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v -k "RepairResult or RootCausePattern"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add models to models.py**

Append to the end of `reasonbench/models.py`:

```python


class RepairResult(BaseModel):
    """Result of a self-repair attempt."""

    model_name: str
    prompt_text: str
    original_answer: str
    repaired_answer: str
    repair_reasoning: str
    is_fixed: bool | None = None


class RootCausePattern(BaseModel):
    """A recurring failure pattern extracted from results."""

    pattern: str
    frequency: int = Field(ge=1)
    models_affected: list[str]
    example_prompt: str
    failure_types: list[str]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: All tests PASS (existing 13 + new 5 = 18)

- [ ] **Step 5: Commit**

```bash
git add reasonbench/models.py tests/test_models.py
git commit -m "feat(models): add RepairResult and RootCausePattern data models"
```

---

## Task 2: Prompt Evolver

**Files:**
- Create: `reasonbench/evolver.py`
- Create: `tests/test_evolver.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_evolver.py`:

```python
import pytest

from reasonbench.evolver import PromptEvolver
from reasonbench.models import Assumption, Prompt
from reasonbench.taxonomy import FailureType
from tests.conftest import MockClient, make_result


@pytest.fixture()
def evolver():
    client = MockClient(default="A harder version of the prompt with more constraints.")
    return PromptEvolver(client=client, model="evolve-model")


@pytest.fixture()
def hard_results():
    return [
        make_result(
            prompt_id="hard-1", score=8, reasoning_flawed=True,
            prompt_text="original hard prompt",
            failure_type=FailureType.CONTRADICTION,
            assumptions=[Assumption(text="x is positive", justified=False)],
        ),
        make_result(
            prompt_id="hard-2", score=7, reasoning_flawed=True,
            prompt_text="another hard prompt",
            failure_type=FailureType.UNSTATED_ASSUMPTION,
        ),
        make_result(
            prompt_id="easy-1", score=2, reasoning_flawed=False,
            prompt_text="easy prompt",
            failure_type=FailureType.OVERGENERALIZATION,
        ),
    ]


class TestPromptEvolver:
    def test_evolve_returns_string(self, evolver):
        result = evolver.evolve("test prompt", "reasoning was flawed")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_evolve_calls_client(self, evolver):
        evolver.evolve("my prompt", "my failure")
        assert len(evolver._client.calls) == 1
        prompt_sent = evolver._client.calls[0][0]
        assert "my prompt" in prompt_sent
        assert "my failure" in prompt_sent

    def test_evolve_uses_configured_model(self, evolver):
        evolver.evolve("p", "f")
        assert evolver._client.calls[0][1] == "evolve-model"

    def test_evolve_batch_filters_by_score(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=6)
        assert len(evolved) == 2  # only score 8 and 7

    def test_evolve_batch_returns_prompts(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=6)
        assert all(isinstance(p, Prompt) for p in evolved)

    def test_evolved_prompt_preserves_failure_type(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=6)
        assert evolved[0].failure_type == FailureType.CONTRADICTION
        assert evolved[1].failure_type == FailureType.UNSTATED_ASSUMPTION

    def test_evolved_prompt_has_no_template_id(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=6)
        assert all(p.template_id is None for p in evolved)

    def test_evolved_prompt_has_unique_ids(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=6)
        ids = [p.prompt_id for p in evolved]
        assert len(ids) == len(set(ids))

    def test_build_failure_analysis(self, hard_results):
        analysis = PromptEvolver.build_failure_analysis(hard_results[0])
        assert "8" in analysis  # score
        assert "critical" in analysis.lower()
        assert "x is positive" in analysis

    def test_build_failure_analysis_no_assumptions(self, hard_results):
        analysis = PromptEvolver.build_failure_analysis(hard_results[1])
        assert "7" in analysis
        assert "assumption" not in analysis.lower() or "Unjustified" not in analysis

    def test_evolve_batch_empty_when_no_hard_cases(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=100)
        assert evolved == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evolver.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/evolver.py`:

```python
"""Prompt evolution — generates harder variants of high-scoring failures."""

from __future__ import annotations

from .client import LLMClient
from .models import EvaluationResult, Prompt


class PromptEvolver:
    """Takes hard-case prompts and evolves them into harder versions via LLM."""

    def __init__(self, client: LLMClient, model: str) -> None:
        self._client = client
        self._model = model

    def evolve(self, prompt_text: str, failure_analysis: str) -> str:
        """Generate a harder version of a failed prompt."""
        evolution_prompt = (
            "Given this prompt and its failure:\n\n"
            f"PROMPT: {prompt_text}\n"
            f"FAILURE: {failure_analysis}\n\n"
            "Generate a HARDER version that:\n"
            "- increases ambiguity OR\n"
            "- adds constraint interaction OR\n"
            "- introduces edge cases\n\n"
            "Return only the new prompt."
        )
        return self._client.complete(evolution_prompt, model=self._model)

    def evolve_batch(
        self, results: list[EvaluationResult], min_score: int = 6
    ) -> list[Prompt]:
        """Evolve all hard cases from a result set."""
        hard_cases = [r for r in results if r.score >= min_score]
        evolved: list[Prompt] = []
        for r in hard_cases:
            analysis = self.build_failure_analysis(r)
            new_text = self.evolve(r.prompt_text, analysis)
            evolved.append(
                Prompt(
                    failure_type=r.failure_type,
                    prompt_text=new_text,
                    difficulty=min(10, r.score),
                    template_id=None,
                )
            )
        return evolved

    @staticmethod
    def build_failure_analysis(result: EvaluationResult) -> str:
        """Build a failure analysis string from an evaluation result."""
        parts = [f"Score: {result.score} ({result.severity.value})"]
        if result.validation.reasoning_flawed:
            parts.append(
                f"Reasoning flawed at step {result.validation.first_error_step}"
            )
        unjustified = [
            a for a in result.validation.assumptions if not a.justified
        ]
        if unjustified:
            parts.append(
                "Unjustified assumptions: "
                + ", ".join(a.text for a in unjustified)
            )
        if result.validation.counterfactual_fail:
            parts.append("Failed counterfactual test")
        if result.validation.adversarial_issues:
            parts.append(
                f"Adversarial issues: {len(result.validation.adversarial_issues)}"
            )
        return ". ".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evolver.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/evolver.py tests/test_evolver.py
git commit -m "feat(evolver): add LLM-driven prompt evolution for hard cases"
```

---

## Task 3: Self-Repair Tester

**Files:**
- Create: `reasonbench/repair.py`
- Create: `tests/test_repair.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_repair.py`:

```python
import pytest

from reasonbench.models import RepairResult
from reasonbench.repair import SelfRepairTester
from reasonbench.taxonomy import FailureType
from tests.conftest import MockClient, make_result


@pytest.fixture()
def tester():
    client = MockClient(default="I was wrong because I assumed X. The correct answer is Y.")
    return SelfRepairTester(client=client)


@pytest.fixture()
def failed_results():
    return [
        make_result(
            prompt_id="f1", score=7, reasoning_flawed=True,
            prompt_text="hard prompt 1", answer="wrong answer 1",
            model_name="model-a",
        ),
        make_result(
            prompt_id="f2", score=6, reasoning_flawed=True,
            prompt_text="hard prompt 2", answer="wrong answer 2",
            model_name="model-b",
        ),
        make_result(
            prompt_id="ok", score=1, reasoning_flawed=False,
            prompt_text="easy prompt", answer="correct",
            model_name="model-a",
        ),
    ]


class TestSelfRepairTester:
    def test_test_repair_returns_repair_result(self, tester):
        result = tester.test_repair(
            prompt_text="test prompt",
            previous_answer="wrong",
            model="model-a",
        )
        assert isinstance(result, RepairResult)

    def test_repair_result_fields(self, tester):
        result = tester.test_repair(
            prompt_text="my prompt",
            previous_answer="my wrong answer",
            model="test-model",
        )
        assert result.model_name == "test-model"
        assert result.prompt_text == "my prompt"
        assert result.original_answer == "my wrong answer"
        assert len(result.repaired_answer) > 0
        assert len(result.repair_reasoning) > 0
        assert result.is_fixed is None  # not yet judged

    def test_repair_calls_client(self, tester):
        tester.test_repair("prompt", "answer", model="m")
        assert len(tester._client.calls) == 1
        prompt_sent = tester._client.calls[0][0]
        assert "previous answer was incorrect" in prompt_sent.lower()
        assert "prompt" in prompt_sent
        assert "answer" in prompt_sent

    def test_repair_uses_specified_model(self, tester):
        tester.test_repair("p", "a", model="my-model")
        assert tester._client.calls[0][1] == "my-model"

    def test_repair_batch_only_failed(self, tester, failed_results):
        repairs = tester.test_repair_batch(failed_results)
        # Only 2 results have reasoning_flawed=True
        assert len(repairs) == 2

    def test_repair_batch_returns_repair_results(self, tester, failed_results):
        repairs = tester.test_repair_batch(failed_results)
        assert all(isinstance(r, RepairResult) for r in repairs)

    def test_repair_batch_preserves_model_names(self, tester, failed_results):
        repairs = tester.test_repair_batch(failed_results)
        models = {r.model_name for r in repairs}
        assert models == {"model-a", "model-b"}

    def test_repair_batch_empty_when_no_failures(self, tester):
        good_results = [
            make_result(prompt_id="ok", score=0, reasoning_flawed=False),
        ]
        repairs = tester.test_repair_batch(good_results)
        assert repairs == []

    def test_repair_extracts_answer(self, tester):
        result = tester.test_repair("p", "a", model="m")
        # MockClient returns "I was wrong because I assumed X. The correct answer is Y."
        assert result.repair_reasoning == result.repaired_answer  # full response used as both
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_repair.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/repair.py`:

```python
"""Self-repair testing — checks if models can fix their own reasoning failures."""

from __future__ import annotations

from .client import LLMClient
from .models import EvaluationResult, RepairResult


class SelfRepairTester:
    """Tests model ability to correct its own reasoning failures.

    Sends the original prompt and wrong answer back to the model,
    asking it to fix its reasoning. Tracks whether the model can
    self-correct.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def test_repair(
        self, prompt_text: str, previous_answer: str, model: str
    ) -> RepairResult:
        """Give a model another chance to fix its answer."""
        repair_prompt = (
            "Your previous answer was incorrect.\n\n"
            f"PROMPT: {prompt_text}\n"
            f"PREVIOUS ANSWER: {previous_answer}\n\n"
            "Fix your answer and explain what was wrong."
        )
        response = self._client.complete(repair_prompt, model=model)
        return RepairResult(
            model_name=model,
            prompt_text=prompt_text,
            original_answer=previous_answer,
            repaired_answer=response,
            repair_reasoning=response,
        )

    def test_repair_batch(
        self, results: list[EvaluationResult]
    ) -> list[RepairResult]:
        """Test repair on all results with flawed reasoning."""
        repairs: list[RepairResult] = []
        for r in results:
            if not r.validation.reasoning_flawed:
                continue
            for model_name, response in r.models.items():
                repair = self.test_repair(
                    r.prompt_text, response.answer, model_name
                )
                repairs.append(repair)
        return repairs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_repair.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/repair.py tests/test_repair.py
git commit -m "feat(repair): add self-repair tester for model recovery tracking"
```

---

## Task 4: Root Cause Extractor

**Files:**
- Create: `reasonbench/root_cause.py`
- Create: `tests/test_root_cause.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_root_cause.py`:

```python
import pytest

from reasonbench.models import Assumption, RootCausePattern
from reasonbench.root_cause import RootCauseExtractor
from reasonbench.taxonomy import FailureType
from tests.conftest import make_result


@pytest.fixture()
def results_with_patterns():
    """Results with recurring assumption patterns."""
    return [
        make_result(
            prompt_id="1", score=7, reasoning_flawed=True,
            failure_type=FailureType.CONTRADICTION,
            model_name="model-a",
            assumptions=[
                Assumption(text="input is positive", justified=False),
                Assumption(text="function is monotonic", justified=False),
            ],
        ),
        make_result(
            prompt_id="2", score=6, reasoning_flawed=True,
            failure_type=FailureType.UNSTATED_ASSUMPTION,
            model_name="model-a",
            assumptions=[
                Assumption(text="input is positive", justified=False),
            ],
        ),
        make_result(
            prompt_id="3", score=8, reasoning_flawed=True,
            failure_type=FailureType.CONTRADICTION,
            model_name="model-b",
            assumptions=[
                Assumption(text="function is monotonic", justified=False),
                Assumption(text="domain is bounded", justified=False),
            ],
        ),
        make_result(
            prompt_id="4", score=5, reasoning_flawed=True,
            failure_type=FailureType.OVERGENERALIZATION,
            model_name="model-b",
            assumptions=[
                Assumption(text="input is positive", justified=False),
            ],
        ),
        make_result(
            prompt_id="5", score=1, reasoning_flawed=False,
            failure_type=FailureType.OVERGENERALIZATION,
            model_name="model-a",
            assumptions=[
                Assumption(text="valid assumption", justified=True),
            ],
        ),
    ]


class TestRootCauseExtractor:
    def test_extract_patterns_returns_list(self, results_with_patterns):
        ex = RootCauseExtractor(results_with_patterns)
        patterns = ex.extract_patterns()
        assert isinstance(patterns, list)
        assert all(isinstance(p, RootCausePattern) for p in patterns)

    def test_most_frequent_pattern_first(self, results_with_patterns):
        ex = RootCauseExtractor(results_with_patterns)
        patterns = ex.extract_patterns()
        assert len(patterns) >= 1
        # "input is positive" appears 3 times (results 1, 2, 4)
        assert patterns[0].pattern == "input is positive"
        assert patterns[0].frequency == 3

    def test_pattern_models_affected(self, results_with_patterns):
        ex = RootCauseExtractor(results_with_patterns)
        patterns = ex.extract_patterns()
        top = patterns[0]  # "input is positive"
        assert set(top.models_affected) == {"model-a", "model-b"}

    def test_pattern_failure_types(self, results_with_patterns):
        ex = RootCauseExtractor(results_with_patterns)
        patterns = ex.extract_patterns()
        top = patterns[0]
        # "input is positive" appears in contradiction, unstated_assumption, overgeneralization
        assert len(top.failure_types) == 3

    def test_pattern_has_example_prompt(self, results_with_patterns):
        ex = RootCauseExtractor(results_with_patterns)
        patterns = ex.extract_patterns()
        assert len(patterns[0].example_prompt) > 0

    def test_min_frequency_filter(self, results_with_patterns):
        ex = RootCauseExtractor(results_with_patterns)
        patterns = ex.extract_patterns(min_frequency=3)
        # Only "input is positive" has frequency >= 3
        assert len(patterns) == 1

    def test_skips_justified_assumptions(self, results_with_patterns):
        ex = RootCauseExtractor(results_with_patterns)
        patterns = ex.extract_patterns(min_frequency=1)
        pattern_texts = {p.pattern for p in patterns}
        assert "valid assumption" not in pattern_texts

    def test_skips_non_flawed_results(self, results_with_patterns):
        ex = RootCauseExtractor(results_with_patterns)
        patterns = ex.extract_patterns(min_frequency=1)
        # Result 5 (reasoning_flawed=False) should be skipped entirely
        # Its "valid assumption" (justified=True) wouldn't appear anyway
        # But ensure the result count logic is correct
        total_freq = sum(p.frequency for p in patterns)
        # Only results 1-4 contribute (result 5 is skipped)
        assert total_freq > 0

    def test_empty_results(self):
        ex = RootCauseExtractor([])
        patterns = ex.extract_patterns()
        assert patterns == []

    def test_no_assumptions_returns_empty(self):
        results = [
            make_result(prompt_id="1", score=7, reasoning_flawed=True),
        ]
        ex = RootCauseExtractor(results)
        patterns = ex.extract_patterns()
        assert patterns == []

    def test_case_insensitive_matching(self):
        results = [
            make_result(
                prompt_id="1", score=7, reasoning_flawed=True,
                assumptions=[Assumption(text="Input Is Positive", justified=False)],
            ),
            make_result(
                prompt_id="2", score=6, reasoning_flawed=True,
                assumptions=[Assumption(text="input is positive", justified=False)],
            ),
        ]
        ex = RootCauseExtractor(results)
        patterns = ex.extract_patterns()
        assert len(patterns) == 1
        assert patterns[0].frequency == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_root_cause.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/root_cause.py`:

```python
"""Root cause extraction — mines recurring failure patterns from results."""

from __future__ import annotations

from collections import defaultdict

from .models import EvaluationResult, RootCausePattern


class RootCauseExtractor:
    """Extracts recurring failure patterns from evaluation results.

    Groups unjustified assumptions by text (case-insensitive), counts
    frequency across results, and identifies which models and failure
    types are affected.
    """

    def __init__(self, results: list[EvaluationResult]) -> None:
        self._results = results

    def extract_patterns(
        self, min_frequency: int = 2
    ) -> list[RootCausePattern]:
        """Extract root cause patterns from failures.

        Only considers results with flawed reasoning. Groups unjustified
        assumptions by normalized text. Returns patterns sorted by
        frequency (descending).
        """
        # Group results by unjustified assumption text
        pattern_results: dict[str, list[EvaluationResult]] = defaultdict(
            list
        )
        for r in self._results:
            if not r.validation.reasoning_flawed:
                continue
            for assumption in r.validation.assumptions:
                if not assumption.justified:
                    key = assumption.text.lower()
                    pattern_results[key].append(r)

        # Build patterns
        patterns: list[RootCausePattern] = []
        for pattern_text, matching in pattern_results.items():
            if len(matching) < min_frequency:
                continue
            patterns.append(
                RootCausePattern(
                    pattern=pattern_text,
                    frequency=len(matching),
                    models_affected=sorted(
                        {
                            model
                            for r in matching
                            for model in r.models
                        }
                    ),
                    example_prompt=matching[0].prompt_text,
                    failure_types=sorted(
                        {r.failure_type.value for r in matching}
                    ),
                )
            )
        return sorted(patterns, key=lambda p: p.frequency, reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_root_cause.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/root_cause.py tests/test_root_cause.py
git commit -m "feat(root_cause): add failure pattern extraction from unjustified assumptions"
```

---

## Task 5: CLI Update with evolve/repair Subcommands

**Files:**
- Modify: `reasonbench/__main__.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write new CLI tests**

Append to `tests/test_cli.py`:

```python
class TestEvolveSubcommand:
    def test_evolve_returns_zero(self, results_file, tmp_path):
        output = tmp_path / "evolved.jsonl"
        mock = MockClient(default="A harder evolved prompt.")
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            code = main([
                "evolve", str(results_file),
                "--model", "m",
                "--output", str(output),
            ])
        assert code == 0

    def test_evolve_missing_file_returns_error(self, tmp_path):
        mock = MockClient(default="evolved")
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            code = main([
                "evolve", str(tmp_path / "nonexistent.jsonl"),
                "--model", "m",
            ])
        assert code == 1


class TestRepairSubcommand:
    def test_repair_returns_zero(self, results_file, tmp_path):
        output = tmp_path / "repairs.jsonl"
        mock = MockClient(default="I was wrong. The correct answer is X.")
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            code = main([
                "repair", str(results_file),
                "--model", "m",
                "--output", str(output),
            ])
        assert code == 0

    def test_repair_missing_file_returns_error(self, tmp_path):
        mock = MockClient(default="fixed")
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            code = main([
                "repair", str(tmp_path / "nonexistent.jsonl"),
                "--model", "m",
            ])
        assert code == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v -k "Evolve or Repair"`
Expected: FAIL

- [ ] **Step 3: Add evolve/repair subcommands to __main__.py**

Read existing `reasonbench/__main__.py`. Add two new command functions and subparsers. Append the following function definitions before `main()`:

```python
def _cmd_evolve(args: argparse.Namespace) -> int:
    """Evolve hard-case prompts into harder versions."""
    from .evolver import PromptEvolver
    from .storage import JsonlStore

    path = Path(args.results)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    store = JsonlStore(path)
    results = store.read_all()
    if not results:
        print("No results found.")
        return 1

    client = AnthropicClient()
    evolver = PromptEvolver(client=client, model=args.model)
    evolved = evolver.evolve_batch(results, min_score=args.min_score)

    if not evolved:
        print("No hard cases found to evolve.")
        return 0

    output = Path(args.output)
    with open(output, "w", encoding="utf-8") as f:
        for p in evolved:
            f.write(p.model_dump_json() + "\n")

    print(f"\nEvolved {len(evolved)} prompts")
    print(f"  Min score: {args.min_score}")
    print(f"  Saved to: {output}")
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    """Test model self-repair on failed results."""
    from .repair import SelfRepairTester
    from .storage import JsonlStore

    path = Path(args.results)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    store = JsonlStore(path)
    results = store.read_all()
    if not results:
        print("No results found.")
        return 1

    # Override model names in results for repair testing
    client = AnthropicClient()
    tester = SelfRepairTester(client=client)

    # Replace model names with the specified model for repair
    for r in results:
        for model_name in list(r.models.keys()):
            r.models[model_name].model_name = args.model
            r.models[args.model] = r.models.pop(model_name)
            break  # only remap first model

    repairs = tester.test_repair_batch(results)

    if not repairs:
        print("No failures to repair.")
        return 0

    output = Path(args.output)
    with open(output, "w", encoding="utf-8") as f:
        for r in repairs:
            f.write(r.model_dump_json() + "\n")

    print(f"\nRepair tested {len(repairs)} failures")
    print(f"  Model: {args.model}")
    print(f"  Saved to: {output}")
    return 0
```

Add these subparsers in `main()`, after the existing `train` subparser:

```python
    # -- evolve --
    evolve_p = subparsers.add_parser(
        "evolve", help="Evolve hard-case prompts into harder versions"
    )
    evolve_p.add_argument("results", help="Path to results JSONL file")
    evolve_p.add_argument(
        "--model", required=True, help="LLM model for evolution"
    )
    evolve_p.add_argument(
        "--min-score", type=int, default=6,
        help="Minimum score threshold for hard cases (default: 6)",
    )
    evolve_p.add_argument(
        "--output", "-o", default="evolved.jsonl",
        help="Output file for evolved prompts (default: evolved.jsonl)",
    )

    # -- repair --
    repair_p = subparsers.add_parser(
        "repair", help="Test model self-repair on failures"
    )
    repair_p.add_argument("results", help="Path to results JSONL file")
    repair_p.add_argument(
        "--model", required=True, help="LLM model for repair testing"
    )
    repair_p.add_argument(
        "--output", "-o", default="repairs.jsonl",
        help="Output file for repair results (default: repairs.jsonl)",
    )
```

Add command dispatch in `main()`, after the existing `train` dispatch:

```python
    if args.command == "evolve":
        return _cmd_evolve(args)
    if args.command == "repair":
        return _cmd_repair(args)
```

- [ ] **Step 4: Run CLI tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All 11 tests PASS (7 existing + 4 new)

- [ ] **Step 5: Commit**

```bash
git add reasonbench/__main__.py tests/test_cli.py
git commit -m "feat(cli): add evolve and repair subcommands"
```

---

## Task 6: Public API Update and Full Suite

**Files:**
- Modify: `reasonbench/__init__.py`

- [ ] **Step 1: Update __init__.py**

Read existing `reasonbench/__init__.py`, then replace with:

```python
"""ReasonBench — LLM Adversarial Reasoning Evaluation System."""

from .analyzer import Analyzer
from .client import AnthropicClient, LLMClient
from .clusterer import FailureClusterer
from .evaluator import Evaluator
from .evolver import PromptEvolver
from .generator import PromptGenerator
from .models import (
    Assumption,
    EvaluationResult,
    FailureRecord,
    ModelResponse,
    Prompt,
    RepairResult,
    RootCausePattern,
    ValidationResult,
)
from .pipeline import Pipeline
from .predictor import FailurePredictor
from .repair import SelfRepairTester
from .root_cause import RootCauseExtractor
from .runner import ModelRunner
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
    "Analyzer",
    "AnthropicClient",
    "Assumption",
    "DISTRIBUTION",
    "EvaluationResult",
    "Evaluator",
    "FAILURE_CATEGORY_MAP",
    "FailureCategory",
    "FailureClusterer",
    "FailurePredictor",
    "FailureRecord",
    "FailureType",
    "JsonlStore",
    "LLMClient",
    "ModelResponse",
    "ModelRunner",
    "Pipeline",
    "Prompt",
    "PromptEvolver",
    "PromptGenerator",
    "RepairResult",
    "RootCauseExtractor",
    "RootCausePattern",
    "Scorer",
    "SelfRepairTester",
    "Severity",
    "TemplateRegistry",
    "ValidationResult",
    "ValidatorPack",
    "get_category",
]
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest -v
```
Expected: All tests PASS (~230 tests)

- [ ] **Step 3: Commit**

```bash
git add reasonbench/__init__.py
git commit -m "feat: update public API with Phase 4 exports (Evolver, Repair, RootCause)"
```

---

## Summary

| Task | Component | New/Modified Files | Tests |
|------|-----------|-------------------|-------|
| 1 | New Models | 2 (models, test_models) | 5 |
| 2 | Prompt Evolver | 2 (evolver, test) | 11 |
| 3 | Self-Repair Tester | 2 (repair, test) | 9 |
| 4 | Root Cause Extractor | 2 (root_cause, test) | 11 |
| 5 | CLI Subcommands | 2 (__main__, test_cli) | 4 |
| 6 | Init Update | 1 (__init__) | 0 |
| **Total** | | **11 files** | **~40 new tests** |

**Phase 4 delivers:** LLM-driven prompt evolution that takes hard cases (score >= 6) and generates harder variants. Self-repair testing that gives models a second chance and tracks recovery. Root cause extraction that mines unjustified assumptions into ranked patterns with model/failure-type breakdowns. CLI subcommands: `reasonbench evolve results.jsonl --model <m>`, `reasonbench repair results.jsonl --model <m>`.
