# ReasonBench Phase 2: Scale — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working end-to-end evaluation pipeline that generates adversarial prompts, runs them against LLMs, validates reasoning with a judge model, scores failures, and stores results as JSONL.

**Architecture:** Protocol-based LLM client enables testability (MockClient) and extensibility (any provider). PromptGenerator draws from JSON parameter banks and renders templates from Phase 1. ModelRunner sends prompts to target models. Evaluator runs the 5-validator pack through a judge LLM and parses structured JSON responses. Pipeline orchestrates the loop. CLI exposes it via `python -m reasonbench`. Model names are never hardcoded — all configurable via CLI args or environment variables. **Note:** In multi-model mode, reasoning validation/scoring reflects the first model's response; disagreement is computed across all models.

**Tech Stack:** Python 3.12+, Anthropic SDK, Pydantic v2 (Phase 1), pytest, uv

**Depends on:** Phase 1 foundation (taxonomy, models, templates, validators, scoring, storage) — all stable and tested.

---

## File Structure

```
reasonbench/
├── reasonbench/
│   ├── __init__.py           # MODIFY: add new exports
│   ├── taxonomy.py           # existing (unchanged)
│   ├── models.py             # existing (unchanged)
│   ├── templates.py          # existing (unchanged)
│   ├── validators.py         # existing (unchanged)
│   ├── scoring.py            # existing (unchanged)
│   ├── storage.py            # existing (unchanged)
│   ├── client.py             # NEW: LLMClient protocol + AnthropicClient
│   ├── generator.py          # NEW: PromptGenerator + parameter bank loader
│   ├── runner.py             # NEW: ModelRunner for multi-model evaluation
│   ├── evaluator.py          # NEW: Validator orchestrator via judge LLM
│   ├── pipeline.py           # NEW: Main evaluation loop
│   ├── __main__.py           # NEW: CLI entry point
│   └── data/                 # NEW: parameter bank JSON files
│       ├── implicit_assumption_trap.json
│       ├── contradictory_constraints.json
│       ├── false_analogy_trap.json
│       ├── recursive_definition_break.json
│       ├── multi_step_dependency.json
│       ├── edge_case_inversion.json
│       ├── ambiguous_spec_trap.json
│       ├── overconstrained_optimization.json
│       ├── hidden_variable_trap.json
│       └── self_consistency_trap.json
├── tests/
│   ├── conftest.py           # NEW: MockClient + shared fixtures
│   ├── test_client.py        # NEW
│   ├── test_generator.py     # NEW
│   ├── test_runner.py        # NEW
│   ├── test_evaluator.py     # NEW
│   └── test_pipeline.py      # NEW
```

**Dependency order:** `client` → (`generator`, `runner`, `evaluator`) → `pipeline` → `__main__`

---

## Task 1: Dependencies and MockClient

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add anthropic SDK dependency**

Update `pyproject.toml` dependencies:

```toml
dependencies = [
    "pydantic>=2.0,<3",
    "anthropic>=0.40",
]
```

- [ ] **Step 2: Install updated dependencies**

```bash
uv sync --dev
```

- [ ] **Step 3: Create shared test fixtures**

Create `tests/conftest.py`:

```python
"""Shared test fixtures for Phase 2 tests."""

from __future__ import annotations

import pytest


class MockClient:
    """Mock LLM client for testing. Matches LLMClient protocol."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "mock response",
    ) -> None:
        self._responses = responses or {}
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def complete(self, prompt: str, *, model: str) -> str:
        self.calls.append((prompt, model))
        for keyword, response in self._responses.items():
            if keyword in prompt:
                return response
        return self._default


@pytest.fixture()
def mock_client():
    """A basic MockClient returning 'mock response'."""
    return MockClient()


JUDGE_RESPONSES: dict[str, str] = {
    "strict reasoning auditor": (
        '{"reasoning_flawed": true, "first_error_step": 2, '
        '"explanation": "Step 2 assumes without evidence"}'
    ),
    "implicit assumptions": (
        '{"assumptions": [{"text": "x is positive", "justified": false}]}'
    ),
    "Does the reasoning still hold": (
        '{"holds": false, "break_point": "Step 2 no longer valid"}'
    ),
    "prove the reasoning WRONG": (
        '{"issues": ["Step 2 has unjustified leap"], "robust": false}'
    ),
    "Is the final answer correct": '{"correct": false}',
}


@pytest.fixture()
def judge_client():
    """MockClient pre-loaded with judge LLM responses."""
    return MockClient(responses=JUDGE_RESPONSES)


MODEL_RESPONSE_TEXT = (
    "Let me work through this step by step.\n"
    "Step 1: I observe the input.\n"
    "Step 2: I apply the rule.\n"
    "ANSWER: The output is 42"
)


@pytest.fixture()
def model_client():
    """MockClient pre-loaded with model evaluation responses."""
    return MockClient(default=MODEL_RESPONSE_TEXT)
```

- [ ] **Step 4: Verify fixtures load**

```bash
uv run pytest --co -q
```
Expected: existing 84 tests collected, no errors

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/conftest.py
git commit -m "chore: add anthropic SDK dependency and shared test fixtures"
```

---

## Task 2: LLM Client Protocol and AnthropicClient

**Files:**
- Create: `reasonbench/client.py`
- Create: `tests/test_client.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_client.py`:

```python
from unittest.mock import MagicMock

import pytest

from reasonbench.client import AnthropicClient, LLMClient
from tests.conftest import MockClient


class TestLLMClientProtocol:
    def test_mock_client_satisfies_protocol(self):
        client: LLMClient = MockClient()
        result = client.complete("hello", model="test")
        assert isinstance(result, str)


class TestMockClient:
    def test_default_response(self):
        client = MockClient(default="hello world")
        assert client.complete("anything", model="m") == "hello world"

    def test_keyword_response(self):
        client = MockClient(responses={"foo": "bar"})
        assert client.complete("contains foo here", model="m") == "bar"

    def test_records_calls(self):
        client = MockClient()
        client.complete("prompt1", model="model-a")
        client.complete("prompt2", model="model-b")
        assert len(client.calls) == 2
        assert client.calls[0] == ("prompt1", "model-a")
        assert client.calls[1] == ("prompt2", "model-b")

    def test_keyword_priority_first_match(self):
        client = MockClient(responses={"alpha": "first", "beta": "second"})
        result = client.complete("has alpha and beta", model="m")
        assert result == "first"


class TestAnthropicClient:
    def test_complete_calls_api(self):
        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="test response")]
        mock_anthropic.messages.create.return_value = mock_message

        client = AnthropicClient.__new__(AnthropicClient)
        client._client = mock_anthropic
        client._max_tokens = 4096

        result = client.complete("test prompt", model="test-model")

        assert result == "test response"
        mock_anthropic.messages.create.assert_called_once_with(
            model="test-model",
            max_tokens=4096,
            messages=[{"role": "user", "content": "test prompt"}],
        )

    def test_complete_extracts_first_content_block(self):
        mock_anthropic = MagicMock()
        block1 = MagicMock(text="first block")
        block2 = MagicMock(text="second block")
        mock_message = MagicMock()
        mock_message.content = [block1, block2]
        mock_anthropic.messages.create.return_value = mock_message

        client = AnthropicClient.__new__(AnthropicClient)
        client._client = mock_anthropic
        client._max_tokens = 4096

        result = client.complete("test", model="m")
        assert result == "first block"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_client.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/client.py`:

```python
"""LLM client protocol and implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM API clients.

    Any object with a `complete(prompt, *, model) -> str` method satisfies this.
    """

    def complete(self, prompt: str, *, model: str) -> str:
        """Send a prompt and return the text response."""
        ...


class AnthropicClient:
    """LLM client using the Anthropic API."""

    def __init__(
        self,
        api_key: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._max_tokens = max_tokens

    def complete(self, prompt: str, *, model: str) -> str:
        """Send a prompt and return the text response."""
        message = self._client.messages.create(
            model=model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_client.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/client.py tests/test_client.py
git commit -m "feat(client): add LLMClient protocol and AnthropicClient"
```

---

## Task 3: Parameter Banks

**Files:**
- Create: `reasonbench/data/implicit_assumption_trap.json`
- Create: `reasonbench/data/contradictory_constraints.json`
- Create: `reasonbench/data/false_analogy_trap.json`
- Create: `reasonbench/data/recursive_definition_break.json`
- Create: `reasonbench/data/multi_step_dependency.json`
- Create: `reasonbench/data/edge_case_inversion.json`
- Create: `reasonbench/data/ambiguous_spec_trap.json`
- Create: `reasonbench/data/overconstrained_optimization.json`
- Create: `reasonbench/data/hidden_variable_trap.json`
- Create: `reasonbench/data/self_consistency_trap.json`

Each file is a JSON array of parameter dictionaries. Keys match the template's `parameters` list exactly. 5 sets per template = 50 total prompt configurations.

- [ ] **Step 1: Create data directory**

```bash
mkdir -p reasonbench/data
```

- [ ] **Step 2: Create implicit_assumption_trap.json**

```json
[
  {
    "rule_a": "inputs must be positive integers",
    "rule_b": "outputs are doubled",
    "edge_case_input": "the input is -3"
  },
  {
    "rule_a": "the system accepts strings of length <= 10",
    "rule_b": "all strings are reversed before output",
    "edge_case_input": "the input is an empty string"
  },
  {
    "rule_a": "only alphabetic characters are valid input",
    "rule_b": "the first character determines the processing mode",
    "edge_case_input": "the input is '123abc'"
  },
  {
    "rule_a": "the queue processes items in FIFO order",
    "rule_b": "items with priority > 5 skip the queue",
    "edge_case_input": "two items arrive simultaneously with priority 5"
  },
  {
    "rule_a": "temperature readings are in Celsius",
    "rule_b": "readings below 0 trigger an alert",
    "edge_case_input": "the reading is exactly 0"
  }
]
```

- [ ] **Step 3: Create contradictory_constraints.json**

```json
[
  {
    "constraint_1": "x must be greater than 10",
    "constraint_2": "x must be less than 5",
    "constraint_3": "x must be a positive integer"
  },
  {
    "constraint_1": "the output must be sorted in ascending order",
    "constraint_2": "the output must preserve the original insertion order",
    "constraint_3": "no element may appear more than once"
  },
  {
    "constraint_1": "the algorithm must run in O(n) time",
    "constraint_2": "the algorithm must sort the input array",
    "constraint_3": "the algorithm must use only O(1) extra memory"
  },
  {
    "constraint_1": "all employees must work exactly 8 hours per day",
    "constraint_2": "no employee may work after 5 PM",
    "constraint_3": "all employees must start after 10 AM"
  },
  {
    "constraint_1": "the container must hold at least 500ml",
    "constraint_2": "the container must weigh less than 100g when empty",
    "constraint_3": "the container must be made of solid steel with walls at least 5mm thick"
  }
]
```

- [ ] **Step 4: Create false_analogy_trap.json**

```json
[
  {
    "well_known_problem": "the traveling salesman problem",
    "key_difference": "the salesman can teleport between any two non-adjacent cities for free, but only once"
  },
  {
    "well_known_problem": "binary search on a sorted array",
    "key_difference": "the array is sorted but contains duplicate elements and you need to find the count of a target value"
  },
  {
    "well_known_problem": "the classic 0/1 knapsack problem",
    "key_difference": "items can be split into fractions, but splitting reduces the item's value-to-weight ratio by 50%"
  },
  {
    "well_known_problem": "Dijkstra's shortest path algorithm",
    "key_difference": "some edges have negative weights, but there are no negative cycles"
  },
  {
    "well_known_problem": "the Monty Hall problem with 3 doors",
    "key_difference": "Monty does not know which door has the prize and opens one at random"
  }
]
```

- [ ] **Step 5: Create recursive_definition_break.json**

```json
[
  {
    "base_case": "f(0) = 1",
    "recursive_rule": "f(n) = f(n-1) + f(n+1) for n > 0"
  },
  {
    "base_case": "f(1) = 0",
    "recursive_rule": "f(n) = n * f(n/2) for all n > 1 (integer division)"
  },
  {
    "base_case": "f(0) = 0, f(1) = 1",
    "recursive_rule": "f(n) = f(n-1) - f(n-2) for n > 1"
  },
  {
    "base_case": "g(1) = 1",
    "recursive_rule": "g(n) = g(n - g(n-1)) for n > 1"
  },
  {
    "base_case": "h(0) = 2",
    "recursive_rule": "h(n) = h(n-1) / h(n-1) for n > 0"
  }
]
```

- [ ] **Step 6: Create multi_step_dependency.json**

```json
[
  {
    "step1": "x = 10 / 2",
    "step2": "y = x - 3 (but if x is even, subtract 4 instead)",
    "step3": "z = y * 2"
  },
  {
    "step1": "Parse the string '12.5abc' as a number (stop at first non-numeric character)",
    "step2": "Round the result to the nearest integer",
    "step3": "Determine if the rounded result is prime"
  },
  {
    "step1": "Sort the list [3, 1, 4, 1, 5] and remove duplicates",
    "step2": "Take the median of the resulting list",
    "step3": "Determine if the median equals the arithmetic mean of the list"
  },
  {
    "step1": "Convert the binary string '10110' to decimal",
    "step2": "Compute the factorial of the result",
    "step3": "Find the sum of the digits of the factorial"
  },
  {
    "step1": "Count the vowels in 'aeiou123aei'",
    "step2": "Raise 2 to the power of that count",
    "step3": "Determine if the result is divisible by the original vowel count"
  }
]
```

- [ ] **Step 7: Create edge_case_inversion.json**

```json
[
  {
    "general_rule": "For any positive integer n, n! > 2^n",
    "edge_case": "n = 1"
  },
  {
    "general_rule": "Quicksort has O(n log n) average time complexity and is faster than insertion sort",
    "edge_case": "the array has exactly 3 elements"
  },
  {
    "general_rule": "The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees",
    "edge_case": "n = 2 (a digon)"
  },
  {
    "general_rule": "Adding more workers to a project reduces the completion time proportionally",
    "edge_case": "the project has a single indivisible task"
  },
  {
    "general_rule": "A hash table provides O(1) average lookup time",
    "edge_case": "all keys hash to the same bucket"
  }
]
```

- [ ] **Step 8: Create ambiguous_spec_trap.json**

```json
[
  {
    "ambiguous_description": "The function returns the 'next' item in the sequence",
    "input": "the sequence [1, 2, 3] at position 3 (0-indexed or 1-indexed?)"
  },
  {
    "ambiguous_description": "Round the number to the nearest integer",
    "input": "the number 2.5 (round half up, half down, or half to even?)"
  },
  {
    "ambiguous_description": "Remove duplicates from the list while preserving order",
    "input": "the list [3, 1, 2, 1, 3, 2] (preserve first or last occurrence?)"
  },
  {
    "ambiguous_description": "The system returns results in alphabetical order",
    "input": "the strings ['apple', 'Apple', '123', 'banana'] (case-sensitive? numbers first?)"
  },
  {
    "ambiguous_description": "Split the string by whitespace",
    "input": "the string '  hello   world  ' (include empty strings? trim first?)"
  }
]
```

- [ ] **Step 9: Create overconstrained_optimization.json**

```json
[
  {
    "objective": "f(x, y) = x + y",
    "constraints": "x >= 0, y >= 0, x + y <= 10, x + y >= 15"
  },
  {
    "objective": "profit = 3x + 5y",
    "constraints": "x + y <= 100, 2x + y <= 150, x >= 60, y >= 60"
  },
  {
    "objective": "the number of items processed per hour",
    "constraints": "each item takes at least 10 minutes to process, the system must process at least 8 items per hour, there is only 1 worker"
  },
  {
    "objective": "total distance traveled",
    "constraints": "visit all 4 cities exactly once, return to start, total distance < 10km, minimum distance between any two cities is 5km"
  },
  {
    "objective": "minimize cost = 2a + 3b",
    "constraints": "a + b >= 100, a <= 30, b <= 60"
  }
]
```

- [ ] **Step 10: Create hidden_variable_trap.json**

```json
[
  {
    "partial_info": "X = 5. The result is X * Y + 3."
  },
  {
    "partial_info": "The area of the rectangle is 24 square units. One side has length 6."
  },
  {
    "partial_info": "The function is f(x) = ax^2 + bx + c. We know f(0) = 3 and f(1) = 7."
  },
  {
    "partial_info": "A car travels from A to B at 60 km/h. The total trip took 2 hours including a stop."
  },
  {
    "partial_info": "The average of three numbers is 10. Two of the numbers are 8 and 12."
  }
]
```

- [ ] **Step 11: Create self_consistency_trap.json**

```json
[
  {
    "reasoning_chain": "Step 1: All mammals are warm-blooded.\nStep 2: A whale is a mammal.\nStep 3: Therefore, a whale is cold-blooded."
  },
  {
    "reasoning_chain": "Step 1: If it rains, the ground gets wet.\nStep 2: The ground is wet.\nStep 3: Therefore, it must have rained."
  },
  {
    "reasoning_chain": "Step 1: x^2 = 4.\nStep 2: Taking the square root of both sides: x = 2.\nStep 3: Therefore, x = 2 is the only solution."
  },
  {
    "reasoning_chain": "Step 1: The set S = {1, 2, 3} has 3 elements.\nStep 2: The power set of S has 2^3 = 8 subsets.\nStep 3: Therefore, S has 8 proper subsets."
  },
  {
    "reasoning_chain": "Step 1: Function f is continuous on [0,1].\nStep 2: f(0) = -1 and f(1) = 1.\nStep 3: By the intermediate value theorem, f(0.5) = 0."
  }
]
```

- [ ] **Step 12: Write data validation tests**

Create `tests/test_data.py`:

```python
"""Validate parameter bank JSON files match template definitions."""

import json
from pathlib import Path

import pytest

from reasonbench.templates import TEMPLATES, TemplateRegistry

DATA_DIR = Path(__file__).resolve().parent.parent / "reasonbench" / "data"


class TestParameterBanks:
    def test_all_templates_have_data_files(self):
        for t in TEMPLATES:
            path = DATA_DIR / f"{t.template_id}.json"
            assert path.exists(), f"Missing data file for {t.template_id}"

    @pytest.mark.parametrize(
        "template",
        TEMPLATES,
        ids=[t.template_id for t in TEMPLATES],
    )
    def test_data_file_is_valid_json_array(self, template):
        path = DATA_DIR / f"{template.template_id}.json"
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 1

    @pytest.mark.parametrize(
        "template",
        TEMPLATES,
        ids=[t.template_id for t in TEMPLATES],
    )
    def test_parameter_keys_match_template(self, template):
        path = DATA_DIR / f"{template.template_id}.json"
        with open(path) as f:
            data = json.load(f)
        expected_keys = set(template.parameters)
        for i, entry in enumerate(data):
            assert set(entry.keys()) == expected_keys, (
                f"{template.template_id}[{i}]: "
                f"expected keys {expected_keys}, got {set(entry.keys())}"
            )

    @pytest.mark.parametrize(
        "template",
        TEMPLATES,
        ids=[t.template_id for t in TEMPLATES],
    )
    def test_parameters_render_without_error(self, template):
        path = DATA_DIR / f"{template.template_id}.json"
        with open(path) as f:
            data = json.load(f)
        registry = TemplateRegistry()
        for entry in data:
            rendered = registry.render(template.template_id, **entry)
            assert len(rendered) > 0
```

- [ ] **Step 13: Run data validation tests**

Run: `uv run pytest tests/test_data.py -v`
Expected: All 31 tests PASS (1 completeness + 10 valid JSON + 10 key match + 10 render)

- [ ] **Step 14: Commit**

```bash
git add reasonbench/data/ tests/test_data.py
git commit -m "feat(data): add parameter banks for all 10 template types (50 prompt configs)"
```

---

## Task 4: PromptGenerator

**Files:**
- Create: `reasonbench/generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_generator.py`:

```python
import json
from pathlib import Path

import pytest

from reasonbench.generator import TEMPLATE_DIFFICULTY, PromptGenerator
from reasonbench.models import Prompt
from reasonbench.taxonomy import FailureType
from reasonbench.templates import DISTRIBUTION


@pytest.fixture()
def params_dir(tmp_path: Path) -> Path:
    """Create a minimal parameter bank for testing."""
    data = [
        {"rule_a": "x > 0", "rule_b": "double it", "edge_case_input": "x = -1"},
        {"rule_a": "len < 5", "rule_b": "reverse", "edge_case_input": "empty string"},
    ]
    (tmp_path / "implicit_assumption_trap.json").write_text(json.dumps(data))

    data2 = [
        {"constraint_1": "a > 10", "constraint_2": "a < 5", "constraint_3": "a > 0"},
    ]
    (tmp_path / "contradictory_constraints.json").write_text(json.dumps(data2))

    return tmp_path


class TestPromptGenerator:
    def test_load_param_banks(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir)
        banks = gen.param_banks
        assert "implicit_assumption_trap" in banks
        assert len(banks["implicit_assumption_trap"]) == 2

    def test_generate_single(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir, seed=42)
        prompts = gen.generate_for_template("implicit_assumption_trap", count=1)
        assert len(prompts) == 1
        p = prompts[0]
        assert isinstance(p, Prompt)
        assert p.failure_type == FailureType.UNSTATED_ASSUMPTION
        assert p.template_id == "implicit_assumption_trap"
        assert "x > 0" in p.prompt_text or "len < 5" in p.prompt_text

    def test_generate_wraps_around_bank(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir, seed=42)
        prompts = gen.generate_for_template("implicit_assumption_trap", count=5)
        assert len(prompts) == 5
        # With 2 bank entries and 5 prompts, params must cycle
        param_sets = [p.parameters for p in prompts]
        assert len(set(p["rule_a"] for p in param_sets)) <= 2

    def test_generate_batch_respects_count(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir, seed=42)
        batch = gen.generate_batch(count=3)
        assert len(batch) == 3

    def test_generate_batch_uses_available_templates_only(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir, seed=42)
        batch = gen.generate_batch(count=10)
        # Only 2 templates have banks in tmp_path
        template_ids = {p.template_id for p in batch}
        assert template_ids <= {"implicit_assumption_trap", "contradictory_constraints"}

    def test_difficulty_from_template_type(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir)
        prompts = gen.generate_for_template("implicit_assumption_trap", count=1)
        assert prompts[0].difficulty == TEMPLATE_DIFFICULTY["implicit_assumption_trap"]

    def test_empty_bank_returns_empty(self, tmp_path):
        gen = PromptGenerator(params_dir=tmp_path)
        prompts = gen.generate_for_template("implicit_assumption_trap", count=5)
        assert prompts == []

    def test_seed_reproducibility(self, params_dir):
        gen1 = PromptGenerator(params_dir=params_dir, seed=42)
        gen2 = PromptGenerator(params_dir=params_dir, seed=42)
        batch1 = gen1.generate_batch(count=5)
        batch2 = gen2.generate_batch(count=5)
        assert [p.prompt_text for p in batch1] == [p.prompt_text for p in batch2]

    def test_unique_prompt_ids(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir, seed=42)
        batch = gen.generate_batch(count=5)
        ids = [p.prompt_id for p in batch]
        assert len(ids) == len(set(ids))


class TestTemplateDifficulty:
    def test_covers_all_templates(self):
        from reasonbench.templates import TEMPLATES

        template_ids = {t.template_id for t in TEMPLATES}
        assert set(TEMPLATE_DIFFICULTY.keys()) == template_ids

    def test_values_in_range(self):
        for difficulty in TEMPLATE_DIFFICULTY.values():
            assert 1 <= difficulty <= 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/generator.py`:

```python
"""Prompt generator using parameter banks and templates."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .models import Prompt
from .templates import DISTRIBUTION, TemplateRegistry

TEMPLATE_DIFFICULTY: dict[str, int] = {
    "implicit_assumption_trap": 6,
    "contradictory_constraints": 7,
    "false_analogy_trap": 7,
    "recursive_definition_break": 8,
    "multi_step_dependency": 7,
    "edge_case_inversion": 5,
    "ambiguous_spec_trap": 6,
    "overconstrained_optimization": 8,
    "hidden_variable_trap": 6,
    "self_consistency_trap": 7,
}


class PromptGenerator:
    """Generates adversarial prompts from templates and parameter banks."""

    def __init__(
        self,
        registry: TemplateRegistry | None = None,
        params_dir: Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._registry = registry or TemplateRegistry()
        self._rng = random.Random(seed)
        self._params_dir = params_dir or Path(__file__).parent / "data"
        self._param_banks: dict[str, list[dict[str, str]]] = {}
        self._load_param_banks()

    @property
    def param_banks(self) -> dict[str, list[dict[str, str]]]:
        """Expose loaded parameter banks (read-only access)."""
        return self._param_banks

    def _load_param_banks(self) -> None:
        """Load parameter banks from JSON files in params_dir."""
        if not self._params_dir.is_dir():
            return
        for template_id in self._registry.template_ids():
            path = self._params_dir / f"{template_id}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self._param_banks[template_id] = json.load(f)

    def generate_for_template(
        self, template_id: str, count: int
    ) -> list[Prompt]:
        """Generate N prompts from a specific template."""
        bank = self._param_banks.get(template_id, [])
        if not bank:
            return []
        template = self._registry.get(template_id)
        difficulty = TEMPLATE_DIFFICULTY.get(template_id, 5)

        prompts: list[Prompt] = []
        for i in range(count):
            params = bank[i % len(bank)]
            prompt_text = self._registry.render(template_id, **params)
            prompts.append(
                Prompt(
                    failure_type=template.failure_target,
                    prompt_text=prompt_text,
                    difficulty=difficulty,
                    template_id=template_id,
                    parameters=params,
                )
            )
        return prompts

    def generate_batch(self, count: int = 100) -> list[Prompt]:
        """Generate a batch of prompts following distribution targets.

        Only generates from templates that have parameter banks loaded.
        Shuffles the result for evaluation variety.
        """
        available = {
            tid: pct
            for tid, pct in DISTRIBUTION.items()
            if tid in self._param_banks
        }
        if not available:
            return []

        # Normalize distribution to available templates
        total_pct = sum(available.values())
        prompts: list[Prompt] = []
        for template_id, pct in available.items():
            n = max(1, round(pct * count / total_pct))
            prompts.extend(self.generate_for_template(template_id, n))

        self._rng.shuffle(prompts)
        return prompts[:count]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generator.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/generator.py tests/test_generator.py
git commit -m "feat(generator): add PromptGenerator with parameter bank loading"
```

---

## Task 5: ModelRunner

**Files:**
- Create: `reasonbench/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_runner.py`:

```python
import pytest

from reasonbench.models import ModelResponse
from reasonbench.runner import ModelRunner
from tests.conftest import MockClient


class TestModelRunner:
    def test_run_single_model(self):
        client = MockClient(default="Step 1: think.\nANSWER: 42")
        runner = ModelRunner(client, models=["model-a"])
        results = runner.run("What is the answer?")
        assert len(results) == 1
        assert "model-a" in results
        resp = results["model-a"]
        assert isinstance(resp, ModelResponse)
        assert resp.model_name == "model-a"

    def test_run_dual_model(self):
        client = MockClient(default="I think the answer is yes.")
        runner = ModelRunner(client, models=["model-a", "model-b"])
        results = runner.run("prompt")
        assert len(results) == 2
        assert "model-a" in results
        assert "model-b" in results

    def test_reasoning_is_full_response(self):
        response = "Step 1: observe.\nStep 2: deduce.\nANSWER: 42"
        client = MockClient(default=response)
        runner = ModelRunner(client, models=["m"])
        resp = runner.run("prompt")["m"]
        assert resp.reasoning == response

    def test_answer_extracted_from_answer_line(self):
        response = "Step 1: think.\nANSWER: The result is 42"
        client = MockClient(default=response)
        runner = ModelRunner(client, models=["m"])
        resp = runner.run("prompt")["m"]
        assert resp.answer == "The result is 42"

    def test_answer_fallback_last_line(self):
        response = "Step 1: think.\nThe result is probably 42"
        client = MockClient(default=response)
        runner = ModelRunner(client, models=["m"])
        resp = runner.run("prompt")["m"]
        assert resp.answer == "The result is probably 42"

    def test_calls_client_with_correct_model(self):
        client = MockClient(default="response")
        runner = ModelRunner(client, models=["fast-model", "large-model"])
        runner.run("test prompt")
        models_called = [call[1] for call in client.calls]
        assert models_called == ["fast-model", "large-model"]

    def test_is_correct_defaults_to_none(self):
        client = MockClient(default="some answer")
        runner = ModelRunner(client, models=["m"])
        resp = runner.run("prompt")["m"]
        assert resp.is_correct is None

    def test_empty_response(self):
        client = MockClient(default="")
        runner = ModelRunner(client, models=["m"])
        resp = runner.run("prompt")["m"]
        assert resp.answer == ""
        assert resp.reasoning == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/runner.py`:

```python
"""Model runner for multi-model evaluation."""

from __future__ import annotations

from .client import LLMClient
from .models import ModelResponse


class ModelRunner:
    """Runs prompts against multiple models and collects responses."""

    def __init__(self, client: LLMClient, models: list[str]) -> None:
        self._client = client
        self._models = models

    def run(self, prompt_text: str) -> dict[str, ModelResponse]:
        """Run a prompt against all configured models."""
        results: dict[str, ModelResponse] = {}
        for model_name in self._models:
            response_text = self._client.complete(
                prompt_text, model=model_name
            )
            results[model_name] = ModelResponse(
                model_name=model_name,
                answer=self._extract_answer(response_text),
                reasoning=response_text,
            )
        return results

    @staticmethod
    def _extract_answer(response: str) -> str:
        """Extract the final answer from a model response.

        Looks for a line starting with 'ANSWER:'. Falls back to last
        non-empty line.
        """
        if not response.strip():
            return ""
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("ANSWER:"):
                return stripped[len("ANSWER:"):].strip()
        # Fallback: last non-empty line
        for line in reversed(response.split("\n")):
            if line.strip():
                return line.strip()
        return response.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_runner.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/runner.py tests/test_runner.py
git commit -m "feat(runner): add ModelRunner for multi-model prompt evaluation"
```

---

## Task 6: Evaluator

**Files:**
- Create: `reasonbench/evaluator.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_evaluator.py`:

```python
import pytest

from reasonbench.evaluator import Evaluator
from reasonbench.models import Assumption, ValidationResult
from tests.conftest import JUDGE_RESPONSES, MockClient


@pytest.fixture()
def evaluator(judge_client):
    return Evaluator(client=judge_client, judge_model="judge-model")


class TestEvaluator:
    def test_evaluate_returns_validation_result(self, evaluator):
        result = evaluator.evaluate(
            prompt="test prompt",
            answer="wrong answer",
            reasoning="Step 1: assume. Step 2: conclude.",
        )
        assert isinstance(result, ValidationResult)

    def test_reasoning_flawed_detected(self, evaluator):
        result = evaluator.evaluate(
            prompt="p", answer="a", reasoning="r",
        )
        assert result.reasoning_flawed is True

    def test_first_error_step_extracted(self, evaluator):
        result = evaluator.evaluate(
            prompt="p", answer="a", reasoning="r",
        )
        assert result.first_error_step == 2

    def test_assumptions_extracted(self, evaluator):
        result = evaluator.evaluate(
            prompt="p", answer="a", reasoning="r",
        )
        assert len(result.assumptions) == 1
        assert result.assumptions[0].text == "x is positive"
        assert result.assumptions[0].justified is False

    def test_counterfactual_fail_detected(self, evaluator):
        result = evaluator.evaluate(
            prompt="p", answer="a", reasoning="r",
        )
        assert result.counterfactual_fail is True

    def test_adversarial_issues_extracted(self, evaluator):
        result = evaluator.evaluate(
            prompt="p", answer="a", reasoning="r",
        )
        assert len(result.adversarial_issues) >= 1
        assert "unjustified" in result.adversarial_issues[0].lower()

    def test_final_answer_correctness(self, evaluator):
        result = evaluator.evaluate(
            prompt="p", answer="a", reasoning="r",
        )
        assert result.final_answer_correct is False

    def test_calls_judge_model(self, judge_client):
        evaluator = Evaluator(client=judge_client, judge_model="my-judge")
        evaluator.evaluate(prompt="p", answer="a", reasoning="r")
        models_used = {call[1] for call in judge_client.calls}
        assert models_used == {"my-judge"}

    def test_five_validator_calls(self, judge_client):
        evaluator = Evaluator(client=judge_client, judge_model="judge")
        evaluator.evaluate(prompt="p", answer="a", reasoning="r")
        assert len(judge_client.calls) == 5


class TestEvaluatorJsonParsing:
    def test_handles_json_parse_failure_gracefully(self):
        client = MockClient(default="not valid json at all")
        evaluator = Evaluator(client=client, judge_model="m")
        result = evaluator.evaluate(prompt="p", answer="a", reasoning="r")
        # Should return safe defaults, not crash
        assert isinstance(result, ValidationResult)
        assert result.reasoning_flawed is False
        assert result.assumptions == []

    def test_handles_json_embedded_in_text(self):
        client = MockClient(
            default='Here is my analysis:\n{"reasoning_flawed": true, "first_error_step": 3, "explanation": "bad"}\nEnd.'
        )
        evaluator = Evaluator(client=client, judge_model="m")
        result = evaluator.evaluate(prompt="p", answer="a", reasoning="r")
        assert isinstance(result, ValidationResult)
        # Verify embedded JSON was actually extracted
        assert result.reasoning_flawed is True
        assert result.first_error_step == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/evaluator.py`:

```python
"""Validator orchestrator — runs 5 validators via judge LLM."""

from __future__ import annotations

import json
import logging

from .client import LLMClient
from .models import Assumption, ValidationResult
from .validators import ValidatorPack

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs the validator prompt pack against model responses via a judge LLM.

    Each validator prompt is sent to the judge with a JSON schema instruction.
    Responses are parsed into structured data. Parse failures produce safe
    defaults (no crash, no false positives).
    """

    def __init__(self, client: LLMClient, judge_model: str) -> None:
        self._client = client
        self._judge_model = judge_model

    def evaluate(
        self,
        prompt: str,
        answer: str,
        reasoning: str,
        perturbation: str = "Change one key value in the input.",
    ) -> ValidationResult:
        """Run all 5 validators and assemble a ValidationResult."""
        critic = self._run_reasoning_critic(prompt, answer, reasoning)
        assumptions = self._run_assumption_extractor(reasoning)
        cf_fail = self._run_counterfactual_test(reasoning, perturbation)
        adv_issues = self._run_adversarial_challenger(reasoning)
        correct = self._run_truth_judge(prompt, answer)

        return ValidationResult(
            reasoning_flawed=critic["flawed"],
            first_error_step=critic.get("step"),
            assumptions=assumptions,
            counterfactual_fail=cf_fail,
            adversarial_issues=adv_issues,
            final_answer_correct=correct,
        )

    def _judge(self, validator_prompt: str, schema_hint: str) -> dict:
        """Call judge LLM and parse JSON response."""
        full_prompt = (
            f"{validator_prompt}\n\n"
            f"Respond ONLY with valid JSON matching: {schema_hint}"
        )
        response = self._client.complete(
            full_prompt, model=self._judge_model
        )
        return self._parse_json(response)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON from text. Tries full parse, then embedded JSON."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to find JSON object embedded in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {}

    def _run_reasoning_critic(
        self, prompt: str, answer: str, reasoning: str
    ) -> dict:
        schema = (
            '{"reasoning_flawed": bool, "first_error_step": int|null, '
            '"explanation": "string"}'
        )
        validator_prompt = ValidatorPack.reasoning_critic(
            prompt, answer, reasoning
        )
        try:
            result = self._judge(validator_prompt, schema)
            return {
                "flawed": bool(result.get("reasoning_flawed", False)),
                "step": result.get("first_error_step"),
            }
        except Exception:
            logger.debug("Reasoning critic parse failed", exc_info=True)
            return {"flawed": False, "step": None}

    def _run_assumption_extractor(self, reasoning: str) -> list[Assumption]:
        schema = '{"assumptions": [{"text": "string", "justified": bool}]}'
        validator_prompt = ValidatorPack.assumption_extractor(reasoning)
        try:
            result = self._judge(validator_prompt, schema)
            return [
                Assumption(text=a["text"], justified=a["justified"])
                for a in result.get("assumptions", [])
            ]
        except Exception:
            logger.debug("Assumption extractor parse failed", exc_info=True)
            return []

    def _run_counterfactual_test(
        self, reasoning: str, perturbation: str
    ) -> bool:
        schema = '{"holds": bool}'
        validator_prompt = ValidatorPack.counterfactual_test(
            reasoning, perturbation
        )
        try:
            result = self._judge(validator_prompt, schema)
            return not result.get("holds", True)
        except Exception:
            logger.debug("Counterfactual test parse failed", exc_info=True)
            return False

    def _run_adversarial_challenger(self, reasoning: str) -> list[str]:
        schema = '{"issues": ["string"], "robust": bool}'
        validator_prompt = ValidatorPack.adversarial_challenger(reasoning)
        try:
            result = self._judge(validator_prompt, schema)
            return result.get("issues", [])
        except Exception:
            logger.debug(
                "Adversarial challenger parse failed", exc_info=True
            )
            return []

    def _run_truth_judge(self, prompt: str, answer: str) -> bool | None:
        schema = '{"correct": bool}'
        validator_prompt = ValidatorPack.truth_judge(prompt, answer)
        try:
            result = self._judge(validator_prompt, schema)
            return result.get("correct")
        except Exception:
            logger.debug("Truth judge parse failed", exc_info=True)
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/evaluator.py tests/test_evaluator.py
git commit -m "feat(evaluator): add judge-LLM validator orchestrator with JSON parsing"
```

---

## Task 7: Pipeline

**Files:**
- Create: `reasonbench/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline.py`:

```python
import json
from pathlib import Path

import pytest

from reasonbench.models import EvaluationResult
from reasonbench.pipeline import Pipeline
from reasonbench.taxonomy import Severity
from tests.conftest import JUDGE_RESPONSES, MODEL_RESPONSE_TEXT, MockClient


@pytest.fixture()
def params_dir(tmp_path: Path) -> Path:
    """Minimal param bank for pipeline testing."""
    data = [
        {"rule_a": "x > 0", "rule_b": "double it", "edge_case_input": "x = -1"},
        {"rule_a": "len < 5", "rule_b": "reverse", "edge_case_input": "empty"},
    ]
    (tmp_path / "implicit_assumption_trap.json").write_text(json.dumps(data))
    return tmp_path


@pytest.fixture()
def pipeline_client():
    """Client that handles both model and judge calls."""
    responses = dict(JUDGE_RESPONSES)
    return MockClient(responses=responses, default=MODEL_RESPONSE_TEXT)


@pytest.fixture()
def output_path(tmp_path: Path) -> Path:
    return tmp_path / "results.jsonl"


class TestPipeline:
    def test_run_returns_results(self, pipeline_client, params_dir, output_path):
        pipeline = Pipeline(
            client=pipeline_client,
            models=["model-a"],
            judge_model="judge",
            output_path=output_path,
            params_dir=params_dir,
            seed=42,
        )
        results = pipeline.run(count=2)
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_results_stored_to_jsonl(self, pipeline_client, params_dir, output_path):
        pipeline = Pipeline(
            client=pipeline_client,
            models=["model-a"],
            judge_model="judge",
            output_path=output_path,
            params_dir=params_dir,
            seed=42,
        )
        pipeline.run(count=2)
        assert output_path.exists()
        lines = [l for l in output_path.read_text().strip().split("\n") if l]
        assert len(lines) == 2

    def test_dual_model_disagreement(self, params_dir, output_path):
        class AlternatingClient:
            def __init__(self):
                self.calls = []
                self._n = 0

            def complete(self, prompt, *, model):
                self.calls.append((prompt, model))
                self._n += 1
                for kw, resp in JUDGE_RESPONSES.items():
                    if kw in prompt:
                        return resp
                if self._n % 2 == 0:
                    return "Step 1: no.\nANSWER: yes"
                return "Step 1: yes.\nANSWER: no"

        client = AlternatingClient()
        pipeline = Pipeline(
            client=client,
            models=["model-a", "model-b"],
            judge_model="judge",
            output_path=output_path,
            params_dir=params_dir,
            seed=42,
        )
        results = pipeline.run(count=1)
        assert len(results) == 1
        assert "model-a" in results[0].models
        assert "model-b" in results[0].models

    def test_score_and_severity_populated(self, pipeline_client, params_dir, output_path):
        pipeline = Pipeline(
            client=pipeline_client,
            models=["model-a"],
            judge_model="judge",
            output_path=output_path,
            params_dir=params_dir,
            seed=42,
        )
        results = pipeline.run(count=1)
        r = results[0]
        assert isinstance(r.score, int)
        assert isinstance(r.severity, Severity)
        assert r.score >= 0

    def test_empty_params_dir_returns_empty(self, pipeline_client, tmp_path):
        empty_params = tmp_path / "empty_params"
        empty_params.mkdir()
        pipeline = Pipeline(
            client=pipeline_client,
            models=["m"],
            judge_model="j",
            output_path=tmp_path / "out.jsonl",
            params_dir=empty_params,
            seed=42,
        )
        results = pipeline.run(count=10)
        assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/pipeline.py`:

```python
"""Main evaluation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from .client import LLMClient
from .evaluator import Evaluator
from .generator import PromptGenerator
from .models import EvaluationResult
from .runner import ModelRunner
from .scoring import Scorer
from .storage import JsonlStore

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates prompt generation, model evaluation, validation, and scoring."""

    def __init__(
        self,
        client: LLMClient,
        models: list[str],
        judge_model: str,
        output_path: Path,
        params_dir: Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._generator = PromptGenerator(
            params_dir=params_dir, seed=seed
        )
        self._runner = ModelRunner(client, models)
        self._evaluator = Evaluator(client, judge_model)
        self._store = JsonlStore(output_path)

    def run(self, count: int = 100) -> list[EvaluationResult]:
        """Run the full pipeline: generate -> evaluate -> validate -> score -> store."""
        prompts = self._generator.generate_batch(count)
        if not prompts:
            return []

        results: list[EvaluationResult] = []
        for i, prompt in enumerate(prompts, 1):
            logger.info(
                "Evaluating prompt %d/%d [%s]",
                i,
                len(prompts),
                prompt.template_id,
            )

            # Run all models
            model_responses = self._runner.run(prompt.prompt_text)

            # Validate the first model's response
            first_response = next(iter(model_responses.values()))
            validation = self._evaluator.evaluate(
                prompt=prompt.prompt_text,
                answer=first_response.answer,
                reasoning=first_response.reasoning,
            )

            # Compute score
            unjustified = sum(
                1 for a in validation.assumptions if not a.justified
            )
            answers = {r.answer for r in model_responses.values()}
            disagreement = len(answers) > 1

            score = Scorer.compute_score(
                is_correct=validation.final_answer_correct or False,
                reasoning_flawed=validation.reasoning_flawed,
                assumption_errors=unjustified,
                counterfactual_fail=validation.counterfactual_fail,
                model_disagreement=disagreement,
            )
            severity = Scorer.severity(score)

            # Assemble and store result
            result = EvaluationResult(
                prompt_id=prompt.prompt_id,
                failure_type=prompt.failure_type,
                prompt_text=prompt.prompt_text,
                models=model_responses,
                validation=validation,
                score=score,
                severity=severity,
            )
            self._store.append(result)
            results.append(result)

            if score >= 6:
                logger.info(
                    "  -> CRITICAL failure (score=%d): %s",
                    score,
                    prompt.template_id,
                )

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): add main evaluation loop with scoring and storage"
```

---

## Task 8: CLI Entry Point and Public API Update

**Files:**
- Create: `reasonbench/__main__.py`
- Modify: `reasonbench/__init__.py`

- [ ] **Step 1: Create CLI entry point**

Create `reasonbench/__main__.py`:

```python
"""CLI entry point: python -m reasonbench."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run the ReasonBench evaluation pipeline."""
    default_model = os.environ.get("REASONBENCH_MODEL", "")
    default_judge = os.environ.get("REASONBENCH_JUDGE_MODEL", "")

    parser = argparse.ArgumentParser(
        prog="reasonbench",
        description="LLM Adversarial Reasoning Evaluation System",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=not bool(default_model),
        default=[default_model] if default_model else None,
        help="Models to evaluate (or set REASONBENCH_MODEL env var)",
    )
    parser.add_argument(
        "--judge",
        required=not bool(default_judge),
        default=default_judge or None,
        help="Judge model for validation (or set REASONBENCH_JUDGE_MODEL env var)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of prompts to generate (default: 10)",
    )
    parser.add_argument(
        "--output",
        default="results.jsonl",
        help="Output JSONL file (default: results.jsonl)",
    )
    parser.add_argument(
        "--params-dir",
        default=None,
        help="Custom parameter banks directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    from .client import AnthropicClient
    from .pipeline import Pipeline

    client = AnthropicClient()
    pipeline = Pipeline(
        client=client,
        models=args.models,
        judge_model=args.judge,
        output_path=Path(args.output),
        params_dir=Path(args.params_dir) if args.params_dir else None,
        seed=args.seed,
    )

    results = pipeline.run(count=args.count)

    # Print summary
    critical = sum(1 for r in results if r.severity.value == "critical")
    high = sum(1 for r in results if r.severity.value == "high")
    medium = sum(1 for r in results if r.severity.value == "medium")
    low = sum(1 for r in results if r.severity.value == "low")

    print(f"\nReasonBench Evaluation Complete")
    print(f"  Prompts evaluated: {len(results)}")
    print(f"  Critical: {critical}")
    print(f"  High:     {high}")
    print(f"  Medium:   {medium}")
    print(f"  Low:      {low}")
    print(f"  Output:   {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Test CLI argument parsing**

```bash
uv run python -m reasonbench --help
```
Expected: Help text printed with all arguments

- [ ] **Step 3: Write CLI tests**

Create `tests/test_cli.py`:

```python
"""Tests for CLI argument parsing and main function."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reasonbench.__main__ import main
from tests.conftest import JUDGE_RESPONSES, MODEL_RESPONSE_TEXT, MockClient


@pytest.fixture()
def params_dir(tmp_path: Path) -> Path:
    data = [
        {"rule_a": "x > 0", "rule_b": "double", "edge_case_input": "x = -1"},
    ]
    (tmp_path / "implicit_assumption_trap.json").write_text(json.dumps(data))
    return tmp_path


class TestCLI:
    def test_main_returns_zero(self, params_dir, tmp_path):
        output = tmp_path / "out.jsonl"
        mock = MockClient(
            responses=JUDGE_RESPONSES, default=MODEL_RESPONSE_TEXT
        )
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            code = main([
                "--models", "m",
                "--judge", "j",
                "--count", "1",
                "--output", str(output),
                "--params-dir", str(params_dir),
                "--seed", "42",
            ])
        assert code == 0
        assert output.exists()

    def test_main_writes_results(self, params_dir, tmp_path):
        output = tmp_path / "out.jsonl"
        mock = MockClient(
            responses=JUDGE_RESPONSES, default=MODEL_RESPONSE_TEXT
        )
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            main([
                "--models", "m",
                "--judge", "j",
                "--count", "2",
                "--output", str(output),
                "--params-dir", str(params_dir),
            ])
        lines = [l for l in output.read_text().strip().split("\n") if l]
        assert len(lines) == 2

    def test_missing_required_args_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0
```

- [ ] **Step 4: Update __init__.py with new exports**

Read the current `reasonbench/__init__.py`, then replace with:

```python
"""ReasonBench — LLM Adversarial Reasoning Evaluation System."""

from .client import AnthropicClient, LLMClient
from .evaluator import Evaluator
from .generator import PromptGenerator
from .models import (
    Assumption,
    EvaluationResult,
    FailureRecord,
    ModelResponse,
    Prompt,
    ValidationResult,
)
from .pipeline import Pipeline
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
    "AnthropicClient",
    "Assumption",
    "DISTRIBUTION",
    "EvaluationResult",
    "Evaluator",
    "FAILURE_CATEGORY_MAP",
    "FailureCategory",
    "FailureRecord",
    "FailureType",
    "JsonlStore",
    "LLMClient",
    "ModelResponse",
    "ModelRunner",
    "Pipeline",
    "Prompt",
    "PromptGenerator",
    "Scorer",
    "Severity",
    "TemplateRegistry",
    "ValidationResult",
    "ValidatorPack",
    "get_category",
]
```

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v
```
Expected: All tests PASS (Phase 1: 84 + Phase 2: ~87 = ~171 tests)

- [ ] **Step 6: Commit**

```bash
git add reasonbench/__main__.py reasonbench/__init__.py tests/test_cli.py
git commit -m "feat: add CLI entry point and update public API with Phase 2 exports"
```

---

## Summary

| Task | Component | New Files | Tests |
|------|-----------|-----------|-------|
| 1 | Dependencies + MockClient | 1 (conftest) | 0 |
| 2 | LLM Client | 2 (client, test) | 7 |
| 3 | Parameter Banks + Validation | 11 (10 JSON + test_data) | 31 |
| 4 | PromptGenerator | 2 (generator, test) | 11 |
| 5 | ModelRunner | 2 (runner, test) | 8 |
| 6 | Evaluator | 2 (evaluator, test) | 11 |
| 7 | Pipeline | 2 (pipeline, test) | 5 |
| 8 | CLI + Init | 3 (__main__, __init__, test_cli) | 3 |
| **Total** | | **25 files** | **~76 new tests** |

**Phase 2 delivers:** A working end-to-end pipeline that generates adversarial prompts from 50 parameter configurations across 10 template types, evaluates them against one or more LLMs, validates reasoning through 5 judge-LLM validators, computes composite failure scores, and stores results as JSONL. All model names configured via CLI args or env vars (REASONBENCH_MODEL, REASONBENCH_JUDGE_MODEL). Runnable via `python -m reasonbench --models <model> --judge <judge-model> --count 10`.
