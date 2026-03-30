---
type: canonical
source: none
sync: none
sla: none
---

# Fallax Phase 5: Experiment Loop & Reporting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the full feedback loop together — run → analyze → evolve → re-run → report — as a multi-round `Experiment` orchestrator with structured reporting.

**Architecture:** `Experiment` composes existing modules (PromptGenerator, ModelRunner, Evaluator, Scorer, Analyzer, PromptEvolver, SelfRepairTester, RootCauseExtractor) into a multi-round loop. Each round evaluates prompts, analyzes results, and evolves hard cases into next-round prompts. `ReportBuilder` summarizes experiment data into structured dicts and markdown. One new data model (`ExperimentRound`) captures per-round metrics.

**Tech Stack:** Python 3.12+, Pydantic v2, pytest, uv (no new dependencies)

**Depends on:** Phase 4 (evolver, repair, root_cause) — 233 passing tests.

---

## File Structure

```
reasonbench/
├── reasonbench/
│   ├── models.py             # MODIFY: add ExperimentRound
│   ├── experiment.py         # NEW: multi-round experiment orchestrator
│   ├── report.py             # NEW: structured report builder
│   ├── __main__.py           # MODIFY: add experiment subcommand
│   └── __init__.py           # MODIFY: add new exports
├── tests/
│   ├── test_models.py        # MODIFY: add ExperimentRound tests
│   ├── test_experiment.py    # NEW
│   └── test_report.py        # NEW
```

**Dependency order:** `models` → (`experiment`, `report`) → `CLI` → `__init__`

---

## Task 1: ExperimentRound Data Model

**Files:**
- Modify: `reasonbench/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for ExperimentRound**

Append to `tests/test_models.py`:

```python
from reasonbench.models import ExperimentRound


class TestExperimentRound:
    def test_create(self):
        r = ExperimentRound(
            round_number=1,
            prompts_evaluated=10,
            avg_score=6.5,
            failure_rate=0.8,
            hard_case_count=8,
            evolved_count=5,
        )
        assert r.round_number == 1
        assert r.prompts_evaluated == 10
        assert r.avg_score == 6.5
        assert r.failure_rate == 0.8
        assert r.hard_case_count == 8
        assert r.evolved_count == 5
        assert r.repair_success_rate is None

    def test_with_repair_rate(self):
        r = ExperimentRound(
            round_number=2,
            prompts_evaluated=5,
            avg_score=7.0,
            failure_rate=1.0,
            hard_case_count=5,
            evolved_count=0,
            repair_success_rate=0.4,
        )
        assert r.repair_success_rate == 0.4

    def test_json_roundtrip(self):
        r = ExperimentRound(
            round_number=1,
            prompts_evaluated=3,
            avg_score=5.0,
            failure_rate=0.5,
            hard_case_count=1,
            evolved_count=1,
        )
        restored = ExperimentRound.model_validate_json(r.model_dump_json())
        assert restored.round_number == 1
        assert restored.avg_score == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v -k "ExperimentRound"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add ExperimentRound to models.py**

Append to the end of `reasonbench/models.py`:

```python


class ExperimentRound(BaseModel):
    """Metadata for one round of an experiment."""

    round_number: int = Field(ge=1)
    prompts_evaluated: int = Field(ge=0)
    avg_score: float
    failure_rate: float
    hard_case_count: int = Field(ge=0)
    evolved_count: int = Field(ge=0)
    repair_success_rate: float | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: All tests PASS (existing 18 + new 3 = 21)

- [ ] **Step 5: Commit**

```bash
git add reasonbench/models.py tests/test_models.py
git commit -m "feat(models): add ExperimentRound data model"
```

---

## Task 2: Experiment Runner

**Files:**
- Create: `reasonbench/experiment.py`
- Create: `tests/test_experiment.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_experiment.py`:

```python
import json

import pytest

from reasonbench.experiment import Experiment
from reasonbench.models import ExperimentRound, RootCausePattern, RepairResult
from tests.conftest import JUDGE_RESPONSES, MODEL_RESPONSE_TEXT, MockClient


@pytest.fixture()
def experiment_client():
    """MockClient handling model, judge, evolver, and repair responses."""
    responses = {
        **JUDGE_RESPONSES,
        "Given this prompt and its failure": "An evolved harder prompt text",
        "Your previous answer was incorrect": "I was wrong. The fixed answer is Z.",
    }
    return MockClient(responses=responses, default=MODEL_RESPONSE_TEXT)


@pytest.fixture()
def params_dir(tmp_path):
    """Parameter bank for round 1 prompt generation."""
    data = [
        {"rule_a": "x > 0", "rule_b": "double", "edge_case_input": "x = -1"},
        {"rule_a": "y < 5", "rule_b": "triple", "edge_case_input": "y = 10"},
    ]
    (tmp_path / "implicit_assumption_trap.json").write_text(json.dumps(data))
    return tmp_path


class TestExperiment:
    def test_run_returns_dict_with_required_keys(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        assert "rounds" in result
        assert "root_cause_patterns" in result
        assert "repair_results" in result
        assert "total_prompts" in result
        assert "total_failures" in result

    def test_run_creates_round_files(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        exp.run(initial_count=2, rounds=2, min_score=6)

        assert (output_dir / "round_1.jsonl").exists()
        assert (output_dir / "round_2.jsonl").exists()

    def test_run_returns_correct_round_count(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=3, min_score=6)

        assert len(result["rounds"]) == 3

    def test_rounds_are_experiment_round_objects(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        assert all(isinstance(r, ExperimentRound) for r in result["rounds"])
        assert result["rounds"][0].round_number == 1
        assert result["rounds"][1].round_number == 2

    def test_round_metadata_populated(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        r1 = result["rounds"][0]
        assert r1.prompts_evaluated > 0
        assert r1.avg_score > 0
        assert r1.hard_case_count > 0

    def test_total_prompts_sums_rounds(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        expected = sum(r.prompts_evaluated for r in result["rounds"])
        assert result["total_prompts"] == expected

    def test_root_cause_patterns_returned(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        assert isinstance(result["root_cause_patterns"], list)

    def test_repair_results_returned(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        assert isinstance(result["repair_results"], list)
        assert all(isinstance(r, RepairResult) for r in result["repair_results"])

    def test_early_termination_when_no_hard_cases(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        # min_score=100 means nothing qualifies as hard → no evolution → round 2 has no prompts
        result = exp.run(initial_count=2, rounds=3, min_score=100)

        assert len(result["rounds"]) == 1

    def test_last_round_does_not_evolve(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        last_round = result["rounds"][-1]
        assert last_round.evolved_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_experiment.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/experiment.py`:

```python
"""Multi-round experiment orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

from .analyzer import Analyzer
from .client import LLMClient
from .evaluator import Evaluator
from .evolver import PromptEvolver
from .generator import PromptGenerator
from .models import EvaluationResult, ExperimentRound, Prompt
from .repair import SelfRepairTester
from .root_cause import RootCauseExtractor
from .runner import ModelRunner
from .scoring import Scorer
from .storage import JsonlStore

logger = logging.getLogger(__name__)


class Experiment:
    """Orchestrates multi-round evaluation with evolution and reporting.

    Round 1 generates prompts via PromptGenerator. Each subsequent round
    evaluates evolved prompts from the prior round. After all rounds,
    root cause extraction runs on combined results and self-repair testing
    runs on the final round's failures.
    """

    def __init__(
        self,
        client: LLMClient,
        models: list[str],
        judge_model: str,
        output_dir: Path,
        evolve_model: str,
        params_dir: Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._client = client
        self._models = models
        self._judge_model = judge_model
        self._output_dir = output_dir
        self._evolve_model = evolve_model
        self._params_dir = params_dir
        self._seed = seed

    def run(
        self,
        initial_count: int = 10,
        rounds: int = 3,
        min_score: int = 6,
    ) -> dict:
        """Run a multi-round experiment.

        Returns a dict with keys: rounds, root_cause_patterns,
        repair_results, total_prompts, total_failures.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        all_results: list[EvaluationResult] = []
        round_summaries: list[ExperimentRound] = []
        last_round_results: list[EvaluationResult] = []
        prompts: list[Prompt] | None = None

        for round_num in range(1, rounds + 1):
            round_path = self._output_dir / f"round_{round_num}.jsonl"

            if round_num == 1:
                generator = PromptGenerator(
                    params_dir=self._params_dir, seed=self._seed
                )
                prompts = generator.generate_batch(initial_count)

            if not prompts:
                break

            logger.info(
                "Round %d: evaluating %d prompts", round_num, len(prompts)
            )
            results = self._evaluate_prompts(prompts, round_path)
            all_results.extend(results)
            last_round_results = results

            analyzer = Analyzer(results)
            summary = analyzer.summary()
            hard_cases = analyzer.hard_cases(min_score)

            evolved: list[Prompt] = []
            if round_num < rounds and hard_cases:
                evolver = PromptEvolver(self._client, self._evolve_model)
                evolved = evolver.evolve_batch(results, min_score)

            round_summaries.append(
                ExperimentRound(
                    round_number=round_num,
                    prompts_evaluated=len(results),
                    avg_score=summary["avg_score"],
                    failure_rate=summary["failure_rate"],
                    hard_case_count=len(hard_cases),
                    evolved_count=len(evolved),
                )
            )

            prompts = evolved if evolved else None

        root_cause_patterns = []
        repair_results = []
        if all_results:
            extractor = RootCauseExtractor(all_results)
            root_cause_patterns = extractor.extract_patterns(
                min_frequency=1
            )
            tester = SelfRepairTester(self._client)
            repair_results = tester.test_repair_batch(last_round_results)

        return {
            "rounds": round_summaries,
            "root_cause_patterns": root_cause_patterns,
            "repair_results": repair_results,
            "total_prompts": sum(
                r.prompts_evaluated for r in round_summaries
            ),
            "total_failures": sum(
                1
                for r in all_results
                if r.validation.reasoning_flawed
            ),
        }

    def _evaluate_prompts(
        self, prompts: list[Prompt], output_path: Path
    ) -> list[EvaluationResult]:
        """Evaluate a list of prompts through the full pipeline."""
        runner = ModelRunner(self._client, self._models)
        evaluator = Evaluator(self._client, self._judge_model)
        store = JsonlStore(output_path)

        results: list[EvaluationResult] = []
        for prompt in prompts:
            model_responses = runner.run(prompt.prompt_text)
            first_response = next(iter(model_responses.values()))
            validation = evaluator.evaluate(
                prompt=prompt.prompt_text,
                answer=first_response.answer,
                reasoning=first_response.reasoning,
            )

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

            result = EvaluationResult(
                prompt_id=prompt.prompt_id,
                failure_type=prompt.failure_type,
                prompt_text=prompt.prompt_text,
                models=model_responses,
                validation=validation,
                score=score,
                severity=severity,
            )
            store.append(result)
            results.append(result)

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_experiment.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/experiment.py tests/test_experiment.py
git commit -m "feat(experiment): add multi-round evaluation loop with evolution"
```

---

## Task 3: Report Builder

**Files:**
- Create: `reasonbench/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_report.py`:

```python
import pytest

from reasonbench.models import ExperimentRound, RootCausePattern, RepairResult
from reasonbench.report import ReportBuilder


@pytest.fixture()
def experiment_data():
    """Hand-crafted experiment data for report tests."""
    return {
        "rounds": [
            ExperimentRound(
                round_number=1,
                prompts_evaluated=10,
                avg_score=6.0,
                failure_rate=0.8,
                hard_case_count=8,
                evolved_count=5,
            ),
            ExperimentRound(
                round_number=2,
                prompts_evaluated=5,
                avg_score=7.5,
                failure_rate=1.0,
                hard_case_count=5,
                evolved_count=0,
            ),
        ],
        "root_cause_patterns": [
            RootCausePattern(
                pattern="assumes positivity",
                frequency=4,
                models_affected=["model-a", "model-b"],
                example_prompt="Is x positive?",
                failure_types=["contradiction"],
            ),
        ],
        "repair_results": [
            RepairResult(
                model_name="model-a",
                prompt_text="p",
                original_answer="a",
                repaired_answer="b",
                repair_reasoning="r",
                is_fixed=True,
            ),
            RepairResult(
                model_name="model-a",
                prompt_text="p2",
                original_answer="c",
                repaired_answer="d",
                repair_reasoning="r2",
                is_fixed=False,
            ),
        ],
        "total_prompts": 15,
        "total_failures": 12,
    }


class TestReportBuilder:
    def test_build_returns_dict(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        assert isinstance(report, dict)

    def test_build_has_required_keys(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        expected_keys = {
            "total_rounds", "total_prompts", "total_failures",
            "score_trend", "failure_trend", "score_delta",
            "failure_delta", "hardening_rate",
            "repair_success_rate", "top_patterns",
        }
        assert expected_keys <= set(report.keys())

    def test_score_trend(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        assert report["score_trend"] == [6.0, 7.5]

    def test_failure_trend(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        assert report["failure_trend"] == [0.8, 1.0]

    def test_score_delta(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        assert report["score_delta"] == pytest.approx(1.5)

    def test_repair_success_rate(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        # 1 fixed out of 2
        assert report["repair_success_rate"] == pytest.approx(0.5)

    def test_top_patterns_limited_to_five(self):
        patterns = [
            RootCausePattern(
                pattern=f"pattern-{i}",
                frequency=10 - i,
                models_affected=["m"],
                example_prompt="p",
                failure_types=["contradiction"],
            )
            for i in range(8)
        ]
        data = {
            "rounds": [
                ExperimentRound(
                    round_number=1,
                    prompts_evaluated=10,
                    avg_score=5.0,
                    failure_rate=0.5,
                    hard_case_count=5,
                    evolved_count=0,
                ),
            ],
            "root_cause_patterns": patterns,
            "repair_results": [],
            "total_prompts": 10,
            "total_failures": 5,
        }
        builder = ReportBuilder(data)
        report = builder.build()
        assert len(report["top_patterns"]) == 5

    def test_to_markdown_contains_sections(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        md = builder.to_markdown()
        assert "# Fallax Experiment Report" in md
        assert "## Overview" in md
        assert "## Per-Round Results" in md
        assert "## Trends" in md
        assert "## Top Root Cause Patterns" in md
        assert "assumes positivity" in md

    def test_empty_rounds(self):
        data = {
            "rounds": [],
            "root_cause_patterns": [],
            "repair_results": [],
            "total_prompts": 0,
            "total_failures": 0,
        }
        builder = ReportBuilder(data)
        report = builder.build()
        assert report["total_rounds"] == 0
        assert report["score_trend"] == []
        assert report["score_delta"] == 0.0

    def test_no_repairs_gives_none_rate(self):
        data = {
            "rounds": [
                ExperimentRound(
                    round_number=1,
                    prompts_evaluated=5,
                    avg_score=3.0,
                    failure_rate=0.4,
                    hard_case_count=2,
                    evolved_count=0,
                ),
            ],
            "root_cause_patterns": [],
            "repair_results": [],
            "total_prompts": 5,
            "total_failures": 2,
        }
        builder = ReportBuilder(data)
        report = builder.build()
        assert report["repair_success_rate"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/report.py`:

```python
"""Report builder for experiment results."""

from __future__ import annotations

from .models import ExperimentRound, RootCausePattern


class ReportBuilder:
    """Builds structured reports from experiment data.

    Takes the dict returned by Experiment.run() and produces
    a summary dict (build()) or markdown string (to_markdown()).
    """

    def __init__(self, experiment_data: dict) -> None:
        self._data = experiment_data

    def build(self) -> dict:
        """Build a structured report summary."""
        rounds: list[ExperimentRound] = self._data["rounds"]

        if not rounds:
            return {
                "total_rounds": 0,
                "total_prompts": 0,
                "total_failures": 0,
                "score_trend": [],
                "failure_trend": [],
                "score_delta": 0.0,
                "failure_delta": 0.0,
                "hardening_rate": 0.0,
                "repair_success_rate": None,
                "top_patterns": [],
            }

        score_trend = [r.avg_score for r in rounds]
        failure_trend = [r.failure_rate for r in rounds]

        hardening_rate = 0.0
        if len(rounds) >= 2:
            total_evolved = sum(r.evolved_count for r in rounds[:-1])
            total_stayed_hard = sum(r.hard_case_count for r in rounds[1:])
            if total_evolved > 0:
                hardening_rate = total_stayed_hard / total_evolved

        repairs = self._data.get("repair_results", [])
        repair_rate = None
        if repairs:
            fixed = sum(1 for r in repairs if r.is_fixed is True)
            repair_rate = fixed / len(repairs)

        patterns: list[RootCausePattern] = self._data.get(
            "root_cause_patterns", []
        )
        top_patterns = [
            {
                "pattern": p.pattern,
                "frequency": p.frequency,
                "models_affected": p.models_affected,
                "failure_types": p.failure_types,
            }
            for p in patterns[:5]
        ]

        return {
            "total_rounds": len(rounds),
            "total_prompts": self._data["total_prompts"],
            "total_failures": self._data["total_failures"],
            "score_trend": score_trend,
            "failure_trend": failure_trend,
            "score_delta": score_trend[-1] - score_trend[0],
            "failure_delta": failure_trend[-1] - failure_trend[0],
            "hardening_rate": hardening_rate,
            "repair_success_rate": repair_rate,
            "top_patterns": top_patterns,
        }

    def to_markdown(self) -> str:
        """Render the report as a markdown string."""
        report = self.build()
        rounds: list[ExperimentRound] = self._data["rounds"]

        lines = [
            "# Fallax Experiment Report",
            "",
            "## Overview",
            "",
            f"- **Rounds:** {report['total_rounds']}",
            f"- **Total prompts:** {report['total_prompts']}",
            f"- **Total failures:** {report['total_failures']}",
            "",
            "## Per-Round Results",
            "",
            "| Round | Prompts | Avg Score | Failure Rate | Evolved |",
            "|-------|---------|-----------|--------------|---------|",
        ]

        for r in rounds:
            lines.append(
                f"| {r.round_number} | {r.prompts_evaluated} | "
                f"{r.avg_score:.2f} | {r.failure_rate:.1%} | "
                f"{r.evolved_count} |"
            )

        total_rounds = report["total_rounds"]
        lines.extend([
            "",
            "## Trends",
            "",
            f"- **Score delta:** {report['score_delta']:+.2f} "
            f"(round 1 to round {total_rounds})",
            f"- **Failure rate delta:** {report['failure_delta']:+.1%}",
            f"- **Hardening rate:** {report['hardening_rate']:.1%}",
        ])

        if report["repair_success_rate"] is not None:
            lines.append(
                f"- **Repair success rate:** "
                f"{report['repair_success_rate']:.1%}"
            )

        if report["top_patterns"]:
            lines.extend([
                "",
                "## Top Root Cause Patterns",
                "",
            ])
            for p in report["top_patterns"]:
                models = ", ".join(p["models_affected"])
                lines.append(
                    f"- **{p['pattern']}** "
                    f"(frequency: {p['frequency']}, models: {models})"
                )

        lines.append("")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/report.py tests/test_report.py
git commit -m "feat(report): add structured report builder with markdown output"
```

---

## Task 4: CLI Experiment Subcommand

**Files:**
- Modify: `reasonbench/__main__.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_cli.py`:

```python
class TestExperimentSubcommand:
    def test_experiment_returns_zero(self, params_dir, tmp_path):
        output_dir = tmp_path / "experiment"
        mock = MockClient(
            responses={
                **JUDGE_RESPONSES,
                "Given this prompt and its failure": "Evolved prompt",
                "Your previous answer was incorrect": "Fixed answer.",
            },
            default=MODEL_RESPONSE_TEXT,
        )
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            code = main([
                "experiment",
                "--models", "m",
                "--judge", "j",
                "--evolve-model", "e",
                "--rounds", "2",
                "--count", "1",
                "--output-dir", str(output_dir),
                "--params-dir", str(params_dir),
            ])
        assert code == 0
        assert (output_dir / "round_1.jsonl").exists()
        assert (output_dir / "report.json").exists()
        assert (output_dir / "report.md").exists()

    def test_experiment_with_seed(self, params_dir, tmp_path):
        output_dir = tmp_path / "experiment"
        mock = MockClient(
            responses={
                **JUDGE_RESPONSES,
                "Given this prompt and its failure": "Evolved",
                "Your previous answer was incorrect": "Fixed.",
            },
            default=MODEL_RESPONSE_TEXT,
        )
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            code = main([
                "experiment",
                "--models", "m",
                "--judge", "j",
                "--evolve-model", "e",
                "--rounds", "1",
                "--count", "1",
                "--output-dir", str(output_dir),
                "--params-dir", str(params_dir),
                "--seed", "42",
            ])
        assert code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v -k "Experiment"`
Expected: FAIL with `SystemExit: 2` (unrecognized subcommand)

- [ ] **Step 3: Add experiment subcommand to __main__.py**

Add this function definition before `main()` in `reasonbench/__main__.py`:

```python
def _cmd_experiment(args: argparse.Namespace) -> int:
    """Run a multi-round experiment."""
    import json as json_mod

    from .experiment import Experiment
    from .report import ReportBuilder

    output_dir = Path(args.output_dir)
    client = AnthropicClient()
    exp = Experiment(
        client=client,
        models=args.models,
        judge_model=args.judge,
        output_dir=output_dir,
        evolve_model=args.evolve_model,
        params_dir=Path(args.params_dir) if args.params_dir else None,
        seed=args.seed,
    )
    data = exp.run(
        initial_count=args.count,
        rounds=args.rounds,
        min_score=args.min_score,
    )

    builder = ReportBuilder(data)
    report = builder.build()

    report_json = output_dir / "report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json_mod.dump(report, f, indent=2, default=str)

    report_md = output_dir / "report.md"
    report_md.write_text(builder.to_markdown(), encoding="utf-8")

    print(f"\nExperiment complete ({len(data['rounds'])} rounds)")
    print(f"  Total prompts:  {data['total_prompts']}")
    print(f"  Total failures: {data['total_failures']}")
    print(f"  Output dir:     {output_dir}")
    return 0
```

Add this subparser block in `main()`, after the existing `repair` subparser:

```python
    # -- experiment --
    exp_p = subparsers.add_parser(
        "experiment", help="Run multi-round evaluation experiment"
    )
    exp_p.add_argument(
        "--models", nargs="+", required=True,
        help="Models to evaluate",
    )
    exp_p.add_argument(
        "--judge", required=True, help="Judge model",
    )
    exp_p.add_argument(
        "--evolve-model", required=True,
        help="Model for prompt evolution",
    )
    exp_p.add_argument("--rounds", type=int, default=3)
    exp_p.add_argument("--count", type=int, default=10)
    exp_p.add_argument("--min-score", type=int, default=6)
    exp_p.add_argument("--output-dir", default="experiment_output")
    exp_p.add_argument("--params-dir", default=None)
    exp_p.add_argument("--seed", type=int, default=None)
```

Add this dispatch in `main()`, after the existing `repair` dispatch:

```python
    if args.command == "experiment":
        return _cmd_experiment(args)
```

- [ ] **Step 4: Run CLI tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All 13 tests PASS (11 existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git add reasonbench/__main__.py tests/test_cli.py
git commit -m "feat(cli): add experiment subcommand for multi-round evaluation"
```

---

## Task 5: Public API Update and Full Suite

**Files:**
- Modify: `reasonbench/__init__.py`

- [ ] **Step 1: Update __init__.py**

Add new imports to `reasonbench/__init__.py`:

After the `from .evolver import PromptEvolver` line, add:
```python
from .experiment import Experiment
```

In the models import block, add `ExperimentRound` after `EvaluationResult`:
```python
    ExperimentRound,
```

After the `from .repair import SelfRepairTester` line, add:
```python
from .report import ReportBuilder
```

In the `__all__` list, add these entries in alphabetical position:
```python
    "Experiment",
    "ExperimentRound",
```

And add after `"PromptGenerator"`:
```python
    "ReportBuilder",
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS (~257 tests)

- [ ] **Step 3: Commit**

```bash
git add reasonbench/__init__.py
git commit -m "feat: update public API with Phase 5 exports (Experiment, ReportBuilder)"
```

---

## Summary

| Task | Component | New/Modified Files | Tests |
|------|-----------|-------------------|-------|
| 1 | ExperimentRound model | 2 (models, test_models) | 3 |
| 2 | Experiment runner | 2 (experiment, test) | 10 |
| 3 | ReportBuilder | 2 (report, test) | 10 |
| 4 | CLI experiment subcommand | 2 (__main__, test_cli) | 2 |
| 5 | Init update | 1 (__init__) | 0 |
| **Total** | | **9 files** | **~25 new tests** |

**Phase 5 delivers:** A multi-round experiment orchestrator that chains the full feedback loop (generate → evaluate → analyze → evolve → re-evaluate) with per-round JSONL output, root cause extraction, self-repair testing, and structured reports in JSON and markdown format. CLI: `reasonbench experiment --models m --judge j --evolve-model e --rounds 3 --count 10`.
