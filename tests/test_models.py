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


from reasonbench.models import RepairResult, RootCausePattern


class TestRepairResult:
    def test_create_minimal(self):
        r = RepairResult(
            model_name="model-a", prompt_text="test prompt",
            original_answer="wrong", repaired_answer="fixed",
            repair_reasoning="I was wrong because...",
        )
        assert r.model_name == "model-a"
        assert r.is_fixed is None

    def test_create_full(self):
        r = RepairResult(
            model_name="model-a", prompt_text="test prompt",
            original_answer="wrong", repaired_answer="fixed",
            repair_reasoning="I was wrong because...", is_fixed=True,
        )
        assert r.is_fixed is True

    def test_json_roundtrip(self):
        r = RepairResult(
            model_name="m", prompt_text="p", original_answer="a",
            repaired_answer="b", repair_reasoning="r", is_fixed=False,
        )
        restored = RepairResult.model_validate_json(r.model_dump_json())
        assert restored.is_fixed is False
        assert restored.original_answer == "a"


class TestRootCausePattern:
    def test_create(self):
        p = RootCausePattern(
            pattern="assumes monotonicity", frequency=15,
            models_affected=["model-a", "model-b"],
            example_prompt="Is f(x) monotonic?",
            failure_types=["contradiction", "unstated_assumption"],
        )
        assert p.frequency == 15
        assert len(p.models_affected) == 2

    def test_json_roundtrip(self):
        p = RootCausePattern(
            pattern="test", frequency=1, models_affected=["m"],
            example_prompt="p", failure_types=["contradiction"],
        )
        restored = RootCausePattern.model_validate_json(p.model_dump_json())
        assert restored.pattern == "test"


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
