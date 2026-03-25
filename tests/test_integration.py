"""End-to-end integration test: template -> render -> mock response -> score -> store -> read."""

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
