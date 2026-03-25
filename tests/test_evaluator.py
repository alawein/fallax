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
