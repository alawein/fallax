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
        result = ValidatorPack.reasoning_critic(prompt="p", answer="a", reasoning="r")
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
        result = ValidatorPack.truth_judge(prompt="my prompt", answer="my answer")
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
