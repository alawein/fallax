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

    def test_negative_assumption_errors_floored_at_zero(self):
        score = Scorer.compute_score(
            is_correct=True,
            reasoning_flawed=False,
            assumption_errors=-5,
            counterfactual_fail=False,
            model_disagreement=False,
        )
        assert score == 0

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
        assert (
            Scorer.hardness(wrong_models=0, reasoning_failures=0, repair_failures=0)
            == 0
        )

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
