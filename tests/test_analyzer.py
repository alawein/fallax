import pytest

from reasonbench.analyzer import Analyzer
from reasonbench.models import Assumption
from reasonbench.taxonomy import FailureType, Severity
from tests.conftest import make_result


@pytest.fixture()
def results():
    """Mixed set of 6 results for analytics testing."""
    return [
        make_result(prompt_id="1", score=8, reasoning_flawed=True, failure_type=FailureType.CONTRADICTION, prompt_text="contradiction prompt", assumptions=[Assumption(text="a", justified=False)]),
        make_result(prompt_id="2", score=7, reasoning_flawed=True, failure_type=FailureType.CONTRADICTION, prompt_text="another contradiction", assumptions=[Assumption(text="b", justified=False), Assumption(text="c", justified=True)]),
        make_result(prompt_id="3", score=3, reasoning_flawed=True, failure_type=FailureType.UNSTATED_ASSUMPTION, prompt_text="assumption prompt"),
        make_result(prompt_id="4", score=1, reasoning_flawed=False, failure_type=FailureType.UNSTATED_ASSUMPTION, prompt_text="easy assumption"),
        make_result(prompt_id="5", score=5, reasoning_flawed=True, failure_type=FailureType.OVERGENERALIZATION, prompt_text="overgeneralization prompt", assumptions=[Assumption(text="d", justified=False)]),
        make_result(prompt_id="6", score=0, reasoning_flawed=False, failure_type=FailureType.OVERGENERALIZATION, prompt_text="easy overgeneralization"),
    ]


class TestAnalyzer:
    def test_summary_total(self, results):
        a = Analyzer(results)
        s = a.summary()
        assert s["total"] == 6

    def test_summary_avg_score(self, results):
        a = Analyzer(results)
        s = a.summary()
        # (8+7+3+1+5+0)/6 = 4.0
        assert abs(s["avg_score"] - 4.0) < 0.01

    def test_summary_failure_rate(self, results):
        a = Analyzer(results)
        s = a.summary()
        assert abs(s["failure_rate"] - 4 / 6) < 0.01

    def test_summary_severity_counts(self, results):
        a = Analyzer(results)
        s = a.summary()
        counts = s["severity_counts"]
        assert counts[Severity.CRITICAL] == 2  # scores 8, 7
        assert counts[Severity.HIGH] == 1  # score 5
        assert counts[Severity.LOW] == 2  # scores 1, 0

    def test_failure_rate_by_type(self, results):
        a = Analyzer(results)
        rates = a.failure_rate_by_type()
        assert rates["contradiction"] == 1.0
        assert rates["unstated_assumption"] == 0.5

    def test_model_accuracy(self, results):
        a = Analyzer(results)
        acc = a.model_accuracy()
        assert "test-model" in acc
        assert abs(acc["test-model"] - 2 / 6) < 0.01

    def test_top_failures(self, results):
        a = Analyzer(results)
        top = a.top_failures(n=3)
        assert len(top) == 3
        assert top[0].score >= top[1].score >= top[2].score
        assert top[0].score == 8

    def test_hard_cases(self, results):
        a = Analyzer(results)
        hard = a.hard_cases(min_score=6)
        assert len(hard) == 2  # scores 8, 7
        assert all(r.score >= 6 for r in hard)

    def test_hard_cases_empty(self, results):
        a = Analyzer(results)
        hard = a.hard_cases(min_score=100)
        assert hard == []

    def test_disagreement_rate_single_model(self, results):
        a = Analyzer(results)
        assert a.disagreement_rate() == 0.0

    def test_assumption_density(self, results):
        a = Analyzer(results)
        density = a.assumption_density()
        assert abs(density - 3 / 6) < 0.01

    def test_empty_results(self):
        a = Analyzer([])
        s = a.summary()
        assert s["total"] == 0
        assert s["avg_score"] == 0.0
        assert s["failure_rate"] == 0.0
        assert a.disagreement_rate() == 0.0
        assert a.assumption_density() == 0.0
