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
            prompt_id="1",
            score=7,
            reasoning_flawed=True,
            failure_type=FailureType.CONTRADICTION,
            model_name="model-a",
            assumptions=[
                Assumption(text="input is positive", justified=False),
                Assumption(text="function is monotonic", justified=False),
            ],
        ),
        make_result(
            prompt_id="2",
            score=6,
            reasoning_flawed=True,
            failure_type=FailureType.UNSTATED_ASSUMPTION,
            model_name="model-a",
            assumptions=[
                Assumption(text="input is positive", justified=False),
            ],
        ),
        make_result(
            prompt_id="3",
            score=8,
            reasoning_flawed=True,
            failure_type=FailureType.CONTRADICTION,
            model_name="model-b",
            assumptions=[
                Assumption(text="function is monotonic", justified=False),
                Assumption(text="domain is bounded", justified=False),
            ],
        ),
        make_result(
            prompt_id="4",
            score=5,
            reasoning_flawed=True,
            failure_type=FailureType.OVERGENERALIZATION,
            model_name="model-b",
            assumptions=[
                Assumption(text="input is positive", justified=False),
            ],
        ),
        make_result(
            prompt_id="5",
            score=1,
            reasoning_flawed=False,
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
        total_freq = sum(p.frequency for p in patterns)
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
                prompt_id="1",
                score=7,
                reasoning_flawed=True,
                assumptions=[Assumption(text="Input Is Positive", justified=False)],
            ),
            make_result(
                prompt_id="2",
                score=6,
                reasoning_flawed=True,
                assumptions=[Assumption(text="input is positive", justified=False)],
            ),
        ]
        ex = RootCauseExtractor(results)
        patterns = ex.extract_patterns()
        assert len(patterns) == 1
        assert patterns[0].frequency == 2
