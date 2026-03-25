from pathlib import Path
import pytest
from reasonbench.predictor import FailurePredictor
from reasonbench.taxonomy import FailureType
from tests.conftest import make_result


@pytest.fixture()
def training_data():
    """20 results with varied scores and distinct prompt texts."""
    data = []
    high_words = [
        "contradictory constraints impossible", "recursive definition undefined",
        "edge case boundary fails", "implicit assumption hidden rule",
        "overconstrained no solution exists", "ambiguous specification unclear",
        "multi step dependency chain error", "false analogy pattern mismatch",
        "self consistency logical flaw", "hidden variable missing information",
    ]
    low_words = [
        "simple arithmetic addition", "basic string reversal",
        "trivial sorting ascending", "easy boolean logic gate",
        "straightforward lookup table", "plain concatenation operation",
        "direct mapping function", "clear input validation",
        "obvious pattern matching", "standard list filtering",
    ]
    for i, text in enumerate(high_words):
        data.append(make_result(prompt_id=f"high-{i}", prompt_text=text, score=7, reasoning_flawed=True))
    for i, text in enumerate(low_words):
        # Alternate between score=1 and score=3 so threshold=2 captures the 3s
        score = 3 if i % 2 == 0 else 1
        data.append(make_result(prompt_id=f"low-{i}", prompt_text=text, score=score, reasoning_flawed=False))
    return data


class TestFailurePredictor:
    def test_not_fitted_initially(self):
        p = FailurePredictor()
        assert p.is_fitted is False

    def test_predict_before_train_raises(self):
        p = FailurePredictor()
        with pytest.raises(RuntimeError, match="not trained"):
            p.predict("some prompt")

    def test_predict_batch_before_train_raises(self):
        p = FailurePredictor()
        with pytest.raises(RuntimeError, match="not trained"):
            p.predict_batch(["a", "b"])

    def test_train_returns_metrics(self, training_data):
        p = FailurePredictor()
        metrics = p.train(training_data, threshold=4)
        assert metrics["samples"] == 20
        assert abs(metrics["positive_rate"] - 0.5) < 0.01
        assert 0.0 <= metrics["cv_accuracy"] <= 1.0

    def test_is_fitted_after_train(self, training_data):
        p = FailurePredictor()
        p.train(training_data)
        assert p.is_fitted is True

    def test_predict_returns_probability(self, training_data):
        p = FailurePredictor()
        p.train(training_data)
        prob = p.predict("contradictory constraints impossible situation")
        assert 0.0 <= prob <= 1.0

    def test_predict_batch_returns_list(self, training_data):
        p = FailurePredictor()
        p.train(training_data)
        probs = p.predict_batch(["hard constraint problem", "easy simple task"])
        assert len(probs) == 2
        assert all(0.0 <= prob <= 1.0 for prob in probs)

    def test_high_failure_prompt_scores_higher(self, training_data):
        p = FailurePredictor()
        p.train(training_data, threshold=4)
        high_prob = p.predict("contradictory recursive undefined edge case")
        low_prob = p.predict("simple basic easy trivial straightforward")
        assert high_prob > low_prob

    def test_save_and_load(self, training_data, tmp_path):
        p = FailurePredictor()
        p.train(training_data)
        model_path = tmp_path / "predictor.pkl"
        p.save(model_path)
        assert model_path.exists()
        p2 = FailurePredictor()
        assert p2.is_fitted is False
        p2.load(model_path)
        assert p2.is_fitted is True
        prob1 = p.predict("test prompt")
        prob2 = p2.predict("test prompt")
        assert abs(prob1 - prob2) < 0.001

    def test_train_with_custom_threshold(self, training_data):
        p = FailurePredictor()
        metrics = p.train(training_data, threshold=2)
        assert metrics["positive_rate"] > 0.5
