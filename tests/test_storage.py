from pathlib import Path

import pytest

from reasonbench.models import (
    EvaluationResult,
    ModelResponse,
    ValidationResult,
)
from reasonbench.storage import JsonlStore
from reasonbench.taxonomy import FailureType, Severity


def _make_result(prompt_id: str, score: int) -> EvaluationResult:
    """Helper to build a minimal EvaluationResult."""
    return EvaluationResult(
        prompt_id=prompt_id,
        failure_type=FailureType.CONTRADICTION,
        prompt_text="test prompt",
        models={
            "test-model": ModelResponse(
                model_name="test-model",
                answer="42",
                reasoning="because",
                is_correct=False,
            ),
        },
        validation=ValidationResult(reasoning_flawed=True),
        score=score,
        severity=Severity.HIGH if score >= 4 else Severity.LOW,
    )


class TestJsonlStore:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> JsonlStore:
        return JsonlStore(tmp_path / "results.jsonl")

    def test_read_nonexistent_returns_empty(self, store):
        assert store.read_all() == []

    def test_count_nonexistent_returns_zero(self, store):
        assert store.count() == 0

    def test_append_and_read_single(self, store):
        result = _make_result("id-1", score=5)
        store.append(result)
        loaded = store.read_all()
        assert len(loaded) == 1
        assert loaded[0].prompt_id == "id-1"
        assert loaded[0].score == 5

    def test_append_multiple_and_count(self, store):
        store.append(_make_result("id-1", score=3))
        store.append(_make_result("id-2", score=7))
        store.append(_make_result("id-3", score=1))
        assert store.count() == 3

    def test_read_by_min_score(self, store):
        store.append(_make_result("low", score=1))
        store.append(_make_result("mid", score=4))
        store.append(_make_result("high", score=8))
        results = store.read_by_min_score(4)
        ids = {r.prompt_id for r in results}
        assert ids == {"mid", "high"}

    def test_roundtrip_preserves_nested_data(self, store):
        result = _make_result("nested", score=6)
        store.append(result)
        loaded = store.read_all()[0]
        assert loaded.models["test-model"].answer == "42"
        assert loaded.validation.reasoning_flawed is True

    def test_append_is_additive(self, store):
        store.append(_make_result("first", score=1))
        assert store.count() == 1
        store.append(_make_result("second", score=2))
        assert store.count() == 2
        # first result still present
        ids = {r.prompt_id for r in store.read_all()}
        assert "first" in ids
