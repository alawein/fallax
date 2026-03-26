"""Tests for benchmark suite management."""

import json

import pytest

from reasonbench.benchmark import (
    BenchmarkBaselines,
    BenchmarkMetadata,
    BenchmarkSuite,
    ModelBaseline,
)
from reasonbench.models import EvaluationResult, ModelResponse, Prompt, ValidationResult
from reasonbench.taxonomy import FailureType, Severity


@pytest.fixture()
def benchmark_dir(tmp_path):
    """Create a temporary benchmark directory with a v1 suite."""
    v1 = tmp_path / "v1"
    v1.mkdir()

    prompts = [
        Prompt(
            prompt_id="p1",
            failure_type=FailureType.UNSTATED_ASSUMPTION,
            prompt_text="Test prompt 1",
            difficulty=5,
            template_id="implicit_assumption_trap",
        ),
        Prompt(
            prompt_id="p2",
            failure_type=FailureType.CONTRADICTION,
            prompt_text="Test prompt 2",
            difficulty=7,
            template_id="recursive_definition_break",
        ),
        Prompt(
            prompt_id="p3",
            failure_type=FailureType.OVERGENERALIZATION,
            prompt_text="Test prompt 3",
            difficulty=5,
            template_id="edge_case_inversion",
        ),
    ]

    with open(v1 / "prompts.jsonl", "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p.model_dump_json() + "\n")

    metadata = {
        "version": "v1",
        "created": "2026-03-25T00:00:00+00:00",
        "description": "Test benchmark v1",
        "prompt_count": 3,
        "failure_types": ["unstated_assumption", "contradiction", "overgeneralization"],
        "categories": ["assumption_error", "logic_error", "generalization_error"],
        "generation_params": {"seed": 42},
    }
    (v1 / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    baselines = {
        "version": "v1",
        "models": [
            {
                "model_name": "test-model",
                "overall_score": 3.5,
                "failure_rate": 0.33,
                "runs": 1,
                "assumption_density": 0.5,
            }
        ],
    }
    (v1 / "baselines.json").write_text(json.dumps(baselines), encoding="utf-8")

    return tmp_path


@pytest.fixture()
def suite(benchmark_dir):
    return BenchmarkSuite(benchmarks_dir=benchmark_dir)


class TestBenchmarkSuite:
    def test_versions(self, suite):
        assert suite.versions() == ["v1"]

    def test_versions_empty(self, tmp_path):
        s = BenchmarkSuite(benchmarks_dir=tmp_path)
        assert s.versions() == []

    def test_versions_nonexistent_dir(self, tmp_path):
        s = BenchmarkSuite(benchmarks_dir=tmp_path / "nope")
        assert s.versions() == []

    def test_load_prompts(self, suite):
        prompts = suite.load_prompts("v1")
        assert len(prompts) == 3
        assert prompts[0].prompt_id == "p1"
        assert prompts[1].failure_type == FailureType.CONTRADICTION

    def test_load_prompts_not_found(self, suite):
        with pytest.raises(FileNotFoundError):
            suite.load_prompts("v99")

    def test_load_metadata(self, suite):
        meta = suite.load_metadata("v1")
        assert meta.version == "v1"
        assert meta.prompt_count == 3
        assert "unstated_assumption" in meta.failure_types

    def test_load_metadata_not_found(self, suite):
        with pytest.raises(FileNotFoundError):
            suite.load_metadata("v99")

    def test_load_baselines(self, suite):
        baselines = suite.load_baselines("v1")
        assert baselines.version == "v1"
        assert len(baselines.models) == 1
        assert baselines.models[0].model_name == "test-model"
        assert baselines.models[0].overall_score == 3.5

    def test_load_baselines_missing_returns_empty(self, suite, benchmark_dir):
        (benchmark_dir / "v1" / "baselines.json").unlink()
        baselines = suite.load_baselines("v1")
        assert baselines.version == "v1"
        assert baselines.models == []

    def test_save_baselines(self, suite, benchmark_dir):
        new_baselines = BenchmarkBaselines(
            version="v1",
            models=[
                ModelBaseline(
                    model_name="new-model",
                    overall_score=4.2,
                    failure_rate=0.5,
                )
            ],
        )
        path = suite.save_baselines(new_baselines)
        assert path.exists()
        loaded = suite.load_baselines("v1")
        assert loaded.models[0].model_name == "new-model"

    def test_create_version(self, suite, benchmark_dir):
        prompts = [
            Prompt(
                prompt_id="new1",
                failure_type=FailureType.INVALID_INFERENCE,
                prompt_text="New prompt",
                difficulty=6,
            ),
        ]
        vdir = suite.create_version("v2", prompts, description="Test v2")
        assert (vdir / "prompts.jsonl").exists()
        assert (vdir / "metadata.json").exists()

        loaded = suite.load_prompts("v2")
        assert len(loaded) == 1
        meta = suite.load_metadata("v2")
        assert meta.version == "v2"
        assert meta.prompt_count == 1

    def test_create_version_duplicate_raises(self, suite):
        with pytest.raises(FileExistsError):
            suite.create_version("v1", [])

    def test_versions_lists_multiple(self, suite, benchmark_dir):
        prompts = [
            Prompt(
                prompt_id="x",
                failure_type=FailureType.AMBIGUITY_FAILURE,
                prompt_text="x",
                difficulty=3,
            ),
        ]
        suite.create_version("v2", prompts)
        assert suite.versions() == ["v1", "v2"]


class TestScoreResults:
    def _make_result(self, score, failure_type=FailureType.UNSTATED_ASSUMPTION):
        return EvaluationResult(
            prompt_id="test",
            failure_type=failure_type,
            prompt_text="test",
            models={"m": ModelResponse(model_name="m", answer="a", reasoning="r")},
            validation=ValidationResult(reasoning_flawed=False),
            score=score,
            severity=Severity.LOW,
        )

    def test_empty_results(self):
        suite = BenchmarkSuite()
        scores = suite.score_results([])
        assert scores["overall_score"] == 0.0
        assert scores["total"] == 0

    def test_single_result(self):
        suite = BenchmarkSuite()
        results = [self._make_result(3)]
        scores = suite.score_results(results)
        assert scores["overall_score"] == 3.0
        assert scores["failure_rate"] == 0.0
        assert scores["total"] == 1

    def test_failure_rate_threshold(self):
        suite = BenchmarkSuite()
        results = [self._make_result(2), self._make_result(5)]
        scores = suite.score_results(results)
        assert scores["failure_rate"] == 0.5

    def test_category_breakdown(self):
        suite = BenchmarkSuite()
        results = [
            self._make_result(3, FailureType.UNSTATED_ASSUMPTION),
            self._make_result(5, FailureType.CONTRADICTION),
        ]
        scores = suite.score_results(results)
        assert "assumption_error" in scores["category_scores"]
        assert "logic_error" in scores["category_scores"]

    def test_type_breakdown(self):
        suite = BenchmarkSuite()
        results = [
            self._make_result(3, FailureType.UNSTATED_ASSUMPTION),
            self._make_result(7, FailureType.UNSTATED_ASSUMPTION),
        ]
        scores = suite.score_results(results)
        assert scores["type_scores"]["unstated_assumption"] == 5.0


class TestBenchmarkModels:
    def test_metadata_fields(self):
        meta = BenchmarkMetadata(
            version="v1",
            created="2026-01-01",
            description="test",
            prompt_count=10,
            failure_types=["a"],
            categories=["b"],
        )
        assert meta.version == "v1"
        assert meta.prompt_count == 10

    def test_model_baseline_defaults(self):
        b = ModelBaseline(
            model_name="test",
            overall_score=5.0,
            failure_rate=0.3,
        )
        assert b.runs == 1
        assert b.assumption_density == 0.0
        assert b.category_scores == {}

    def test_baselines_roundtrip(self):
        baselines = BenchmarkBaselines(
            version="v1",
            models=[
                ModelBaseline(
                    model_name="m1",
                    overall_score=4.0,
                    failure_rate=0.25,
                    category_scores={"logic_error": 3.5},
                ),
            ],
        )
        data = baselines.model_dump_json()
        loaded = BenchmarkBaselines.model_validate_json(data)
        assert loaded.models[0].model_name == "m1"
        assert loaded.models[0].category_scores["logic_error"] == 3.5
