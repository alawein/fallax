import json
from pathlib import Path

import pytest

from reasonbench.models import EvaluationResult
from reasonbench.pipeline import Pipeline
from reasonbench.taxonomy import Severity
from tests.conftest import JUDGE_RESPONSES, MODEL_RESPONSE_TEXT, MockClient


@pytest.fixture()
def params_dir(tmp_path: Path) -> Path:
    """Minimal param bank for pipeline testing."""
    data = [
        {"rule_a": "x > 0", "rule_b": "double it", "edge_case_input": "x = -1"},
        {"rule_a": "len < 5", "rule_b": "reverse", "edge_case_input": "empty"},
    ]
    (tmp_path / "implicit_assumption_trap.json").write_text(json.dumps(data))
    return tmp_path


@pytest.fixture()
def pipeline_client():
    """Client that handles both model and judge calls."""
    responses = dict(JUDGE_RESPONSES)
    return MockClient(responses=responses, default=MODEL_RESPONSE_TEXT)


@pytest.fixture()
def output_path(tmp_path: Path) -> Path:
    return tmp_path / "results.jsonl"


class TestPipeline:
    def test_run_returns_results(self, pipeline_client, params_dir, output_path):
        pipeline = Pipeline(
            client=pipeline_client,
            models=["model-a"],
            judge_model="judge",
            output_path=output_path,
            params_dir=params_dir,
            seed=42,
        )
        results = pipeline.run(count=2)
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_results_stored_to_jsonl(self, pipeline_client, params_dir, output_path):
        pipeline = Pipeline(
            client=pipeline_client,
            models=["model-a"],
            judge_model="judge",
            output_path=output_path,
            params_dir=params_dir,
            seed=42,
        )
        pipeline.run(count=2)
        assert output_path.exists()
        lines = [l for l in output_path.read_text().strip().split("\n") if l]
        assert len(lines) == 2

    def test_dual_model_disagreement(self, params_dir, output_path):
        class AlternatingClient:
            def __init__(self):
                self.calls = []
                self._n = 0

            def complete(self, prompt, *, model):
                self.calls.append((prompt, model))
                self._n += 1
                for kw, resp in JUDGE_RESPONSES.items():
                    if kw in prompt:
                        return resp
                if self._n % 2 == 0:
                    return "Step 1: no.\nANSWER: yes"
                return "Step 1: yes.\nANSWER: no"

        client = AlternatingClient()
        pipeline = Pipeline(
            client=client,
            models=["model-a", "model-b"],
            judge_model="judge",
            output_path=output_path,
            params_dir=params_dir,
            seed=42,
        )
        results = pipeline.run(count=1)
        assert len(results) == 1
        assert "model-a" in results[0].models
        assert "model-b" in results[0].models

    def test_score_and_severity_populated(self, pipeline_client, params_dir, output_path):
        pipeline = Pipeline(
            client=pipeline_client,
            models=["model-a"],
            judge_model="judge",
            output_path=output_path,
            params_dir=params_dir,
            seed=42,
        )
        results = pipeline.run(count=1)
        r = results[0]
        assert isinstance(r.score, int)
        assert isinstance(r.severity, Severity)
        assert r.score >= 0

    def test_empty_params_dir_returns_empty(self, pipeline_client, tmp_path):
        empty_params = tmp_path / "empty_params"
        empty_params.mkdir()
        pipeline = Pipeline(
            client=pipeline_client,
            models=["m"],
            judge_model="j",
            output_path=tmp_path / "out.jsonl",
            params_dir=empty_params,
            seed=42,
        )
        results = pipeline.run(count=10)
        assert results == []
