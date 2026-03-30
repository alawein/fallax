"""Tests for the dashboard API."""

import json

import pytest
from starlette.testclient import TestClient

from dashboard.api import create_app
from reasonbench.models import (
    EvaluationResult,
    ModelResponse,
    ValidationResult,
)
from reasonbench.taxonomy import FailureType, Severity


@pytest.fixture()
def experiment_dir(tmp_path):
    """Create a mock experiment output directory."""
    exp = tmp_path / "test_experiment"
    exp.mkdir()

    results = [
        EvaluationResult(
            prompt_id="p1",
            failure_type=FailureType.UNSTATED_ASSUMPTION,
            prompt_text="Test prompt 1",
            models={
                "model-a": ModelResponse(
                    model_name="model-a",
                    answer="42",
                    reasoning="Because math",
                    is_correct=True,
                ),
            },
            validation=ValidationResult(reasoning_flawed=False),
            score=2,
            severity=Severity.LOW,
        ),
        EvaluationResult(
            prompt_id="p2",
            failure_type=FailureType.CONTRADICTION,
            prompt_text="Test prompt 2",
            models={
                "model-a": ModelResponse(
                    model_name="model-a",
                    answer="wrong",
                    reasoning="Bad logic",
                    is_correct=False,
                ),
            },
            validation=ValidationResult(reasoning_flawed=True),
            score=7,
            severity=Severity.CRITICAL,
        ),
    ]

    with open(exp / "round_1.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")

    report = {
        "total_rounds": 1,
        "total_prompts": 2,
        "total_failures": 1,
        "score_trend": [4.5],
        "failure_trend": [0.5],
        "score_delta": 0.0,
        "failure_delta": 0.0,
        "hardening_rate": 0.0,
        "repair_success_rate": None,
        "top_patterns": [
            {
                "pattern": "Missing edge case",
                "frequency": 2,
                "models_affected": ["model-a"],
                "failure_types": ["unstated_assumption"],
            }
        ],
    }
    (exp / "report.json").write_text(json.dumps(report), encoding="utf-8")

    return tmp_path


@pytest.fixture()
def client(experiment_dir):
    """Create a test client for the dashboard API."""
    app = create_app(data_dir=experiment_dir)
    return TestClient(app)


class TestListExperiments:
    def test_returns_experiments(self, client):
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "test_experiment"
        assert data[0]["total_prompts"] == 2
        assert data[0]["total_rounds"] == 1

    def test_empty_dir(self, tmp_path):
        app = create_app(data_dir=tmp_path)
        c = TestClient(app)
        resp = c.get("/api/experiments")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_nonexistent_dir(self, tmp_path):
        app = create_app(data_dir=tmp_path / "nope")
        c = TestClient(app)
        resp = c.get("/api/experiments")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetReport:
    def test_returns_report(self, client):
        resp = client.get("/api/experiments/test_experiment/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rounds"] == 1
        assert data["total_prompts"] == 2
        assert len(data["top_patterns"]) == 1

    def test_not_found(self, client):
        resp = client.get("/api/experiments/nonexistent/report")
        assert resp.status_code == 404


class TestGetResults:
    def test_returns_all_results(self, client):
        resp = client.get("/api/experiments/test_experiment/results")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_filter_by_min_score(self, client):
        resp = client.get("/api/experiments/test_experiment/results?min_score=5")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["score"] == 7

    def test_filter_by_round(self, client):
        resp = client.get("/api/experiments/test_experiment/results?round_num=1")
        data = resp.json()
        assert len(data) == 2

    def test_nonexistent_round(self, client):
        resp = client.get("/api/experiments/test_experiment/results?round_num=99")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_not_found(self, client):
        resp = client.get("/api/experiments/nonexistent/results")
        assert resp.status_code == 404


class TestGetSummary:
    def test_returns_summary(self, client):
        resp = client.get("/api/experiments/test_experiment/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["avg_score"] == 4.5
        assert data["failure_rate"] == 0.5
        assert "assumption_error" in data["by_category"]
        assert "logic_error" in data["by_category"]
        assert "unstated_assumption" in data["by_type"]
        assert "2" in data["score_distribution"]
        assert "7" in data["score_distribution"]

    def test_severity_breakdown(self, client):
        resp = client.get("/api/experiments/test_experiment/summary")
        data = resp.json()
        assert data["by_severity"]["low"] == 1
        assert data["by_severity"]["critical"] == 1


class TestModelComparison:
    def test_returns_model_stats(self, client):
        resp = client.get("/api/experiments/test_experiment/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["model"] == "model-a"
        assert data[0]["total"] == 2
        assert data[0]["correct"] == 1
        assert data[0]["accuracy"] == 0.5

    def test_not_found(self, client):
        resp = client.get("/api/experiments/nonexistent/models")
        assert resp.status_code == 404
