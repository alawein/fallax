"""Tests for the dashboard API."""

import json
import os

import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from dashboard.api import _resolve_experiment_dir, create_app
from fallax.models import (
    EvaluationResult,
    ModelResponse,
    ValidationResult,
)
from fallax.taxonomy import FailureType, Severity


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


class TestPathTraversal:
    """Route-level smoke test that hostile names are rejected.

    Many of these payloads never reach `_resolve_experiment_dir` because
    Starlette's router rejects them earlier (path with `/`, leading slash,
    etc.). That is fine — the assertion is that nothing leaks. For direct
    coverage of the guard itself, see TestExperimentDirGuard below.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "../etc",
            "..%2F..%2Fetc",
            "..\\..\\windows",
            "/etc/passwd",
            "test_experiment/../..",
        ],
    )
    def test_traversal_rejected(self, client, name):
        for path in (
            f"/api/experiments/{name}/report",
            f"/api/experiments/{name}/results",
            f"/api/experiments/{name}/summary",
            f"/api/experiments/{name}/models",
        ):
            resp = client.get(path)
            # Strict 404 only; 405 here would indicate a routing surprise,
            # not the guard firing.
            assert resp.status_code == 404, (path, resp.status_code)
            # And nothing absolute-looking should ever appear in the response.
            assert "/etc/" not in resp.text
            assert "C:\\" not in resp.text


class TestExperimentDirGuard:
    """Direct unit tests against `_resolve_experiment_dir`.

    Bypassing the router is the only way to confirm the .resolve() +
    is_relative_to() check actually runs. Locks against silent regression
    of the security-critical guard.
    """

    def test_legitimate_subdir(self, tmp_path):
        (tmp_path / "real_exp").mkdir()
        resolved = _resolve_experiment_dir("real_exp", tmp_path)
        assert resolved == (tmp_path / "real_exp").resolve()

    def test_nonexistent_subdir(self, tmp_path):
        with pytest.raises(HTTPException) as exc:
            _resolve_experiment_dir("nope", tmp_path)
        assert exc.value.status_code == 404

    @pytest.mark.parametrize("name", ["", ".", ".."])
    def test_empty_or_dot_rejected(self, tmp_path, name):
        with pytest.raises(HTTPException) as exc:
            _resolve_experiment_dir(name, tmp_path)
        assert exc.value.status_code == 404

    def test_traversal_via_relative_parent(self, tmp_path):
        (tmp_path / "real").mkdir()
        outside = tmp_path.parent / "outside"
        outside.mkdir(exist_ok=True)
        try:
            # Build a name that resolves outside `tmp_path`. ".." inside the
            # name is normalized by Path.resolve(), so this escapes base.
            with pytest.raises(HTTPException) as exc:
                _resolve_experiment_dir("real/../../outside", tmp_path)
            assert exc.value.status_code == 404
        finally:
            outside.rmdir()

    @pytest.mark.skipif(
        os.name == "nt",
        reason="symlink creation needs admin or Developer Mode on Windows",
    )
    def test_symlink_escape_rejected(self, tmp_path):
        outside = tmp_path.parent / "symlink_outside"
        outside.mkdir(exist_ok=True)
        try:
            link = tmp_path / "escape"
            link.symlink_to(outside, target_is_directory=True)
            with pytest.raises(HTTPException) as exc:
                _resolve_experiment_dir("escape", tmp_path)
            assert exc.value.status_code == 404
        finally:
            outside.rmdir()

    def test_path_with_nul_byte_rejected(self, tmp_path):
        with pytest.raises(HTTPException) as exc:
            _resolve_experiment_dir("real\x00bypass", tmp_path)
        assert exc.value.status_code == 404


class TestSilentFailureHandling:
    """Verify the per-record try/except hardening behaves as designed."""

    def test_list_experiments_skips_corrupt_report(self, tmp_path):
        good = tmp_path / "good"
        good.mkdir()
        (good / "report.json").write_text(
            json.dumps(
                {
                    "total_rounds": 1,
                    "total_prompts": 1,
                    "total_failures": 0,
                    "score_trend": [],
                    "failure_trend": [],
                    "score_delta": 0.0,
                    "failure_delta": 0.0,
                    "hardening_rate": 0.0,
                    "repair_success_rate": None,
                    "top_patterns": [],
                }
            )
        )
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "report.json").write_text("{not json")

        client = TestClient(create_app(data_dir=tmp_path))
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        names = [e["name"] for e in resp.json()]
        assert names == ["good"]

    def test_get_report_500s_on_malformed_json(self, tmp_path):
        exp = tmp_path / "exp"
        exp.mkdir()
        (exp / "report.json").write_text("{not json")
        client = TestClient(create_app(data_dir=tmp_path))
        resp = client.get("/api/experiments/exp/report")
        assert resp.status_code == 500
        assert "malformed" in resp.json()["detail"].lower()

    def test_get_results_500s_on_malformed_jsonl_line(self, tmp_path):
        exp = tmp_path / "exp"
        exp.mkdir()
        (exp / "report.json").write_text("{}")
        (exp / "round_1.jsonl").write_text("not a json line\n")
        client = TestClient(create_app(data_dir=tmp_path))
        resp = client.get("/api/experiments/exp/results")
        assert resp.status_code == 500
        # Error should cite the file and line for diagnosability.
        assert "round_1.jsonl:1" in resp.json()["detail"]
