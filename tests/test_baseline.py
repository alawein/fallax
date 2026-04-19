"""Tests for baseline subcommands."""

from __future__ import annotations

import functools
import json

import pytest

from reasonbench.__main__ import main
from reasonbench.benchmark import BenchmarkSuite


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def benchmark_dir(tmp_path):
    """Temp benchmark dir with a v1 suite that has one baseline entry."""
    v1 = tmp_path / "v1"
    v1.mkdir()

    # minimal prompts.jsonl so load_prompts() doesn't raise
    from reasonbench.models import Prompt
    from reasonbench.taxonomy import FailureType
    prompt = Prompt(
        prompt_id="p1",
        failure_type=FailureType.UNSTATED_ASSUMPTION,
        prompt_text="Test prompt",
        difficulty=5,
        template_id="implicit_assumption_trap",
    )
    (v1 / "prompts.jsonl").write_text(prompt.model_dump_json() + "\n", encoding="utf-8")

    baselines = {
        "version": "v1",
        "models": [
            {
                "model_name": "base-model",
                "overall_score": 4.0,
                "failure_rate": 0.2,
                "runs": 1,
                "assumption_density": 0.0,
                "captured_at": "2026-04-18T10:00:00+00:00",
            }
        ],
    }
    (v1 / "baselines.json").write_text(json.dumps(baselines), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def patch_suite(benchmark_dir, monkeypatch):
    """Patch BenchmarkSuite in __main__ to use benchmark_dir."""
    monkeypatch.setattr(
        "reasonbench.__main__.BenchmarkSuite",
        functools.partial(BenchmarkSuite, benchmarks_dir=benchmark_dir),
    )
    return benchmark_dir


# ---------------------------------------------------------------------------
# baseline status
# ---------------------------------------------------------------------------

class TestBaselineStatus:
    def test_status_prints_model_name(self, patch_suite, capsys):
        code = main(["baseline", "status", "--version", "v1"])
        assert code == 0
        captured = capsys.readouterr()
        assert "base-model" in captured.out

    def test_status_empty_baselines(self, patch_suite, benchmark_dir, capsys):
        (benchmark_dir / "v1" / "baselines.json").write_text(
            json.dumps({"version": "v1", "models": []}), encoding="utf-8"
        )
        code = main(["baseline", "status", "--version", "v1"])
        assert code == 0
        captured = capsys.readouterr()
        assert "No baselines" in captured.out


# ---------------------------------------------------------------------------
# baseline capture
# ---------------------------------------------------------------------------

class TestBaselineCapture:
    def _fake_results(self):
        """Two fake EvaluationResults for deterministic scoring."""
        from tests.conftest import make_result
        return [make_result(score=3), make_result(score=5)]

    def test_capture_writes_entry(self, patch_suite, benchmark_dir, monkeypatch, capsys):
        from unittest.mock import patch as mock_patch
        fake = self._fake_results()
        with mock_patch("reasonbench.__main__.Pipeline") as MockPipeline:
            MockPipeline.return_value.run_prompts.return_value = fake
            code = main([
                "baseline", "capture",
                "--version", "v1",
                "--model", "new-model",
                "--judge", "judge-model",
            ])
        assert code == 0
        # reload baselines from disk
        baselines_path = benchmark_dir / "v1" / "baselines.json"
        data = json.loads(baselines_path.read_text(encoding="utf-8"))
        names = [m["model_name"] for m in data["models"]]
        assert "new-model" in names

    def test_capture_replaces_existing_entry(self, patch_suite, benchmark_dir, monkeypatch):
        from unittest.mock import patch as mock_patch
        fake = self._fake_results()
        # capture for "base-model" which already has an entry
        with mock_patch("reasonbench.__main__.Pipeline") as MockPipeline:
            MockPipeline.return_value.run_prompts.return_value = fake
            main([
                "baseline", "capture",
                "--version", "v1",
                "--model", "base-model",
                "--judge", "judge-model",
            ])
        data = json.loads(
            (benchmark_dir / "v1" / "baselines.json").read_text(encoding="utf-8")
        )
        # still exactly one entry for base-model (not duplicated)
        assert sum(1 for m in data["models"] if m["model_name"] == "base-model") == 1

    def test_capture_missing_version_returns_1(self, patch_suite, capsys):
        code = main([
            "baseline", "capture",
            "--version", "v99",
            "--model", "m",
            "--judge", "j",
        ])
        assert code == 1
        assert "not found" in capsys.readouterr().err
