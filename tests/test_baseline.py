"""Tests for baseline subcommands."""

from __future__ import annotations

import functools
import json
import re

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

    def test_status_no_baselines_file(self, patch_suite, benchmark_dir, capsys):
        """First-run case: baselines.json doesn't exist yet. Must not crash."""
        (benchmark_dir / "v1" / "baselines.json").unlink()
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

    def test_capture_writes_entry(
        self, patch_suite, benchmark_dir, monkeypatch, capsys
    ):
        from unittest.mock import patch as mock_patch

        fake = self._fake_results()
        with (
            mock_patch("reasonbench.__main__._make_client"),
            mock_patch("reasonbench.__main__.Pipeline") as MockPipeline,
        ):
            MockPipeline.return_value.run_prompts.return_value = fake
            code = main(
                [
                    "baseline",
                    "capture",
                    "--version",
                    "v1",
                    "--model",
                    "new-model",
                    "--judge",
                    "judge-model",
                ]
            )
        assert code == 0
        # reload baselines from disk
        baselines_path = benchmark_dir / "v1" / "baselines.json"
        data = json.loads(baselines_path.read_text(encoding="utf-8"))
        names = [m["model_name"] for m in data["models"]]
        assert "new-model" in names
        # captured_at must be a valid ISO-8601 UTC timestamp, not empty/local
        entry = next(m for m in data["models"] if m["model_name"] == "new-model")
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$",
            entry["captured_at"],
        ), f"captured_at not ISO-8601 UTC: {entry['captured_at']!r}"

    def test_capture_replaces_existing_entry(
        self, patch_suite, benchmark_dir, monkeypatch
    ):
        from unittest.mock import patch as mock_patch

        fake = self._fake_results()
        # capture for "base-model" which already has an entry
        with (
            mock_patch("reasonbench.__main__._make_client"),
            mock_patch("reasonbench.__main__.Pipeline") as MockPipeline,
        ):
            MockPipeline.return_value.run_prompts.return_value = fake
            main(
                [
                    "baseline",
                    "capture",
                    "--version",
                    "v1",
                    "--model",
                    "base-model",
                    "--judge",
                    "judge-model",
                ]
            )
        data = json.loads(
            (benchmark_dir / "v1" / "baselines.json").read_text(encoding="utf-8")
        )
        # still exactly one entry for base-model (not duplicated)
        assert sum(1 for m in data["models"] if m["model_name"] == "base-model") == 1
        # the surviving entry is the NEW capture, not the frozen fixture one —
        # guards against a "dedupe keeps old" bug
        base = next(m for m in data["models"] if m["model_name"] == "base-model")
        assert base["captured_at"] != "2026-04-18T10:00:00+00:00"

    def test_capture_missing_version_returns_1(self, patch_suite, capsys):
        code = main(
            [
                "baseline",
                "capture",
                "--version",
                "v99",
                "--model",
                "m",
                "--judge",
                "j",
            ]
        )
        assert code == 1
        assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# baseline compare
# ---------------------------------------------------------------------------


class TestBaselineCompare:
    def _fake_results_with_score(self, avg_score: float):
        """Return results whose average score equals avg_score."""
        from tests.conftest import make_result

        score_int = round(avg_score)
        return [make_result(score=score_int), make_result(score=score_int)]

    def test_compare_exits_zero_within_threshold(self, patch_suite, monkeypatch):
        """4.0 baseline, 3.9 current → round to 4 → delta 0.0, threshold 0.5 → exit 0."""
        from unittest.mock import patch as mock_patch

        fake = self._fake_results_with_score(3.9)
        with (
            mock_patch("reasonbench.__main__._make_client"),
            mock_patch("reasonbench.__main__.Pipeline") as MockPipeline,
        ):
            MockPipeline.return_value.run_prompts.return_value = fake
            code = main(
                [
                    "baseline",
                    "compare",
                    "--version",
                    "v1",
                    "--model",
                    "base-model",
                    "--judge",
                    "judge-model",
                    "--threshold",
                    "0.5",
                ]
            )
        assert code == 0

    def test_compare_exits_two_on_regression(self, patch_suite, monkeypatch, capsys):
        """4.0 baseline, 1.0 current → delta -3.0, threshold 0.05 → exit 2."""
        from unittest.mock import patch as mock_patch

        fake = self._fake_results_with_score(1.0)
        with (
            mock_patch("reasonbench.__main__._make_client"),
            mock_patch("reasonbench.__main__.Pipeline") as MockPipeline,
        ):
            MockPipeline.return_value.run_prompts.return_value = fake
            code = main(
                [
                    "baseline",
                    "compare",
                    "--version",
                    "v1",
                    "--model",
                    "base-model",
                    "--judge",
                    "judge-model",
                    "--threshold",
                    "0.05",
                ]
            )
        assert code == 2
        assert "REGRESSION" in capsys.readouterr().out

    def test_compare_exits_one_on_missing_baseline(self, patch_suite, capsys):
        """No baseline for 'unknown-model' → exit 1."""
        code = main(
            [
                "baseline",
                "compare",
                "--version",
                "v1",
                "--model",
                "unknown-model",
                "--judge",
                "judge-model",
            ]
        )
        assert code == 1
        assert "no baseline" in capsys.readouterr().err.lower()

    def test_compare_exits_one_on_missing_version(self, patch_suite, capsys):
        code = main(
            [
                "baseline",
                "compare",
                "--version",
                "v99",
                "--model",
                "base-model",
                "--judge",
                "judge-model",
            ]
        )
        assert code == 1
        assert "not found" in capsys.readouterr().err

    def test_compare_exits_one_on_pipeline_error(self, patch_suite, capsys):
        """Unhandled pipeline/client errors map to exit 1, not a traceback."""
        from unittest.mock import patch as mock_patch

        with (
            mock_patch("reasonbench.__main__._make_client"),
            mock_patch("reasonbench.__main__.Pipeline") as MockPipeline,
        ):
            MockPipeline.return_value.run_prompts.side_effect = RuntimeError(
                "anthropic quota exhausted"
            )
            code = main(
                [
                    "baseline",
                    "compare",
                    "--version",
                    "v1",
                    "--model",
                    "base-model",
                    "--judge",
                    "judge-model",
                ]
            )
        assert code == 1
        err = capsys.readouterr().err
        assert "benchmark run failed" in err
        assert "anthropic quota exhausted" in err


# ---------------------------------------------------------------------------
# BenchmarkSuite.load_baselines error handling
# ---------------------------------------------------------------------------


class TestLoadBaselinesErrors:
    def test_malformed_json_raises_with_path(self, benchmark_dir):
        (benchmark_dir / "v1" / "baselines.json").write_text(
            "{not valid json", encoding="utf-8"
        )
        suite = BenchmarkSuite(benchmarks_dir=benchmark_dir)
        with pytest.raises(ValueError) as exc:
            suite.load_baselines("v1")
        assert "baselines.json" in str(exc.value)
        assert "malformed" in str(exc.value)
