"""Tests for CLI subcommands."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reasonbench.__main__ import main
from tests.conftest import (
    JUDGE_RESPONSES,
    MODEL_RESPONSE_TEXT,
    MockClient,
    make_result,
)


@pytest.fixture()
def params_dir(tmp_path: Path) -> Path:
    data = [
        {"rule_a": "x > 0", "rule_b": "double", "edge_case_input": "x = -1"},
    ]
    (tmp_path / "implicit_assumption_trap.json").write_text(json.dumps(data))
    return tmp_path


@pytest.fixture()
def results_file(tmp_path: Path) -> Path:
    """Write mock results to a JSONL file."""
    path = tmp_path / "results.jsonl"
    results = [
        make_result(prompt_id=f"r{i}", score=i * 2, reasoning_flawed=i > 2)
        for i in range(6)
    ]
    with open(path, "w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    return path


class TestRunSubcommand:
    def test_run_returns_zero(self, params_dir, tmp_path):
        output = tmp_path / "out.jsonl"
        mock = MockClient(responses=JUDGE_RESPONSES, default=MODEL_RESPONSE_TEXT)
        with patch("reasonbench.__main__._make_client", return_value=mock):
            code = main(
                [
                    "run",
                    "--models",
                    "m",
                    "--judge",
                    "j",
                    "--count",
                    "1",
                    "--output",
                    str(output),
                    "--params-dir",
                    str(params_dir),
                    "--seed",
                    "42",
                ]
            )
        assert code == 0
        assert output.exists()


class TestAnalyzeSubcommand:
    def test_analyze_returns_zero(self, results_file):
        code = main(["analyze", str(results_file)])
        assert code == 0

    def test_analyze_with_top_flag(self, results_file):
        code = main(["analyze", str(results_file), "--top", "3"])
        assert code == 0

    def test_analyze_missing_file_returns_error(self, tmp_path):
        code = main(["analyze", str(tmp_path / "nonexistent.jsonl")])
        assert code == 1


class TestTrainSubcommand:
    def test_train_returns_zero(self, results_file, tmp_path):
        model_path = tmp_path / "predictor.pkl"
        code = main(
            [
                "train",
                str(results_file),
                "--output",
                str(model_path),
            ]
        )
        assert code == 0
        assert model_path.exists()

    def test_train_missing_file_returns_error(self, tmp_path):
        code = main(
            [
                "train",
                str(tmp_path / "nonexistent.jsonl"),
                "--output",
                str(tmp_path / "model.pkl"),
            ]
        )
        assert code == 1


class TestEvolveSubcommand:
    def test_evolve_returns_zero(self, results_file, tmp_path):
        output = tmp_path / "evolved.jsonl"
        mock = MockClient(default="A harder evolved prompt.")
        with patch("reasonbench.__main__._make_client", return_value=mock):
            code = main(
                [
                    "evolve",
                    str(results_file),
                    "--model",
                    "m",
                    "--output",
                    str(output),
                ]
            )
        assert code == 0

    def test_evolve_missing_file_returns_error(self, tmp_path):
        mock = MockClient(default="evolved")
        with patch("reasonbench.__main__._make_client", return_value=mock):
            code = main(
                [
                    "evolve",
                    str(tmp_path / "nonexistent.jsonl"),
                    "--model",
                    "m",
                ]
            )
        assert code == 1


class TestRepairSubcommand:
    def test_repair_returns_zero(self, results_file, tmp_path):
        output = tmp_path / "repairs.jsonl"
        mock = MockClient(default="I was wrong. The correct answer is X.")
        with patch("reasonbench.__main__._make_client", return_value=mock):
            code = main(
                [
                    "repair",
                    str(results_file),
                    "--model",
                    "m",
                    "--output",
                    str(output),
                ]
            )
        assert code == 0

    def test_repair_missing_file_returns_error(self, tmp_path):
        mock = MockClient(default="fixed")
        with patch("reasonbench.__main__._make_client", return_value=mock):
            code = main(
                [
                    "repair",
                    str(tmp_path / "nonexistent.jsonl"),
                    "--model",
                    "m",
                ]
            )
        assert code == 1


class TestExperimentSubcommand:
    def test_experiment_returns_zero(self, params_dir, tmp_path):
        output_dir = tmp_path / "experiment"
        mock = MockClient(
            responses={
                **JUDGE_RESPONSES,
                "Given this prompt and its failure": "Evolved prompt",
                "Your previous answer was incorrect": "Fixed answer.",
            },
            default=MODEL_RESPONSE_TEXT,
        )
        with patch("reasonbench.__main__._make_client", return_value=mock):
            code = main(
                [
                    "experiment",
                    "--models",
                    "m",
                    "--judge",
                    "j",
                    "--evolve-model",
                    "e",
                    "--rounds",
                    "2",
                    "--count",
                    "1",
                    "--output-dir",
                    str(output_dir),
                    "--params-dir",
                    str(params_dir),
                ]
            )
        assert code == 0
        assert (output_dir / "round_1.jsonl").exists()
        assert (output_dir / "report.json").exists()
        assert (output_dir / "report.md").exists()

    def test_experiment_with_seed(self, params_dir, tmp_path):
        output_dir = tmp_path / "experiment"
        mock = MockClient(
            responses={
                **JUDGE_RESPONSES,
                "Given this prompt and its failure": "Evolved",
                "Your previous answer was incorrect": "Fixed.",
            },
            default=MODEL_RESPONSE_TEXT,
        )
        with patch("reasonbench.__main__._make_client", return_value=mock):
            code = main(
                [
                    "experiment",
                    "--models",
                    "m",
                    "--judge",
                    "j",
                    "--evolve-model",
                    "e",
                    "--rounds",
                    "1",
                    "--count",
                    "1",
                    "--output-dir",
                    str(output_dir),
                    "--params-dir",
                    str(params_dir),
                    "--seed",
                    "42",
                ]
            )
        assert code == 0


class TestBenchmarkSubcommand:
    @pytest.fixture()
    def bench_dir(self, tmp_path):
        """Create a minimal benchmark v1 in a temp directory."""
        v1 = tmp_path / "benchmarks" / "v1"
        v1.mkdir(parents=True)
        prompt = {
            "prompt_id": "b1",
            "failure_type": "unstated_assumption",
            "prompt_text": "Test prompt",
            "difficulty": 5,
        }
        (v1 / "prompts.jsonl").write_text(json.dumps(prompt) + "\n")
        meta = {
            "version": "v1",
            "created": "2026-01-01",
            "description": "Test benchmark",
            "prompt_count": 1,
            "failure_types": ["unstated_assumption"],
            "categories": ["assumption_error"],
        }
        (v1 / "metadata.json").write_text(json.dumps(meta))
        (v1 / "baselines.json").write_text(json.dumps({"version": "v1", "models": []}))
        return tmp_path / "benchmarks"

    def test_benchmark_list(self, bench_dir):
        with patch("reasonbench.benchmark.BenchmarkSuite.__init__", return_value=None), patch("reasonbench.benchmark.BenchmarkSuite.versions", return_value=["v1"]), patch("reasonbench.benchmark.BenchmarkSuite.load_metadata") as mock_meta:
            from reasonbench.benchmark import BenchmarkMetadata

            mock_meta.return_value = BenchmarkMetadata(
                version="v1",
                created="2026-01-01",
                description="Test",
                prompt_count=1,
                failure_types=[],
                categories=[],
            )
            code = main(["benchmark", "--list"])
        assert code == 0

    def test_benchmark_info_no_models(self, bench_dir):
        from reasonbench.benchmark import BenchmarkBaselines, BenchmarkMetadata

        with patch("reasonbench.benchmark.BenchmarkSuite.__init__", return_value=None), patch("reasonbench.benchmark.BenchmarkSuite.load_prompts", return_value=[]), patch("reasonbench.benchmark.BenchmarkSuite.load_metadata") as mock_meta, patch("reasonbench.benchmark.BenchmarkSuite.load_baselines") as mock_base:
            mock_meta.return_value = BenchmarkMetadata(
                version="v1",
                created="2026-01-01",
                description="Test",
                prompt_count=1,
                failure_types=[],
                categories=[],
            )
            mock_base.return_value = BenchmarkBaselines(version="v1", models=[])
            code = main(["benchmark", "--version", "v1"])
        assert code == 0

    def test_benchmark_not_found(self):
        with patch("reasonbench.benchmark.BenchmarkSuite.__init__", return_value=None), patch("reasonbench.benchmark.BenchmarkSuite.load_prompts", side_effect=FileNotFoundError("not found")):
            code = main(["benchmark", "--version", "v99"])
        assert code == 1


class TestNoSubcommand:
    def test_no_args_returns_one(self):
        code = main([])
        assert code == 1
