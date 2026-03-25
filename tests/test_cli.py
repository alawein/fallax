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
        mock = MockClient(
            responses=JUDGE_RESPONSES, default=MODEL_RESPONSE_TEXT
        )
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            code = main([
                "run",
                "--models", "m",
                "--judge", "j",
                "--count", "1",
                "--output", str(output),
                "--params-dir", str(params_dir),
                "--seed", "42",
            ])
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
        code = main([
            "train", str(results_file),
            "--output", str(model_path),
        ])
        assert code == 0
        assert model_path.exists()

    def test_train_missing_file_returns_error(self, tmp_path):
        code = main([
            "train", str(tmp_path / "nonexistent.jsonl"),
            "--output", str(tmp_path / "model.pkl"),
        ])
        assert code == 1


class TestNoSubcommand:
    def test_no_args_returns_one(self):
        code = main([])
        assert code == 1
