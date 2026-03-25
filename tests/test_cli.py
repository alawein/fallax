"""Tests for CLI argument parsing and main function."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reasonbench.__main__ import main
from tests.conftest import JUDGE_RESPONSES, MODEL_RESPONSE_TEXT, MockClient


@pytest.fixture()
def params_dir(tmp_path: Path) -> Path:
    data = [
        {"rule_a": "x > 0", "rule_b": "double", "edge_case_input": "x = -1"},
    ]
    (tmp_path / "implicit_assumption_trap.json").write_text(json.dumps(data))
    return tmp_path


class TestCLI:
    def test_main_returns_zero(self, params_dir, tmp_path):
        output = tmp_path / "out.jsonl"
        mock = MockClient(
            responses=JUDGE_RESPONSES, default=MODEL_RESPONSE_TEXT
        )
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            code = main([
                "--models", "m",
                "--judge", "j",
                "--count", "1",
                "--output", str(output),
                "--params-dir", str(params_dir),
                "--seed", "42",
            ])
        assert code == 0
        assert output.exists()

    def test_main_writes_results(self, params_dir, tmp_path):
        output = tmp_path / "out.jsonl"
        mock = MockClient(
            responses=JUDGE_RESPONSES, default=MODEL_RESPONSE_TEXT
        )
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            main([
                "--models", "m",
                "--judge", "j",
                "--count", "2",
                "--output", str(output),
                "--params-dir", str(params_dir),
            ])
        lines = [l for l in output.read_text().strip().split("\n") if l]
        assert len(lines) == 2

    def test_missing_required_args_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0
