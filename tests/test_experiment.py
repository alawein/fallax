import json

import pytest

from reasonbench.experiment import Experiment
from reasonbench.models import ExperimentRound, RepairResult
from tests.conftest import JUDGE_RESPONSES, MODEL_RESPONSE_TEXT, MockClient


@pytest.fixture()
def experiment_client():
    """MockClient handling model, judge, evolver, and repair responses."""
    responses = {
        **JUDGE_RESPONSES,
        "Given this prompt and its failure": "An evolved harder prompt text",
        "Your previous answer was incorrect": "I was wrong. The fixed answer is Z.",
    }
    return MockClient(responses=responses, default=MODEL_RESPONSE_TEXT)


@pytest.fixture()
def params_dir(tmp_path):
    """Parameter bank for round 1 prompt generation."""
    data = [
        {"rule_a": "x > 0", "rule_b": "double", "edge_case_input": "x = -1"},
        {"rule_a": "y < 5", "rule_b": "triple", "edge_case_input": "y = 10"},
    ]
    (tmp_path / "implicit_assumption_trap.json").write_text(json.dumps(data))
    return tmp_path


class TestExperiment:
    def test_run_returns_dict_with_required_keys(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        assert "rounds" in result
        assert "root_cause_patterns" in result
        assert "repair_results" in result
        assert "total_prompts" in result
        assert "total_failures" in result

    def test_run_creates_round_files(self, experiment_client, params_dir, tmp_path):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        exp.run(initial_count=2, rounds=2, min_score=6)

        assert (output_dir / "round_1.jsonl").exists()
        assert (output_dir / "round_2.jsonl").exists()

    def test_run_returns_correct_round_count(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=3, min_score=6)

        assert len(result["rounds"]) == 3

    def test_rounds_are_experiment_round_objects(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        assert all(isinstance(r, ExperimentRound) for r in result["rounds"])
        assert result["rounds"][0].round_number == 1
        assert result["rounds"][1].round_number == 2

    def test_round_metadata_populated(self, experiment_client, params_dir, tmp_path):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        r1 = result["rounds"][0]
        assert r1.prompts_evaluated > 0
        assert r1.avg_score > 0
        assert r1.hard_case_count > 0

    def test_total_prompts_sums_rounds(self, experiment_client, params_dir, tmp_path):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        expected = sum(r.prompts_evaluated for r in result["rounds"])
        assert result["total_prompts"] == expected

    def test_root_cause_patterns_returned(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        assert isinstance(result["root_cause_patterns"], list)

    def test_repair_results_returned(self, experiment_client, params_dir, tmp_path):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        assert isinstance(result["repair_results"], list)
        assert all(isinstance(r, RepairResult) for r in result["repair_results"])

    def test_early_termination_when_no_hard_cases(
        self, experiment_client, params_dir, tmp_path
    ):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        # min_score=100 means nothing qualifies as hard → no evolution → round 2 has no prompts
        result = exp.run(initial_count=2, rounds=3, min_score=100)

        assert len(result["rounds"]) == 1

    def test_last_round_does_not_evolve(self, experiment_client, params_dir, tmp_path):
        output_dir = tmp_path / "experiment"
        exp = Experiment(
            client=experiment_client,
            models=["m"],
            judge_model="j",
            output_dir=output_dir,
            evolve_model="e",
            params_dir=params_dir,
        )
        result = exp.run(initial_count=2, rounds=2, min_score=6)

        last_round = result["rounds"][-1]
        assert last_round.evolved_count == 0
