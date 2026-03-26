import pytest

from reasonbench.models import ExperimentRound, RepairResult, RootCausePattern
from reasonbench.report import ReportBuilder


@pytest.fixture()
def experiment_data():
    """Hand-crafted experiment data for report tests."""
    return {
        "rounds": [
            ExperimentRound(
                round_number=1,
                prompts_evaluated=10,
                avg_score=6.0,
                failure_rate=0.8,
                hard_case_count=8,
                evolved_count=5,
            ),
            ExperimentRound(
                round_number=2,
                prompts_evaluated=5,
                avg_score=7.5,
                failure_rate=1.0,
                hard_case_count=5,
                evolved_count=0,
            ),
        ],
        "root_cause_patterns": [
            RootCausePattern(
                pattern="assumes positivity",
                frequency=4,
                models_affected=["model-a", "model-b"],
                example_prompt="Is x positive?",
                failure_types=["contradiction"],
            ),
        ],
        "repair_results": [
            RepairResult(
                model_name="model-a",
                prompt_text="p",
                original_answer="a",
                repaired_answer="b",
                repair_reasoning="r",
                is_fixed=True,
            ),
            RepairResult(
                model_name="model-a",
                prompt_text="p2",
                original_answer="c",
                repaired_answer="d",
                repair_reasoning="r2",
                is_fixed=False,
            ),
        ],
        "total_prompts": 15,
        "total_failures": 12,
    }


class TestReportBuilder:
    def test_build_returns_dict(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        assert isinstance(report, dict)

    def test_build_has_required_keys(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        expected_keys = {
            "total_rounds",
            "total_prompts",
            "total_failures",
            "score_trend",
            "failure_trend",
            "score_delta",
            "failure_delta",
            "hardening_rate",
            "repair_success_rate",
            "top_patterns",
        }
        assert expected_keys <= set(report.keys())

    def test_score_trend(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        assert report["score_trend"] == [6.0, 7.5]

    def test_failure_trend(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        assert report["failure_trend"] == [0.8, 1.0]

    def test_score_delta(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        assert report["score_delta"] == pytest.approx(1.5)

    def test_repair_success_rate(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        report = builder.build()
        # 1 fixed out of 2
        assert report["repair_success_rate"] == pytest.approx(0.5)

    def test_top_patterns_limited_to_five(self):
        patterns = [
            RootCausePattern(
                pattern=f"pattern-{i}",
                frequency=10 - i,
                models_affected=["m"],
                example_prompt="p",
                failure_types=["contradiction"],
            )
            for i in range(8)
        ]
        data = {
            "rounds": [
                ExperimentRound(
                    round_number=1,
                    prompts_evaluated=10,
                    avg_score=5.0,
                    failure_rate=0.5,
                    hard_case_count=5,
                    evolved_count=0,
                ),
            ],
            "root_cause_patterns": patterns,
            "repair_results": [],
            "total_prompts": 10,
            "total_failures": 5,
        }
        builder = ReportBuilder(data)
        report = builder.build()
        assert len(report["top_patterns"]) == 5

    def test_to_markdown_contains_sections(self, experiment_data):
        builder = ReportBuilder(experiment_data)
        md = builder.to_markdown()
        assert "# ReasonBench Experiment Report" in md
        assert "## Overview" in md
        assert "## Per-Round Results" in md
        assert "## Trends" in md
        assert "## Top Root Cause Patterns" in md
        assert "assumes positivity" in md

    def test_empty_rounds(self):
        data = {
            "rounds": [],
            "root_cause_patterns": [],
            "repair_results": [],
            "total_prompts": 0,
            "total_failures": 0,
        }
        builder = ReportBuilder(data)
        report = builder.build()
        assert report["total_rounds"] == 0
        assert report["score_trend"] == []
        assert report["score_delta"] == 0.0

    def test_no_repairs_gives_none_rate(self):
        data = {
            "rounds": [
                ExperimentRound(
                    round_number=1,
                    prompts_evaluated=5,
                    avg_score=3.0,
                    failure_rate=0.4,
                    hard_case_count=2,
                    evolved_count=0,
                ),
            ],
            "root_cause_patterns": [],
            "repair_results": [],
            "total_prompts": 5,
            "total_failures": 2,
        }
        builder = ReportBuilder(data)
        report = builder.build()
        assert report["repair_success_rate"] is None
