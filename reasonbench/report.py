"""Report builder for experiment results."""

from __future__ import annotations

from .models import ExperimentRound, RootCausePattern


class ReportBuilder:
    """Builds structured reports from experiment data.

    Takes the dict returned by Experiment.run() and produces
    a summary dict (build()) or markdown string (to_markdown()).
    """

    def __init__(self, experiment_data: dict) -> None:
        self._data = experiment_data

    def build(self) -> dict:
        """Build a structured report summary."""
        rounds: list[ExperimentRound] = self._data["rounds"]

        if not rounds:
            return {
                "total_rounds": 0,
                "total_prompts": 0,
                "total_failures": 0,
                "score_trend": [],
                "failure_trend": [],
                "score_delta": 0.0,
                "failure_delta": 0.0,
                "hardening_rate": 0.0,
                "repair_success_rate": None,
                "top_patterns": [],
            }

        score_trend = [r.avg_score for r in rounds]
        failure_trend = [r.failure_rate for r in rounds]

        hardening_rate = 0.0
        if len(rounds) >= 2:
            total_evolved = sum(r.evolved_count for r in rounds[:-1])
            total_stayed_hard = sum(r.hard_case_count for r in rounds[1:])
            if total_evolved > 0:
                hardening_rate = total_stayed_hard / total_evolved

        repairs = self._data.get("repair_results", [])
        repair_rate = None
        if repairs:
            fixed = sum(1 for r in repairs if r.is_fixed is True)
            repair_rate = fixed / len(repairs)

        patterns: list[RootCausePattern] = self._data.get("root_cause_patterns", [])
        top_patterns = [
            {
                "pattern": p.pattern,
                "frequency": p.frequency,
                "models_affected": p.models_affected,
                "failure_types": p.failure_types,
            }
            for p in patterns[:5]
        ]

        return {
            "total_rounds": len(rounds),
            "total_prompts": self._data["total_prompts"],
            "total_failures": self._data["total_failures"],
            "score_trend": score_trend,
            "failure_trend": failure_trend,
            "score_delta": score_trend[-1] - score_trend[0],
            "failure_delta": failure_trend[-1] - failure_trend[0],
            "hardening_rate": hardening_rate,
            "repair_success_rate": repair_rate,
            "top_patterns": top_patterns,
        }

    def to_markdown(self) -> str:
        """Render the report as a markdown string."""
        report = self.build()
        rounds: list[ExperimentRound] = self._data["rounds"]

        lines = [
            "# ReasonBench Experiment Report",
            "",
            "## Overview",
            "",
            f"- **Rounds:** {report['total_rounds']}",
            f"- **Total prompts:** {report['total_prompts']}",
            f"- **Total failures:** {report['total_failures']}",
            "",
            "## Per-Round Results",
            "",
            "| Round | Prompts | Avg Score | Failure Rate | Evolved |",
            "|-------|---------|-----------|--------------|---------|",
        ]

        for r in rounds:
            lines.append(
                f"| {r.round_number} | {r.prompts_evaluated} | "
                f"{r.avg_score:.2f} | {r.failure_rate:.1%} | "
                f"{r.evolved_count} |"
            )

        total_rounds = report["total_rounds"]
        lines.extend(
            [
                "",
                "## Trends",
                "",
                f"- **Score delta:** {report['score_delta']:+.2f} "
                f"(round 1 to round {total_rounds})",
                f"- **Failure rate delta:** {report['failure_delta']:+.1%}",
                f"- **Hardening rate:** {report['hardening_rate']:.1%}",
            ]
        )

        if report["repair_success_rate"] is not None:
            lines.append(
                f"- **Repair success rate:** {report['repair_success_rate']:.1%}"
            )

        if report["top_patterns"]:
            lines.extend(
                [
                    "",
                    "## Top Root Cause Patterns",
                    "",
                ]
            )
            for p in report["top_patterns"]:
                models = ", ".join(p["models_affected"])
                lines.append(
                    f"- **{p['pattern']}** "
                    f"(frequency: {p['frequency']}, models: {models})"
                )

        lines.append("")
        return "\n".join(lines)
