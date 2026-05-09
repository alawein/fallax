"""Analytics engine for evaluation results."""

from __future__ import annotations

from collections import Counter, defaultdict

from .models import EvaluationResult


class Analyzer:
    """Computes metrics and extracts insights from evaluation results."""

    def __init__(self, results: list[EvaluationResult]) -> None:
        self._results = results

    def summary(self) -> dict:
        n = len(self._results)
        if n == 0:
            return {
                "total": 0,
                "severity_counts": {},
                "avg_score": 0.0,
                "failure_rate": 0.0,
            }
        return {
            "total": n,
            "severity_counts": dict(Counter(r.severity for r in self._results)),
            "avg_score": sum(r.score for r in self._results) / n,
            "failure_rate": sum(
                1 for r in self._results if r.validation.reasoning_flawed
            )
            / n,
        }

    def failure_rate_by_type(self) -> dict[str, float]:
        by_type: dict[str, list[bool]] = defaultdict(list)
        for r in self._results:
            by_type[r.failure_type.value].append(r.validation.reasoning_flawed)
        return {ft: sum(flags) / len(flags) for ft, flags in by_type.items()}

    def model_accuracy(self) -> dict[str, float]:
        by_model: dict[str, list[bool]] = defaultdict(list)
        for r in self._results:
            correct = r.validation.final_answer_correct or False
            for model_name in r.models:
                by_model[model_name].append(correct)
        return {model: sum(flags) / len(flags) for model, flags in by_model.items()}

    def top_failures(self, n: int = 10) -> list[EvaluationResult]:
        return sorted(self._results, key=lambda r: r.score, reverse=True)[:n]

    def hard_cases(self, min_score: int = 6) -> list[EvaluationResult]:
        return [r for r in self._results if r.score >= min_score]

    def disagreement_rate(self) -> float:
        if not self._results:
            return 0.0
        disagreements = sum(
            1
            for r in self._results
            if len({resp.answer for resp in r.models.values()}) > 1
        )
        return disagreements / len(self._results)

    def assumption_density(self) -> float:
        if not self._results:
            return 0.0
        total = sum(
            sum(1 for a in r.validation.assumptions if not a.justified)
            for r in self._results
        )
        return total / len(self._results)
