"""Scoring engine for evaluation results."""

from .taxonomy import Severity


class Scorer:
    """Computes composite failure scores, severity levels, and prompt hardness."""

    @staticmethod
    def compute_score(
        *,
        is_correct: bool,
        reasoning_flawed: bool,
        assumption_errors: int,
        counterfactual_fail: bool,
        model_disagreement: bool,
    ) -> int:
        """Compute composite failure score. Higher = more severe failure."""
        score = 0
        if not is_correct:
            score += 2
        if reasoning_flawed:
            score += 3
        score += assumption_errors
        if counterfactual_fail:
            score += 2
        if model_disagreement:
            score += 1
        return score

    @staticmethod
    def severity(score: int) -> Severity:
        """Map a composite score to a severity level."""
        if score >= 6:
            return Severity.CRITICAL
        if score >= 4:
            return Severity.HIGH
        if score >= 2:
            return Severity.MEDIUM
        return Severity.LOW

    @staticmethod
    def hardness(
        *,
        wrong_models: int,
        reasoning_failures: int,
        repair_failures: int,
    ) -> int:
        """Compute prompt hardness for ranking difficulty."""
        return (
            wrong_models * 2
            + reasoning_failures * 2
            + repair_failures * 3
        )
