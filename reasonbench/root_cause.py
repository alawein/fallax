"""Root cause extraction — mines recurring failure patterns from results."""

from __future__ import annotations

from collections import defaultdict

from .models import EvaluationResult, RootCausePattern


class RootCauseExtractor:
    """Extracts recurring failure patterns from evaluation results.

    Groups unjustified assumptions by text (case-insensitive), counts
    frequency across results, and identifies which models and failure
    types are affected.
    """

    def __init__(self, results: list[EvaluationResult]) -> None:
        self._results = results

    def extract_patterns(
        self, min_frequency: int = 2
    ) -> list[RootCausePattern]:
        """Extract root cause patterns from failures.

        Only considers results with flawed reasoning. Groups unjustified
        assumptions by normalized text. Returns patterns sorted by
        frequency (descending).
        """
        # Group results by unjustified assumption text
        pattern_results: dict[str, list[EvaluationResult]] = defaultdict(
            list
        )
        for r in self._results:
            if not r.validation.reasoning_flawed:
                continue
            for assumption in r.validation.assumptions:
                if not assumption.justified:
                    key = assumption.text.lower()
                    pattern_results[key].append(r)

        # Build patterns
        patterns: list[RootCausePattern] = []
        for pattern_text, matching in pattern_results.items():
            if len(matching) < min_frequency:
                continue
            patterns.append(
                RootCausePattern(
                    pattern=pattern_text,
                    frequency=len(matching),
                    models_affected=sorted(
                        {
                            model
                            for r in matching
                            for model in r.models
                        }
                    ),
                    example_prompt=matching[0].prompt_text,
                    failure_types=sorted(
                        {r.failure_type.value for r in matching}
                    ),
                )
            )
        return sorted(patterns, key=lambda p: p.frequency, reverse=True)
