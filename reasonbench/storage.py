"""JSONL storage for evaluation results."""

from __future__ import annotations

from pathlib import Path

from .models import EvaluationResult


class JsonlStore:
    """Append-only JSONL storage for evaluation results."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, result: EvaluationResult) -> None:
        """Append a single result to the JSONL file."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

    def read_all(self) -> list[EvaluationResult]:
        """Read all results from the JSONL file."""
        if not self.path.exists():
            return []
        results: list[EvaluationResult] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(
                        EvaluationResult.model_validate_json(line)
                    )
        return results

    def count(self) -> int:
        """Count results in the file."""
        if not self.path.exists():
            return 0
        total = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
        return total

    def read_by_min_score(self, min_score: int) -> list[EvaluationResult]:
        """Read results with score >= min_score."""
        return [r for r in self.read_all() if r.score >= min_score]
