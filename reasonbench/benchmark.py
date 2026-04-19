"""Versioned benchmark suite for reproducible cross-model comparison."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import EvaluationResult, Prompt
from .taxonomy import get_category


class BenchmarkMetadata(BaseModel):
    """Metadata for a benchmark version."""

    version: str
    created: str
    description: str
    prompt_count: int = Field(ge=0)
    failure_types: list[str]
    categories: list[str]
    generation_params: dict[str, Any] = Field(default_factory=dict)


class ModelBaseline(BaseModel):
    """Baseline scores for a single model on a benchmark."""

    model_name: str
    overall_score: float
    failure_rate: float
    category_scores: dict[str, float] = Field(default_factory=dict)
    type_scores: dict[str, float] = Field(default_factory=dict)
    assumption_density: float = 0.0
    runs: int = Field(ge=1, default=1)
    captured_at: str = ""


class BenchmarkBaselines(BaseModel):
    """Collection of model baselines for a benchmark version."""

    version: str
    models: list[ModelBaseline] = Field(default_factory=list)


class BenchmarkSuite:
    """Manages versioned benchmark prompt sets and baselines."""

    def __init__(self, benchmarks_dir: Path | None = None) -> None:
        self._dir = benchmarks_dir or Path(__file__).parent.parent / "benchmarks"

    def versions(self) -> list[str]:
        """List available benchmark versions."""
        if not self._dir.exists():
            return []
        return sorted(
            d.name
            for d in self._dir.iterdir()
            if d.is_dir() and (d / "prompts.jsonl").exists()
        )

    def _version_dir(self, version: str) -> Path:
        return self._dir / version

    def load_prompts(self, version: str) -> list[Prompt]:
        """Load the fixed prompt set for a benchmark version."""
        path = self._version_dir(version) / "prompts.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Benchmark {version} not found: {path}")
        prompts: list[Prompt] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompts.append(Prompt.model_validate_json(line))
        return prompts

    def load_metadata(self, version: str) -> BenchmarkMetadata:
        """Load metadata for a benchmark version."""
        path = self._version_dir(version) / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Metadata not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return BenchmarkMetadata.model_validate(data)

    def load_baselines(self, version: str) -> BenchmarkBaselines:
        """Load baseline scores for a benchmark version."""
        path = self._version_dir(version) / "baselines.json"
        if not path.exists():
            return BenchmarkBaselines(version=version)
        data = json.loads(path.read_text(encoding="utf-8"))
        return BenchmarkBaselines.model_validate(data)

    def save_baselines(self, baselines: BenchmarkBaselines) -> Path:
        """Save baseline scores."""
        path = self._version_dir(baselines.version) / "baselines.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            baselines.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return path

    def create_version(
        self,
        version: str,
        prompts: list[Prompt],
        description: str = "",
        generation_params: dict[str, Any] | None = None,
    ) -> Path:
        """Create a new benchmark version from a list of prompts."""
        vdir = self._version_dir(version)
        if vdir.exists() and (vdir / "prompts.jsonl").exists():
            raise FileExistsError(f"Benchmark {version} already exists")

        vdir.mkdir(parents=True, exist_ok=True)

        prompts_path = vdir / "prompts.jsonl"
        with open(prompts_path, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(p.model_dump_json() + "\n")

        failure_types = sorted({p.failure_type.value for p in prompts})
        categories = sorted({get_category(p.failure_type).value for p in prompts})

        metadata = BenchmarkMetadata(
            version=version,
            created=datetime.now(UTC).isoformat(),
            description=description or f"Benchmark {version}",
            prompt_count=len(prompts),
            failure_types=failure_types,
            categories=categories,
            generation_params=generation_params or {},
        )
        meta_path = vdir / "metadata.json"
        meta_path.write_text(
            metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return vdir

    def score_results(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """Compute benchmark scores from evaluation results."""
        if not results:
            return {
                "overall_score": 0.0,
                "failure_rate": 0.0,
                "category_scores": {},
                "type_scores": {},
                "total": 0,
            }

        total = len(results)
        avg_score = sum(r.score for r in results) / total
        failures = sum(1 for r in results if r.score >= 4)
        failure_rate = failures / total

        by_category: dict[str, list[int]] = {}
        by_type: dict[str, list[int]] = {}

        for r in results:
            cat = get_category(r.failure_type).value
            by_category.setdefault(cat, []).append(r.score)
            by_type.setdefault(r.failure_type.value, []).append(r.score)

        category_scores = {
            cat: sum(scores) / len(scores)
            for cat, scores in sorted(by_category.items())
        }
        type_scores = {
            ft: sum(scores) / len(scores) for ft, scores in sorted(by_type.items())
        }

        return {
            "overall_score": avg_score,
            "failure_rate": failure_rate,
            "category_scores": category_scores,
            "type_scores": type_scores,
            "total": total,
        }
