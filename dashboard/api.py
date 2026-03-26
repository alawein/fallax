"""FastAPI backend for the ReasonBench dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from reasonbench.models import EvaluationResult

STATIC_DIR = Path(__file__).parent / "static"


def create_app(data_dir: Path | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        data_dir: Parent directory containing experiment output directories.
                  Each subdirectory with a report.json is treated as an experiment.
    """
    app = FastAPI(title="ReasonBench Dashboard", version="1.0.0")
    resolved_dir = data_dir or Path.cwd()

    app.state.data_dir = resolved_dir

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(404, "Dashboard frontend not found")
        return FileResponse(str(index_path))

    @app.get("/api/experiments")
    def list_experiments() -> list[dict[str, Any]]:
        """List all experiment directories that contain a report.json."""
        base: Path = app.state.data_dir
        if not base.exists():
            return []
        experiments = []
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            report_path = d / "report.json"
            if not report_path.exists():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rounds_files = sorted(d.glob("round_*.jsonl"))
            experiments.append(
                {
                    "name": d.name,
                    "total_rounds": report.get("total_rounds", 0),
                    "total_prompts": report.get("total_prompts", 0),
                    "total_failures": report.get("total_failures", 0),
                    "rounds_available": len(rounds_files),
                }
            )
        return experiments

    @app.get("/api/experiments/{name}/report")
    def get_report(name: str) -> dict[str, Any]:
        """Get the full report for an experiment."""
        report_path = _experiment_dir(name) / "report.json"
        if not report_path.exists():
            raise HTTPException(404, f"Report not found for {name}")
        return json.loads(report_path.read_text(encoding="utf-8"))

    @app.get("/api/experiments/{name}/results")
    def get_results(
        name: str,
        round_num: int | None = None,
        min_score: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get evaluation results, optionally filtered by round and score."""
        exp_dir = _experiment_dir(name)

        if round_num is not None:
            files = [exp_dir / f"round_{round_num}.jsonl"]
        else:
            files = sorted(exp_dir.glob("round_*.jsonl"))

        results: list[dict[str, Any]] = []
        for f in files:
            if not f.exists():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if min_score is not None and record.get("score", 0) < min_score:
                    continue
                results.append(record)
        return results

    @app.get("/api/experiments/{name}/summary")
    def get_summary(name: str) -> dict[str, Any]:
        """Get aggregated summary statistics across all rounds."""
        exp_dir = _experiment_dir(name)
        all_results = _load_all_results(exp_dir)

        if not all_results:
            return {
                "total": 0,
                "avg_score": 0.0,
                "failure_rate": 0.0,
                "by_category": {},
                "by_type": {},
                "by_severity": {},
                "score_distribution": {},
            }

        from reasonbench.taxonomy import get_category

        total = len(all_results)
        avg_score = sum(r.score for r in all_results) / total
        failures = sum(1 for r in all_results if r.score >= 4)

        by_category: dict[str, list[int]] = {}
        by_type: dict[str, list[int]] = {}
        by_severity: dict[str, int] = {}
        score_dist: dict[str, int] = {}

        for r in all_results:
            cat = get_category(r.failure_type).value
            by_category.setdefault(cat, []).append(r.score)
            by_type.setdefault(r.failure_type.value, []).append(r.score)
            sev = r.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
            bucket = str(r.score)
            score_dist[bucket] = score_dist.get(bucket, 0) + 1

        return {
            "total": total,
            "avg_score": avg_score,
            "failure_rate": failures / total,
            "by_category": {
                k: sum(v) / len(v) for k, v in sorted(by_category.items())
            },
            "by_type": {
                k: sum(v) / len(v) for k, v in sorted(by_type.items())
            },
            "by_severity": dict(sorted(by_severity.items())),
            "score_distribution": dict(sorted(score_dist.items())),
        }

    @app.get("/api/experiments/{name}/models")
    def get_model_comparison(name: str) -> list[dict[str, Any]]:
        """Get per-model accuracy comparison."""
        exp_dir = _experiment_dir(name)
        all_results = _load_all_results(exp_dir)

        model_stats: dict[str, dict[str, Any]] = {}
        for r in all_results:
            for model_name, resp in r.models.items():
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        "model": model_name,
                        "total": 0,
                        "correct": 0,
                    }
                model_stats[model_name]["total"] += 1
                if resp.is_correct:
                    model_stats[model_name]["correct"] += 1

        for stats in model_stats.values():
            stats["accuracy"] = (
                stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            )

        return sorted(model_stats.values(), key=lambda x: x["model"])

    def _experiment_dir(name: str) -> Path:
        base: Path = app.state.data_dir
        exp_dir = base / name
        if not exp_dir.is_dir():
            raise HTTPException(404, f"Experiment {name!r} not found")
        return exp_dir

    def _load_all_results(exp_dir: Path) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for f in sorted(exp_dir.glob("round_*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    results.append(EvaluationResult.model_validate_json(line))
        return results

    return app
