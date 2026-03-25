"""CLI entry point: python -m reasonbench."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .client import AnthropicClient
from .pipeline import Pipeline


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the evaluation pipeline."""
    client = AnthropicClient()
    pipeline = Pipeline(
        client=client,
        models=args.models,
        judge_model=args.judge,
        output_path=Path(args.output),
        params_dir=Path(args.params_dir) if args.params_dir else None,
        seed=args.seed,
    )
    results = pipeline.run(count=args.count)

    critical = sum(1 for r in results if r.severity.value == "critical")
    high = sum(1 for r in results if r.severity.value == "high")
    medium = sum(1 for r in results if r.severity.value == "medium")
    low = sum(1 for r in results if r.severity.value == "low")

    print(f"\nReasonBench Evaluation Complete")
    print(f"  Prompts evaluated: {len(results)}")
    print(f"  Critical: {critical}")
    print(f"  High:     {high}")
    print(f"  Medium:   {medium}")
    print(f"  Low:      {low}")
    print(f"  Output:   {args.output}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze evaluation results."""
    from .analyzer import Analyzer
    from .storage import JsonlStore

    path = Path(args.results)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    store = JsonlStore(path)
    results = store.read_all()
    if not results:
        print("No results found.")
        return 0

    analyzer = Analyzer(results)
    summary = analyzer.summary()

    print(f"\nReasonBench Analysis ({summary['total']} results)")
    print(f"  Avg score:     {summary['avg_score']:.2f}")
    print(f"  Failure rate:  {summary['failure_rate']:.1%}")
    print(f"  Disagreement:  {analyzer.disagreement_rate():.1%}")
    print(f"  Assumption density: {analyzer.assumption_density():.2f}")

    print(f"\nSeverity distribution:")
    for sev, count in sorted(
        summary["severity_counts"].items(), key=lambda x: x[0].value
    ):
        print(f"  {sev.value:>8}: {count}")

    print(f"\nFailure rate by type:")
    for ft, rate in sorted(
        analyzer.failure_rate_by_type().items(), key=lambda x: -x[1]
    ):
        print(f"  {ft:>30}: {rate:.1%}")

    print(f"\nModel accuracy:")
    for model, acc in analyzer.model_accuracy().items():
        print(f"  {model}: {acc:.1%}")

    top = analyzer.top_failures(n=args.top)
    if top:
        print(f"\nTop {len(top)} failures:")
        for r in top:
            print(
                f"  [{r.severity.value:>8}] score={r.score} "
                f"type={r.failure_type.value} id={r.prompt_id}"
            )

    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train failure predictor on results."""
    from .predictor import FailurePredictor
    from .storage import JsonlStore

    path = Path(args.results)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    store = JsonlStore(path)
    results = store.read_all()
    if not results:
        print("No results to train on.")
        return 1

    predictor = FailurePredictor()
    metrics = predictor.train(results, threshold=args.threshold)

    output = Path(args.output)
    predictor.save(output)

    print(f"\nPredictor trained")
    print(f"  Samples:     {metrics['samples']}")
    print(f"  Positive rate: {metrics['positive_rate']:.1%}")
    print(f"  CV accuracy:   {metrics['cv_accuracy']:.1%}")
    print(f"  Saved to:    {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """ReasonBench CLI entry point."""
    default_model = os.environ.get("REASONBENCH_MODEL", "")
    default_judge = os.environ.get("REASONBENCH_JUDGE_MODEL", "")

    parser = argparse.ArgumentParser(
        prog="reasonbench",
        description="LLM Adversarial Reasoning Evaluation System",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- run --
    run_p = subparsers.add_parser("run", help="Run evaluation pipeline")
    run_p.add_argument(
        "--models", nargs="+",
        required=not bool(default_model),
        default=[default_model] if default_model else None,
        help="Models to evaluate (or set REASONBENCH_MODEL env var)",
    )
    run_p.add_argument(
        "--judge",
        required=not bool(default_judge),
        default=default_judge or None,
        help="Judge model (or set REASONBENCH_JUDGE_MODEL env var)",
    )
    run_p.add_argument("--count", type=int, default=10)
    run_p.add_argument("--output", default="results.jsonl")
    run_p.add_argument("--params-dir", default=None)
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--verbose", "-v", action="store_true")

    # -- analyze --
    analyze_p = subparsers.add_parser("analyze", help="Analyze evaluation results")
    analyze_p.add_argument("results", help="Path to results JSONL file")
    analyze_p.add_argument("--top", type=int, default=10, help="Number of top failures to show")

    # -- train --
    train_p = subparsers.add_parser("train", help="Train failure predictor")
    train_p.add_argument("results", help="Path to results JSONL file")
    train_p.add_argument("--output", "-o", default="predictor.pkl", help="Output model file")
    train_p.add_argument("--threshold", type=int, default=4, help="Score threshold for failure label")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "train":
        return _cmd_train(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
