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


def _cmd_evolve(args: argparse.Namespace) -> int:
    """Evolve hard-case prompts into harder versions."""
    from .evolver import PromptEvolver
    from .storage import JsonlStore

    path = Path(args.results)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    store = JsonlStore(path)
    results = store.read_all()
    if not results:
        print("No results found.")
        return 1

    client = AnthropicClient()
    evolver = PromptEvolver(client=client, model=args.model)
    evolved = evolver.evolve_batch(results, min_score=args.min_score)

    if not evolved:
        print("No hard cases found to evolve.")
        return 0

    output = Path(args.output)
    with open(output, "w", encoding="utf-8") as f:
        for p in evolved:
            f.write(p.model_dump_json() + "\n")

    print(f"\nEvolved {len(evolved)} prompts")
    print(f"  Min score: {args.min_score}")
    print(f"  Saved to: {output}")
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    """Test model self-repair on failed results."""
    from .repair import SelfRepairTester
    from .storage import JsonlStore

    path = Path(args.results)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    store = JsonlStore(path)
    results = store.read_all()
    if not results:
        print("No results found.")
        return 1

    client = AnthropicClient()
    tester = SelfRepairTester(client=client)

    for r in results:
        for model_name in list(r.models.keys()):
            r.models[model_name].model_name = args.model
            r.models[args.model] = r.models.pop(model_name)
            break

    repairs = tester.test_repair_batch(results)

    if not repairs:
        print("No failures to repair.")
        return 0

    output = Path(args.output)
    with open(output, "w", encoding="utf-8") as f:
        for r in repairs:
            f.write(r.model_dump_json() + "\n")

    print(f"\nRepair tested {len(repairs)} failures")
    print(f"  Model: {args.model}")
    print(f"  Saved to: {output}")
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    """Run a multi-round experiment."""
    import json as json_mod

    from .experiment import Experiment
    from .report import ReportBuilder

    output_dir = Path(args.output_dir)
    client = AnthropicClient()
    exp = Experiment(
        client=client,
        models=args.models,
        judge_model=args.judge,
        output_dir=output_dir,
        evolve_model=args.evolve_model,
        params_dir=Path(args.params_dir) if args.params_dir else None,
        seed=args.seed,
    )
    data = exp.run(
        initial_count=args.count,
        rounds=args.rounds,
        min_score=args.min_score,
    )

    builder = ReportBuilder(data)
    report = builder.build()

    report_json = output_dir / "report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json_mod.dump(report, f, indent=2, default=str)

    report_md = output_dir / "report.md"
    report_md.write_text(builder.to_markdown(), encoding="utf-8")

    print(f"\nExperiment complete ({len(data['rounds'])} rounds)")
    print(f"  Total prompts:  {data['total_prompts']}")
    print(f"  Total failures: {data['total_failures']}")
    print(f"  Output dir:     {output_dir}")
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

    # -- evolve --
    evolve_p = subparsers.add_parser(
        "evolve", help="Evolve hard-case prompts into harder versions"
    )
    evolve_p.add_argument("results", help="Path to results JSONL file")
    evolve_p.add_argument(
        "--model", required=True, help="LLM model for evolution"
    )
    evolve_p.add_argument(
        "--min-score", type=int, default=6,
        help="Minimum score threshold for hard cases (default: 6)",
    )
    evolve_p.add_argument(
        "--output", "-o", default="evolved.jsonl",
        help="Output file for evolved prompts (default: evolved.jsonl)",
    )

    # -- repair --
    repair_p = subparsers.add_parser(
        "repair", help="Test model self-repair on failures"
    )
    repair_p.add_argument("results", help="Path to results JSONL file")
    repair_p.add_argument(
        "--model", required=True, help="LLM model for repair testing"
    )
    repair_p.add_argument(
        "--output", "-o", default="repairs.jsonl",
        help="Output file for repair results (default: repairs.jsonl)",
    )

    # -- experiment --
    exp_p = subparsers.add_parser(
        "experiment", help="Run multi-round evaluation experiment"
    )
    exp_p.add_argument(
        "--models", nargs="+", required=True,
        help="Models to evaluate",
    )
    exp_p.add_argument(
        "--judge", required=True, help="Judge model",
    )
    exp_p.add_argument(
        "--evolve-model", required=True,
        help="Model for prompt evolution",
    )
    exp_p.add_argument("--rounds", type=int, default=3)
    exp_p.add_argument("--count", type=int, default=10)
    exp_p.add_argument("--min-score", type=int, default=6)
    exp_p.add_argument("--output-dir", default="experiment_output")
    exp_p.add_argument("--params-dir", default=None)
    exp_p.add_argument("--seed", type=int, default=None)

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
    if args.command == "evolve":
        return _cmd_evolve(args)
    if args.command == "repair":
        return _cmd_repair(args)
    if args.command == "experiment":
        return _cmd_experiment(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
