"""CLI entry point: python -m reasonbench."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .client import AnthropicClient
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    """Run the ReasonBench evaluation pipeline."""
    default_model = os.environ.get("REASONBENCH_MODEL", "")
    default_judge = os.environ.get("REASONBENCH_JUDGE_MODEL", "")

    parser = argparse.ArgumentParser(
        prog="reasonbench",
        description="LLM Adversarial Reasoning Evaluation System",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=not bool(default_model),
        default=[default_model] if default_model else None,
        help="Models to evaluate (or set REASONBENCH_MODEL env var)",
    )
    parser.add_argument(
        "--judge",
        required=not bool(default_judge),
        default=default_judge or None,
        help="Judge model for validation (or set REASONBENCH_JUDGE_MODEL env var)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of prompts to generate (default: 10)",
    )
    parser.add_argument(
        "--output",
        default="results.jsonl",
        help="Output JSONL file (default: results.jsonl)",
    )
    parser.add_argument(
        "--params-dir",
        default=None,
        help="Custom parameter banks directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

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

    # Print summary
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


if __name__ == "__main__":
    sys.exit(main())
