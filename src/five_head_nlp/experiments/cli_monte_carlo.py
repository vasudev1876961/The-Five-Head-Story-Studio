"""Canonical CLI entry point for the Monte Carlo experimental suite."""

from __future__ import annotations

import argparse
import sys


def main(args: list[str] | None = None) -> int:
    """Run Phase 5 Monte Carlo experimental suite with side-effect-free --help."""
    parser = argparse.ArgumentParser(
        prog="five-head-monte-carlo",
        description="Run Phase 5 Monte Carlo experimental suite across language model configurations.",
    )
    parser.add_argument(
        "--stories-per-config",
        type=int,
        default=200,
        help="Number of stories to generate per experimental configuration (default: 200).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/monte_carlo",
        help="Target output directory for simulation artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without running simulation.",
    )

    parsed = parser.parse_args(args)
    if parsed.dry_run:
        print("Dry run completed: configuration is valid.")
        return 0

    from experiments.run_monte_carlo import main as run_main

    run_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
