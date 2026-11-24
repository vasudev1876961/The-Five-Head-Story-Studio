"""Canonical CLI entry point for next-token prediction demonstration."""

from __future__ import annotations

import argparse
import sys


def main(args: list[str] | None = None) -> int:
    """Run Phase 3 next-token prediction demonstration with side-effect-free --help."""
    parser = argparse.ArgumentParser(
        prog="five-head-prediction",
        description="Demonstrate next-token probability prediction and decoding strategies.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/next_token_prediction",
        help="Directory to save prediction evaluation metrics and artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate parameters and exit without generating artifacts.",
    )

    parsed = parser.parse_args(args)
    if parsed.dry_run:
        print("Dry run completed: prediction demo configuration valid.")
        return 0

    from experiments.run_next_token_prediction import main as run_main

    run_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
