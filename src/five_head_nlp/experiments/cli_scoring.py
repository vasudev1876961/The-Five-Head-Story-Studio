"""Canonical CLI entry point for five-head transparent scoring demonstration."""

from __future__ import annotations

import argparse
import sys


def main(args: list[str] | None = None) -> int:
    """Run Phase 4 five-head scoring demonstration with side-effect-free --help."""
    parser = argparse.ArgumentParser(
        prog="five-head-scoring",
        description="Demonstrate the five transparent scoring heads and score fusion engine.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/five_head_scoring",
        help="Directory to save scoring demonstration artifacts and charts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate parameters and exit without generating artifacts.",
    )

    parsed = parser.parse_args(args)
    if parsed.dry_run:
        print("Dry run completed: scoring demo configuration valid.")
        return 0

    from experiments.run_five_head_scoring import main as run_main

    run_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
