"""CLI Entry Point for Phase 5 Monte Carlo Story Generation Experiments.

Usage:
------
    python experiments/run_monte_carlo.py [--stories-per-config N] [--output-dir DIR] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project src is in python path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from five_head_nlp.experiments.aggregation import (
    aggregate_config_metrics,
    build_comparison_table,
    export_aggregate_metrics_csv,
    export_configuration_comparison_csv,
    export_per_story_metrics_csv,
)
from five_head_nlp.experiments.experiment_registry import (
    EXPERIMENT_REGISTRY,
    get_experiment_config,
    list_all_experiments,
)
from five_head_nlp.experiments.generation_harness import GenerationHarness
from five_head_nlp.experiments.reporting import (
    generate_experiment_plots,
    generate_markdown_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase 5 Monte Carlo experimental suite across language model configurations."
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
        default=str(ROOT_DIR / "experiments" / "results" / "monte_carlo"),
        help="Directory to save experimental artifacts, CSVs, and plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible Monte Carlo generation (default: 42).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Specific configuration IDs to run (e.g. E1 E2 E6). Defaults to all (E1-E8).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute batch Monte Carlo story generation experiment."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PHASE 5 — MONTE CARLO STORY GENERATION HARNESS")
    print("=" * 78)
    print(f"Target Stories per Config : {args.stories_per_config}")
    print(f"Base Random Seed          : {args.seed}")
    print(f"Output Directory          : {output_dir}")

    # Determine configurations to execute
    if args.configs:
        selected_configs = [get_experiment_config(cid) for cid in args.configs]
    else:
        selected_configs = list_all_experiments()

    print(f"Active Configurations     : {[c.config_id for c in selected_configs]}")
    total_target = len(selected_configs) * args.stories_per_config
    print(f"Total Stories Target      : {total_target}")
    print("-" * 78)

    # Save configurations specification
    configs_json_path = output_dir / "configurations.json"
    configs_dict = {c.config_id: c.to_dict() for c in selected_configs}
    with open(configs_json_path, "w", encoding="utf-8") as f:
        json.dump(configs_dict, f, indent=2)

    # Initialize harness
    harness = GenerationHarness(base_seed=args.seed)

    def progress_callback(config_id: str, done: int, total: int) -> None:
        pct = (done / total) * 100.0
        sys.stdout.write(f"\r  [{config_id}] Generated {done}/{total} stories ({pct:5.1f}%)")
        sys.stdout.flush()

    start_time = time.perf_counter()

    all_stories = []
    all_failures = []
    config_names = {c.config_id: c.name for c in selected_configs}

    for c_idx, cfg in enumerate(selected_configs):
        print(f"\n--> Running Condition {cfg.config_id}: {cfg.name}...")
        res = harness.run_configuration(
            config=cfg,
            num_stories=args.stories_per_config,
            config_index=c_idx,
            progress_callback=lambda d, t: progress_callback(cfg.config_id, d, t),
        )
        print()
        all_stories.extend(res.stories)
        all_failures.extend(res.failures)

    total_wall_time = time.perf_counter() - start_time
    print("-" * 78)
    print(f"Generation Complete! Total Generated: {len(all_stories)}, Failures: {len(all_failures)}")
    print(f"Total Elapsed Time: {total_wall_time:.2f} seconds ({len(all_stories) / max(0.001, total_wall_time):.1f} stories/sec)")

    # 1. Save stories to JSONL
    stories_jsonl_path = output_dir / "generated_stories.jsonl"
    print(f"Writing {stories_jsonl_path}...")
    with open(stories_jsonl_path, "w", encoding="utf-8") as f:
        for s in all_stories:
            f.write(json.dumps(s.to_dict()) + "\n")

    # 2. Save failure records to JSONL
    failures_jsonl_path = output_dir / "failure_records.jsonl"
    print(f"Writing {failures_jsonl_path}...")
    with open(failures_jsonl_path, "w", encoding="utf-8") as f:
        for fl in all_failures:
            f.write(json.dumps(fl.to_dict()) + "\n")

    # 3. Export per-story metrics CSV
    per_story_csv_path = output_dir / "per_story_metrics.csv"
    print(f"Writing {per_story_csv_path}...")
    export_per_story_metrics_csv(all_stories, per_story_csv_path)

    # 4. Aggregate metrics and export CSVs
    print("Computing aggregate metrics...")
    agg_metrics = aggregate_config_metrics(all_stories)
    agg_csv_path = output_dir / "aggregate_metrics.csv"
    export_aggregate_metrics_csv(agg_metrics, agg_csv_path)

    # 5. Build and export configuration comparison CSV
    comparison_rows = build_comparison_table(agg_metrics, config_names)
    comp_csv_path = output_dir / "configuration_comparison.csv"
    export_configuration_comparison_csv(comparison_rows, comp_csv_path)

    # 6. Save run metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_seed": args.seed,
        "stories_per_config": args.stories_per_config,
        "total_stories_generated": len(all_stories),
        "total_failures": len(all_failures),
        "wall_time_sec": round(total_wall_time, 2),
        "configurations_evaluated": [c.config_id for c in selected_configs],
    }
    meta_path = output_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 7. Generate publication-quality Matplotlib plots
    print("Generating Matplotlib analytical plots...")
    plot_files = generate_experiment_plots(comparison_rows, plots_dir)
    print(f"Generated {len(plot_files)} charts in {plots_dir}.")

    # 8. Generate markdown summary
    summary_md_path = output_dir / "experiment_summary.md"
    print(f"Writing summary report to {summary_md_path}...")
    generate_markdown_summary(comparison_rows, summary_md_path, metadata)

    # 9. Print concise tabular summary to stdout
    print("\n" + "=" * 90)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 90)
    header = f"{'ID':<4} {'Name':<22} {'Tokens':<12} {'Sents':<6} {'BigramRep%':<11} {'TTR':<6} {'CharCont':<9} {'DynAct%':<8} {'Lat(ms)':<8}"
    print(header)
    print("-" * 90)
    for r in comparison_rows:
        tok_str = f"{r['mean_tokens']:.1f}±{r['std_tokens']:.1f}"
        print(
            f"{r['config_id']:<4} {r['config_name']:<22} {tok_str:<12} {r['mean_sentences']:<6.1f} "
            f"{r['repeated_bigram_pct']:<11.1f} {r['ttr']:<6.3f} {r['character_continuity']:<9.3f} "
            f"{r['dynamic_action_ratio']*100:<8.1f} {r['token_latency_ms']:<8.2f}"
        )
    print("=" * 90)
    print(f"All Phase 5 Monte Carlo artifacts successfully generated in {output_dir}")


if __name__ == "__main__":
    main()
