"""Statistical aggregation routines for Monte Carlo story generation results."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from five_head_nlp.experiments.story_generator import GeneratedStory


def flatten_story_metrics(story: GeneratedStory) -> Dict[str, Any]:
    """Flatten story metadata and hierarchical metrics into a single row dictionary."""
    row: Dict[str, Any] = {
        "story_id": story.story_id,
        "config_id": story.config_id,
        "config_name": story.config_name,
        "seed": story.seed,
        "sentence_count": story.sentence_count,
        "token_count": story.token_count,
        "generation_time_sec": round(story.generation_time_sec, 4),
    }

    # Flatten nested metrics dictionaries
    for category, cat_dict in story.metrics.items():
        if isinstance(cat_dict, dict):
            for key, val in cat_dict.items():
                if isinstance(val, (list, tuple, set)):
                    continue
                row[f"{category}_{key}"] = val

    return row


def export_per_story_metrics_csv(
    stories: Sequence[GeneratedStory],
    output_path: Path,
) -> None:
    """Export per-story individual metric records to CSV."""
    if not stories:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_story_metrics(s) for s in stories]
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_config_metrics(
    stories: Sequence[GeneratedStory],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute mean, median, std, min, max for each metric grouped by config_id.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, float]]]
        {config_id: {metric_name: {"mean": x, "median": y, "std": z, "min": a, "max": b, "count": n}}}
    """
    grouped_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for story in stories:
        flat = flatten_story_metrics(story)
        cid = story.config_id
        for k, v in flat.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if k not in {"seed"}:
                    grouped_data[cid][k].append(float(v))

    agg_result: Dict[str, Dict[str, Dict[str, float]]] = {}

    for cid in sorted(grouped_data.keys()):
        agg_result[cid] = {}
        for metric, values in grouped_data[cid].items():
            if not values:
                continue
            cnt = len(values)
            m_val = float(statistics.mean(values))
            med_val = float(statistics.median(values))
            s_val = float(statistics.stdev(values)) if cnt > 1 else 0.0
            min_val = float(min(values))
            max_val = float(max(values))

            agg_result[cid][metric] = {
                "count": cnt,
                "mean": round(m_val, 4),
                "median": round(med_val, 4),
                "std": round(s_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
            }

    return agg_result


def export_aggregate_metrics_csv(
    agg_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """Export detailed aggregate metrics (mean, std, median, min, max) to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id",
        "metric",
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max",
    ]

    rows: List[Dict[str, Any]] = []
    for cid, metrics in agg_metrics.items():
        for metric_name, stats in metrics.items():
            rows.append({
                "config_id": cid,
                "metric": metric_name,
                "count": stats["count"],
                "mean": stats["mean"],
                "median": stats["median"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_comparison_table(
    agg_metrics: Dict[str, Dict[str, Dict[str, float]]],
    config_names: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Build a consolidated summary table comparing configurations across key indicators."""
    rows: List[Dict[str, Any]] = []

    for cid in sorted(agg_metrics.keys()):
        m = agg_metrics[cid]
        cname = config_names.get(cid, cid)

        row = {
            "config_id": cid,
            "config_name": cname,
            "stories_count": int(m.get("token_count", {}).get("count", 0)),
            "mean_tokens": m.get("length_total_tokens", {}).get("mean", 0.0),
            "std_tokens": m.get("length_total_tokens", {}).get("std", 0.0),
            "mean_sentences": m.get("length_sentence_count", {}).get("mean", 0.0),
            "repeated_bigram_pct": round(m.get("repetition_repeated_bigram_rate", {}).get("mean", 0.0) * 100.0, 2),
            "repeated_trigram_pct": round(m.get("repetition_repeated_trigram_rate", {}).get("mean", 0.0) * 100.0, 2),
            "window_repeat_pct": round(m.get("repetition_window_repetition_rate", {}).get("mean", 0.0) * 100.0, 2),
            "ttr": m.get("diversity_ttr", {}).get("mean", 0.0),
            "mattr_20": m.get("diversity_mattr_20", {}).get("mean", 0.0),
            "character_density": m.get("character_character_density", {}).get("mean", 0.0),
            "character_continuity": m.get("character_character_continuity_ratio", {}).get("mean", 0.0),
            "setting_density": m.get("setting_setting_density", {}).get("mean", 0.0),
            "setting_consistency": m.get("setting_setting_consistency_ratio", {}).get("mean", 0.0),
            "action_density": m.get("action_action_density", {}).get("mean", 0.0),
            "dynamic_action_ratio": m.get("action_dynamic_action_ratio", {}).get("mean", 0.0),
            "token_latency_ms": m.get("latency_mean_token_latency_ms", {}).get("mean", 0.0),
            "tokens_per_sec": m.get("latency_tokens_per_second", {}).get("mean", 0.0),
        }
        rows.append(row)

    return rows


def export_configuration_comparison_csv(
    comparison_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Export the high-level configuration comparison table to CSV."""
    if not comparison_rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(comparison_rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
