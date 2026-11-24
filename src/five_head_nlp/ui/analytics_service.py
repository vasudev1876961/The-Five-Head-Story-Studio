"""Service loading and querying real Phase 5 Monte Carlo experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _get_project_root() -> Path:
    """Resolve project root directory."""
    return Path(__file__).resolve().parents[3]


class AnalyticsService:
    """Reads, caches, and filters Phase 5 Monte Carlo experimental results."""

    RESULTS_DIR: Path = _get_project_root() / "experiments" / "results" / "monte_carlo"

    @classmethod
    def is_artifacts_available(cls) -> bool:
        """Check if Phase 5 experiment CSVs exist on disk."""
        comp_file = cls.RESULTS_DIR / "configuration_comparison.csv"
        return comp_file.is_file()

    @classmethod
    def load_metadata(cls) -> Dict[str, Any]:
        """Load experiment run metadata (seed, total stories, timestamp)."""
        meta_file = cls.RESULTS_DIR / "run_metadata.json"
        if meta_file.is_file():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @classmethod
    def load_comparison_table(cls) -> pd.DataFrame:
        """Load high-level configuration comparison table (E1 to E8)."""
        csv_file = cls.RESULTS_DIR / "configuration_comparison.csv"
        if csv_file.is_file():
            try:
                return pd.read_csv(csv_file)
            except Exception:
                pass
        return pd.DataFrame()

    @classmethod
    def load_aggregate_metrics(cls) -> pd.DataFrame:
        """Load detailed aggregate metrics (mean, std, median, min, max)."""
        csv_file = cls.RESULTS_DIR / "aggregate_metrics.csv"
        if csv_file.is_file():
            try:
                return pd.read_csv(csv_file)
            except Exception:
                pass
        return pd.DataFrame()

    @classmethod
    def load_per_story_metrics(cls) -> pd.DataFrame:
        """Load individual story records across all 1,600 runs."""
        csv_file = cls.RESULTS_DIR / "per_story_metrics.csv"
        if csv_file.is_file():
            try:
                return pd.read_csv(csv_file)
            except Exception:
                pass
        return pd.DataFrame()

    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Return list of numeric metrics available for comparative plotting."""
        return [
            "repeated_bigram_pct",
            "repeated_trigram_pct",
            "window_repeat_pct",
            "ttr",
            "mattr_20",
            "character_continuity",
            "character_density",
            "setting_consistency",
            "setting_density",
            "dynamic_action_ratio",
            "action_density",
            "mean_tokens",
            "mean_sentences",
            "token_latency_ms",
            "tokens_per_sec",
        ]

    @classmethod
    def get_metric_display_name(cls, metric_key: str) -> str:
        """Return user-friendly display label for a metric key."""
        labels = {
            "repeated_bigram_pct": "Repeated Bigram Rate (%)",
            "repeated_trigram_pct": "Repeated Trigram Rate (%)",
            "window_repeat_pct": "Rolling Window Repetition Rate (%)",
            "ttr": "Lexical Type-Token Ratio (TTR)",
            "mattr_20": "Moving-Average TTR (MATTR-20)",
            "character_continuity": "Character Continuity Ratio",
            "character_density": "Character Mention Density",
            "setting_consistency": "Setting Consistency Ratio",
            "setting_density": "Setting Mention Density",
            "dynamic_action_ratio": "Dynamic Action Proportion (%)",
            "action_density": "Action Verb Density",
            "mean_tokens": "Mean Story Length (Tokens)",
            "mean_sentences": "Mean Completed Sentences",
            "token_latency_ms": "Mean Latency per Token (ms)",
            "tokens_per_sec": "Generation Throughput (Tokens/s)",
        }
        return labels.get(metric_key, metric_key.replace("_", " ").title())
