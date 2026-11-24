"""Unit tests for statistical aggregation routines."""

import pytest

from five_head_nlp.experiments.aggregation import (
    aggregate_config_metrics,
    build_comparison_table,
    flatten_story_metrics,
)
from five_head_nlp.experiments.story_generator import GeneratedStory


class TestAggregation:
    """Test metrics flattening, aggregation, and comparison table creation."""

    @pytest.fixture
    def mock_stories(self):
        s1 = GeneratedStory(
            story_id="E1_0001",
            config_id="E1",
            config_name="Baseline MLE",
            seed=42,
            tokens=["milo", "found", "a", "key", "."],
            text="Milo found a key.",
            sentence_count=1,
            token_count=5,
            generation_time_sec=0.01,
            token_latencies_ms=[2.0] * 5,
            metrics={
                "length": {"total_tokens": 5, "sentence_count": 1},
                "repetition": {"repeated_bigram_rate": 0.20, "repeated_trigram_rate": 0.10, "window_repetition_rate": 0.15},
                "diversity": {"ttr": 0.80, "mattr_20": 0.80},
                "character": {"character_density": 0.20, "character_continuity_ratio": 0.50},
                "setting": {"setting_density": 0.10, "setting_consistency_ratio": 0.90},
                "action": {"action_density": 0.20, "dynamic_action_ratio": 0.75},
                "readability": {"lexical_density": 0.60},
                "latency": {"mean_token_latency_ms": 2.0, "tokens_per_second": 500.0},
            },
        )
        s2 = GeneratedStory(
            story_id="E1_0002",
            config_id="E1",
            config_name="Baseline MLE",
            seed=43,
            tokens=["pip", "flew", "home", "."],
            text="Pip flew home.",
            sentence_count=1,
            token_count=4,
            generation_time_sec=0.01,
            token_latencies_ms=[2.0] * 4,
            metrics={
                "length": {"total_tokens": 4, "sentence_count": 1},
                "repetition": {"repeated_bigram_rate": 0.10, "repeated_trigram_rate": 0.05, "window_repetition_rate": 0.10},
                "diversity": {"ttr": 0.90, "mattr_20": 0.90},
                "character": {"character_density": 0.25, "character_continuity_ratio": 0.60},
                "setting": {"setting_density": 0.10, "setting_consistency_ratio": 0.80},
                "action": {"action_density": 0.25, "dynamic_action_ratio": 0.80},
                "readability": {"lexical_density": 0.70},
                "latency": {"mean_token_latency_ms": 2.0, "tokens_per_second": 400.0},
            },
        )
        return [s1, s2]

    def test_flatten_story_metrics(self, mock_stories):
        flat = flatten_story_metrics(mock_stories[0])
        assert flat["story_id"] == "E1_0001"
        assert flat["length_total_tokens"] == 5
        assert flat["repetition_repeated_bigram_rate"] == 0.20
        assert flat["diversity_ttr"] == 0.80

    def test_aggregate_config_metrics(self, mock_stories):
        agg = aggregate_config_metrics(mock_stories)
        assert "E1" in agg
        tokens_stat = agg["E1"]["length_total_tokens"]
        assert tokens_stat["count"] == 2
        assert tokens_stat["mean"] == 4.5
        assert tokens_stat["min"] == 4.0
        assert tokens_stat["max"] == 5.0

    def test_build_comparison_table(self, mock_stories):
        agg = aggregate_config_metrics(mock_stories)
        names = {"E1": "Baseline MLE"}
        table = build_comparison_table(agg, names)

        assert len(table) == 1
        row = table[0]
        assert row["config_id"] == "E1"
        assert row["stories_count"] == 2
        assert row["mean_tokens"] == 4.5
        assert row["ttr"] == 0.85
