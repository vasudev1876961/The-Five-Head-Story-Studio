"""Unit tests for Phase 5 metrics modules."""

import pytest

from five_head_nlp.experiments.metrics import (
    calculate_action_metrics,
    calculate_character_metrics,
    calculate_diversity_metrics,
    calculate_latency_metrics,
    calculate_length_metrics,
    calculate_readability_metrics,
    calculate_repetition_metrics,
    calculate_setting_metrics,
    compute_story_metrics,
)


class TestMetrics:
    """Test all metric computation functions with deterministic synthetic inputs."""

    def test_length_metrics(self):
        tokens = ["milo", "found", "a", "key", ".", "pip", "flew", "home", "."]
        m = calculate_length_metrics(tokens)

        assert m["total_tokens"] == 9
        assert m["total_words"] == 7
        assert m["sentence_count"] == 2
        assert m["sentence_lengths"] == [5, 4]
        assert m["mean_sentence_length"] == 4.5
        assert m["min_sentence_length"] == 4
        assert m["max_sentence_length"] == 5

    def test_repetition_metrics(self):
        # Repetitive sequence: "milo milo found key key"
        tokens = ["the", "milo", "milo", "found", "key", "key", "."]
        m = calculate_repetition_metrics(tokens)

        assert m["immediate_token_repeat_count"] == 2
        assert m["immediate_token_repeat_rate"] > 0.0
        assert m["has_repetition_event"] is True

        # Clean sequence
        clean_tokens = ["the", "clever", "cat", "milo", "lived", "near", "castle", "."]
        clean_m = calculate_repetition_metrics(clean_tokens)
        assert clean_m["immediate_token_repeat_count"] == 0
        assert clean_m["immediate_token_repeat_rate"] == 0.0

    def test_diversity_metrics(self):
        tokens = ["milo", "liked", "the", "crystal", "castle", "."]
        m = calculate_diversity_metrics(tokens)

        # 5 distinct words, total words 5 (excluding '.')
        assert m["vocabulary_size"] == 5
        assert m["ttr"] == 1.0
        assert m["hapax_count"] == 5
        assert m["hapax_ratio"] == 1.0

    def test_character_metrics(self):
        tokens = ["the", "clever", "cat", "milo", "liked", "pip", "and", "milo", "."]
        m = calculate_character_metrics(tokens)

        assert m["character_mentions"] >= 3
        assert "milo" in m["characters_found"]
        assert "pip" in m["characters_found"]
        assert m["unique_characters"] >= 2
        assert m["character_density"] > 0.0

    def test_setting_metrics(self):
        tokens = ["the", "castle", "was", "near", "the", "forest", "and", "garden", "."]
        m = calculate_setting_metrics(tokens)

        assert m["setting_mentions"] >= 3
        assert "castle" in m["settings_found"]
        assert "forest" in m["settings_found"]
        assert "garden" in m["settings_found"]
        assert m["setting_density"] > 0.0

    def test_action_metrics(self):
        tokens = ["pip", "flew", "and", "milo", "ran", "while", "owl", "rested", "."]
        m = calculate_action_metrics(tokens)

        assert m["dynamic_action_count"] == 2  # flew, ran
        assert m["stative_action_count"] == 1  # rested
        assert m["total_action_count"] == 3
        assert m["dynamic_action_ratio"] == pytest.approx(2 / 3, rel=1e-3)

    def test_readability_metrics(self):
        tokens = ["milo", "found", "the", "silver", "key", "."]
        m = calculate_readability_metrics(tokens)

        assert m["avg_sentence_length_words"] == 5.0
        assert m["avg_word_length_chars"] > 0.0
        assert 0.0 <= m["lexical_density"] <= 1.0

    def test_latency_metrics(self):
        lats = [10.0, 12.0, 14.0, 16.0]
        m = calculate_latency_metrics(generation_time_sec=0.052, token_latencies_ms=lats)

        assert m["tokens_measured"] == 4
        assert m["mean_token_latency_ms"] == 13.0
        assert m["median_token_latency_ms"] == 13.0
        assert m["tokens_per_second"] > 0.0

    def test_compute_story_metrics_integration(self):
        tokens = ["the", "clever", "cat", "milo", "lived", "near", "the", "crystal", "castle", "."]
        sm = compute_story_metrics(tokens=tokens, generation_time_sec=0.02, token_latencies_ms=[2.0]*10)

        d = sm.to_dict()
        assert "length" in d
        assert "repetition" in d
        assert "diversity" in d
        assert "character" in d
        assert "setting" in d
        assert "action" in d
        assert "readability" in d
        assert "latency" in d

        flat = sm.to_flat_dict()
        assert "length_total_tokens" in flat
        assert "repetition_repeated_bigram_rate" in flat
        assert "diversity_ttr" in flat
        assert flat["length_total_tokens"] == 10
