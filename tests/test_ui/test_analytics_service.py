"""Unit tests for AnalyticsService querying Phase 5 artifacts."""

import pytest

from five_head_nlp.ui.analytics_service import AnalyticsService


class TestAnalyticsService:
    """Test loading and querying real Phase 5 Monte Carlo results."""

    def test_artifacts_availability(self):
        # We generated the real Phase 5 artifacts earlier
        avail = AnalyticsService.is_artifacts_available()
        assert avail is True

    def test_load_metadata(self):
        meta = AnalyticsService.load_metadata()
        assert "total_stories_generated" in meta
        assert meta["total_stories_generated"] == 1600
        assert meta["base_seed"] == 42

    def test_load_comparison_table(self):
        df = AnalyticsService.load_comparison_table()
        assert not df.empty
        assert len(df) == 8
        assert "config_id" in df.columns
        assert list(df["config_id"]) == ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]

    def test_load_per_story_and_aggregate_metrics(self):
        per_story = AnalyticsService.load_per_story_metrics()
        assert not per_story.empty
        assert len(per_story) == 1600

        agg = AnalyticsService.load_aggregate_metrics()
        assert not agg.empty
        assert "metric" in agg.columns

    def test_metric_labels(self):
        label = AnalyticsService.get_metric_display_name("repeated_bigram_pct")
        assert "Repeated Bigram" in label
