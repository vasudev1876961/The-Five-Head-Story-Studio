"""Unit tests for ExportService."""

import json
import pytest

from five_head_nlp.ui.export_service import ExportService


class TestExportService:
    """Test exporting story records to TXT, Markdown, CSV, and JSON."""

    @pytest.fixture
    def mock_story(self):
        return {
            "story_id": "UI_001234",
            "text": "The clever cat milo lived near the crystal castle. Pip flew to the forest.",
            "token_count": 16,
            "sentence_count": 2,
            "generation_time_sec": 0.05,
            "tokens_per_second": 320.0,
            "termination_reason": "Target length achieved",
            "timestamp": "2026-11-24 01:00:00",
            "configuration": {
                "mode": "five_head",
                "smoother_name": "Trigram Backoff",
                "decoder_name": "Top-k",
                "seed": 42,
            },
            "metrics": {
                "repetition": {"repeated_bigram_rate": 0.05, "window_repetition_rate": 0.08},
                "diversity": {"ttr": 0.85},
                "character": {"character_continuity_ratio": 0.40},
                "action": {"dynamic_action_ratio": 0.90},
            },
        }

    def test_export_plain_text(self, mock_story):
        txt = ExportService.export_plain_text(mock_story)
        assert "THE FIVE-HEAD STORY STUDIO" in txt
        assert "UI_001234" in txt
        assert "The clever cat milo lived" in txt

    def test_export_markdown(self, mock_story):
        md = ExportService.export_markdown(mock_story)
        assert "# Generated Story: `UI_001234`" in md
        assert "## Narrative Text" in md
        assert "## Quality Metrics Summary" in md

    def test_export_scoring_csv(self):
        steps = [
            {
                "step_index": 1,
                "token": "castle",
                "context": ["the", "crystal"],
                "orchestrator_result": {
                    "fused_candidates": [{"token": "castle", "fused_score": 0.9, "rank": 1}],
                    "individual_head_scores": {
                        "castle": {
                            "Grammar Scout": {"normalized_score": 0.9},
                            "Character Keeper": {"normalized_score": 0.2},
                            "Setting Painter": {"normalized_score": 0.85},
                            "Action Booster": {"normalized_score": 0.2},
                            "Repeat Blocker": {"normalized_score": 1.0},
                        }
                    },
                },
            }
        ]
        csv_str = ExportService.export_scoring_csv(steps)
        assert "step_index,token,context" in csv_str
        assert "castle" in csv_str

    def test_export_json(self, mock_story):
        json_str = ExportService.export_story_json(mock_story)
        d = json.loads(json_str)
        assert d["story_id"] == "UI_001234"
