"""Integration tests for GenerationService."""

import pytest

from five_head_nlp.ui.generation_service import GenerationService


class TestGenerationService:
    """Test story generation service across modes, constraints, and diagnostics."""

    @pytest.fixture
    def service(self):
        return GenerationService()

    def test_baseline_generation(self, service):
        config = {
            "mode": "baseline",
            "smoother_name": "Trigram Backoff",
            "decoder_name": "Greedy",
            "seed": 42,
            "max_tokens": 40,
            "min_sentences": 2,
            "max_sentences": 5,
            "starter_prompt": "pip lived near",
        }
        story, steps = service.generate_story(config)

        assert story is not None
        assert story["token_count"] >= 10
        assert story["sentence_count"] >= 2
        assert story["text"].endswith(".")
        assert len(steps) > 0
        assert steps[0]["mode"] == "baseline"
        assert len(steps[0]["candidates"]) > 0

    def test_five_head_generation_with_diagnostics(self, service):
        config = {
            "mode": "five_head",
            "smoother_name": "Trigram Backoff",
            "decoder_name": "Top-k",
            "decoder_params": {"k": 5},
            "seed": 100,
            "max_tokens": 50,
            "min_sentences": 2,
            "max_sentences": 5,
            "starter_prompt": "the clever cat",
            "weights": {
                "Grammar Scout": 0.40,
                "Character Keeper": 0.15,
                "Setting Painter": 0.15,
                "Action Booster": 0.10,
                "Repeat Blocker": 0.20,
            },
            "enabled_heads": {
                "Grammar Scout": True,
                "Character Keeper": True,
                "Setting Painter": True,
                "Action Booster": True,
                "Repeat Blocker": True,
            },
        }
        story, steps = service.generate_story(config)

        assert story is not None
        assert story["token_count"] >= 10
        assert "metrics" in story
        assert len(steps) > 0

        # Verify that five-head diagnostic metadata was captured
        has_orch = any(s.get("orchestrator_result") is not None for s in steps)
        assert has_orch is True

    def test_seed_determinism(self, service):
        config = {
            "mode": "five_head",
            "smoother_name": "Trigram Backoff",
            "decoder_name": "Top-k",
            "decoder_params": {"k": 5},
            "seed": 777,
            "max_tokens": 30,
            "min_sentences": 2,
            "max_sentences": 4,
            "starter_prompt": "milo found",
        }
        story1, _ = service.generate_story(config)
        story2, _ = service.generate_story(config)

        assert story1["tokens"] == story2["tokens"]
        assert story1["text"] == story2["text"]
