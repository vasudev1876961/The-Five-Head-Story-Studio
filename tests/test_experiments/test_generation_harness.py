"""Integration tests for story generation and harness."""

import pytest

from five_head_nlp.experiments.experiment_registry import get_experiment_config
from five_head_nlp.experiments.generation_harness import GenerationHarness
from five_head_nlp.experiments.story_generator import (
    StoryGenerator,
    detokenize_story,
)


class TestGenerationHarness:
    """Test story generator, constraints, seed reproducibility, and harness execution."""

    def test_detokenize_story(self):
        tokens = ["the", "clever", "cat", "milo", "lived", "near", "the", "castle", ".", "pip", "flew", "."]
        text = detokenize_story(tokens)
        assert text == "The clever cat milo lived near the castle. Pip flew."

    def test_story_generator_e3_backoff_produces_valid_story(self):
        cfg = get_experiment_config("E3")
        generator = StoryGenerator(config=cfg)

        story, failure = generator.generate_story(seed=42, story_id="TEST_001")
        assert failure is None
        assert story is not None
        assert story.story_id == "TEST_001"
        assert story.config_id == "E3"
        assert story.token_count >= 80
        assert story.sentence_count >= 3
        assert story.tokens[-1] == "."
        assert story.text.endswith(".")
        assert "length" in story.metrics

    def test_story_generator_e6_five_head_produces_valid_story(self):
        cfg = get_experiment_config("E6")
        generator = StoryGenerator(config=cfg)

        story, failure = generator.generate_story(seed=101, story_id="E6_TEST")
        assert failure is None
        assert story is not None
        assert story.config_id == "E6"
        assert story.token_count >= 80
        assert story.sentence_count >= 3
        assert story.tokens[-1] == "."

    def test_reproducibility_with_identical_seed(self):
        cfg = get_experiment_config("E6")
        generator = StoryGenerator(config=cfg)

        story1, _ = generator.generate_story(seed=999, story_id="RUN_1")
        story2, _ = generator.generate_story(seed=999, story_id="RUN_2")

        assert story1.tokens == story2.tokens
        assert story1.text == story2.text

    def test_harness_run_small_batch(self):
        harness = GenerationHarness(base_seed=123)
        cfg = get_experiment_config("E7")

        res = harness.run_configuration(config=cfg, num_stories=2, config_index=0)
        assert len(res.stories) == 2
        assert res.config_counts["E7"] == 2
        assert res.stories[0].story_id == "E7_0001"
        assert res.stories[1].story_id == "E7_0002"
