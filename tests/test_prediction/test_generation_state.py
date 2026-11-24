"""Unit tests for GenerationState."""

import pytest

from five_head_nlp.prediction.generation_state import GenerationState


class TestGenerationState:
    """Test token tracking, context rolling, and sentence-boundary termination."""

    def test_initialization_empty_prompt(self):
        state = GenerationState(max_tokens=10)
        assert state.prompt == ()
        assert state.generated_tokens == []
        assert state.all_tokens == []
        assert state.num_generated == 0
        assert not state.is_terminated
        assert state.can_continue()

    def test_initialization_with_prompt(self):
        state = GenerationState(prompt=["the", "crystal"], max_tokens=10)
        assert state.prompt == ("the", "crystal")
        assert state.all_tokens == ["the", "crystal"]
        assert state.can_continue()

    def test_prompt_ending_with_period_is_immediately_terminated(self):
        state = GenerationState(prompt=["milo", "smiled", "."])
        assert state.is_terminated
        assert not state.can_continue()

    def test_append_updates_sequence_and_context(self):
        state = GenerationState(prompt=["milo"], max_tokens=5)
        state.append("found")
        state.append("the")

        assert state.num_generated == 2
        assert state.all_tokens == ["milo", "found", "the"]
        assert state.get_context(window_size=2) == ("found", "the")
        assert state.get_context(window_size=1) == ("the",)
        assert state.can_continue()

    def test_period_halts_generation_immediately(self):
        state = GenerationState(prompt=["milo"], max_tokens=10)
        state.append("smiled")
        assert state.can_continue()

        state.append(".")
        assert state.is_terminated
        assert not state.can_continue()

        # Cannot append after termination
        with pytest.raises(RuntimeError, match="already terminated"):
            state.append("again")

    def test_max_tokens_enforced(self):
        state = GenerationState(prompt=[], max_tokens=3)
        state.append("word1")
        state.append("word2")
        assert state.can_continue()

        state.append("word3")
        assert state.is_terminated
        assert not state.can_continue()

        with pytest.raises(RuntimeError, match="already terminated"):
            state.append("word4")

    def test_reset(self):
        state = GenerationState(prompt=["a"], max_tokens=5)
        state.append("b")
        state.append(".")
        assert state.is_terminated

        state.reset(new_prompt=["x", "y"])
        assert state.prompt == ("x", "y")
        assert state.generated_tokens == []
        assert not state.is_terminated
        assert state.can_continue()
