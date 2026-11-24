"""Unit tests for Head 2 — Character Keeper."""

import pytest

from five_head_nlp.heads.character_keeper import CharacterKeeper
from five_head_nlp.heads.scoring_config import CharacterConfig
from five_head_nlp.prediction.prediction_result import CandidateToken


class TestCharacterKeeper:
    """Test Character Keeper continuity tracking, introduction bonuses, and state updates."""

    def test_character_identification(self):
        head = CharacterKeeper()
        assert head.is_character("milo")
        assert head.is_character("pip")
        assert head.is_character("zara")
        assert head.is_character("owl")
        assert head.is_character("fox")
        assert head.is_character("cat")
        assert head.is_character("dragon")
        assert not head.is_character("castle")
        assert not head.is_character("the")

    def test_introduction_bonus(self):
        head = CharacterKeeper()
        candidates = [
            CandidateToken("milo", 0.5, 1, -0.69),
            CandidateToken("castle", 0.5, 2, -0.69),
        ]
        scores = head.score_candidates(context=(), candidates=candidates)

        # 'milo' has not appeared yet -> introduction bonus
        assert scores["milo"].raw_score == 0.50
        assert "Introduces new character" in scores["milo"].explanation
        # 'castle' is non-character -> neutral score
        assert scores["castle"].raw_score == 0.20

    def test_continuity_bonus_and_state_update(self):
        head = CharacterKeeper()
        assert head.active_character is None

        # Select 'milo'
        head.update_state("milo")
        assert head.active_character == "milo"
        assert head.mentioned_characters == ["milo"]

        candidates = [
            CandidateToken("milo", 0.4, 1, -0.91),
            CandidateToken("pip", 0.3, 2, -1.20),
            CandidateToken("tree", 0.3, 3, -1.20),
        ]
        scores = head.score_candidates(context=("and",), candidates=candidates)

        # 'milo' matches active character -> continuity bonus
        assert scores["milo"].raw_score == 0.85
        assert "active character focus" in scores["milo"].explanation

        # 'pip' is a new character -> introduction bonus
        assert scores["pip"].raw_score == 0.50

    def test_immediate_repeat_penalty(self):
        head = CharacterKeeper()
        head.update_state("milo")
        candidates = [CandidateToken("milo", 0.5, 1, -0.69)]

        # Context ends with 'milo' -> immediate repeat penalized
        scores = head.score_candidates(context=("milo",), candidates=candidates)
        assert scores["milo"].raw_score == 0.10
        assert "immediate duplicate" in scores["milo"].explanation

        # Context is empty, but state.all_tokens ends with 'milo'
        from five_head_nlp.prediction.generation_state import GenerationState
        state = GenerationState(prompt=["the", "brave", "milo"])
        scores_state = head.score_candidates(context=(), candidates=candidates, state=state)
        assert scores_state["milo"].raw_score == 0.10
        assert "immediate duplicate" in scores_state["milo"].explanation

    def test_reset_state(self):
        head = CharacterKeeper()
        head.update_state("zara")
        assert head.active_character == "zara"

        head.reset_state()
        assert head.active_character is None
        assert head.mentioned_characters == []
