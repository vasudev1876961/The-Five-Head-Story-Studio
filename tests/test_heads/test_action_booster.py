"""Unit tests for Head 4 — Action Booster."""

import pytest

from five_head_nlp.heads.action_booster import ActionBooster
from five_head_nlp.prediction.prediction_result import CandidateToken


class TestActionBooster:
    """Test Action Booster dynamic vs state verb recognition and neutral scoring."""

    def test_action_identification(self):
        head = ActionBooster()
        assert head.is_action("flew")
        assert head.is_action("ran")
        assert head.is_action("found")
        assert head.is_action("saw")
        assert head.is_action("rested")
        assert not head.is_action("crystal")
        assert not head.is_action("milo")

    def test_dynamic_vs_stative_action_bonuses(self):
        head = ActionBooster()
        candidates = [
            CandidateToken("flew", 0.4, 1, -0.91),    # dynamic action
            CandidateToken("watched", 0.3, 2, -1.20), # state/perception verb
            CandidateToken("stone", 0.3, 3, -1.20),   # noun
        ]
        scores = head.score_candidates(context=("pip",), candidates=candidates)

        assert scores["flew"].raw_score == 0.90
        assert "Dynamic action verb" in scores["flew"].explanation

        assert scores["watched"].raw_score == 0.50
        assert "Perceptual or state verb" in scores["watched"].explanation

        assert scores["stone"].raw_score == 0.20
        assert "Non-action token" in scores["stone"].explanation

    def test_update_and_reset_state(self):
        head = ActionBooster()
        head.update_state("ran")
        assert head.action_history == ["ran"]
        head.update_state("castle")  # non-action
        assert head.action_history == ["ran"]

        head.reset_state()
        assert head.action_history == []
