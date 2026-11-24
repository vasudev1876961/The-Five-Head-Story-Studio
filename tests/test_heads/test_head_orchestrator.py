"""Unit tests for HeadOrchestrator."""

import pytest

from five_head_nlp.heads.head_orchestrator import HeadOrchestrator
from five_head_nlp.heads.scoring_config import ScoringConfig
from five_head_nlp.prediction.prediction_result import CandidateToken


class TestHeadOrchestrator:
    """Test HeadOrchestrator coordination, pure evaluation, and state propagation."""

    def test_orchestrator_initialization_and_weights(self):
        orchestrator = HeadOrchestrator()
        assert len(orchestrator.heads) == 5
        weights = orchestrator.get_active_weights()
        assert weights["Grammar Scout"] == 0.40
        assert weights["Character Keeper"] == 0.15

    def test_pure_evaluation_does_not_mutate_head_states(self):
        """Audit requirement: candidate scoring must NOT mutate internal state."""
        orchestrator = HeadOrchestrator()
        candidates = [CandidateToken("milo", 0.5, 1, -0.69), CandidateToken("castle", 0.5, 2, -0.69)]

        # Before scoring
        assert orchestrator.character_head.active_character is None
        assert orchestrator.setting_head.current_setting is None

        res = orchestrator.evaluate_candidates(context=("the",), candidates=candidates)
        assert len(res.fused_candidates) == 2
        assert res.d10_allocation["Grammar Scout"] == 4

        # After scoring, state must remain unmutated!
        assert orchestrator.character_head.active_character is None
        assert orchestrator.setting_head.current_setting is None

    def test_state_updates_after_token_selection(self):
        orchestrator = HeadOrchestrator()
        orchestrator.update_state("milo")
        assert orchestrator.character_head.active_character == "milo"

        orchestrator.update_state("castle")
        assert orchestrator.setting_head.current_setting == "castle"

    def test_empty_candidates_returns_clean_result(self):
        orchestrator = HeadOrchestrator()
        res = orchestrator.evaluate_candidates(context=("the",), candidates=[])
        assert res.selected_token is None
        assert res.fused_candidates == []
