"""Unit tests for Head 3 — Setting Painter."""

import pytest

from five_head_nlp.heads.scoring_config import SettingConfig
from five_head_nlp.heads.setting_painter import SettingPainter
from five_head_nlp.prediction.prediction_result import CandidateToken


class TestSettingPainter:
    """Test Setting Painter environment tracking, consistency bonuses, and shifts."""

    def test_setting_identification(self):
        head = SettingPainter()
        assert head.is_setting("castle")
        assert head.is_setting("cave")
        assert head.is_setting("forest")
        assert head.is_setting("garden")
        assert head.is_setting("tree")
        assert head.is_setting("lake")
        assert not head.is_setting("milo")
        assert not head.is_setting("flew")

    def test_initial_setting_establishment(self):
        head = SettingPainter()
        candidates = [CandidateToken("castle", 0.5, 1, -0.69), CandidateToken("the", 0.5, 2, -0.69)]
        scores = head.score_candidates(context=(), candidates=candidates)

        assert scores["castle"].raw_score == 0.85
        assert "Establishes initial setting" in scores["castle"].explanation
        assert scores["the"].raw_score == 0.20

    def test_setting_consistency_and_shift(self):
        head = SettingPainter()
        head.update_state("forest")
        assert head.current_setting == "forest"

        candidates = [
            CandidateToken("forest", 0.4, 1, -0.91),
            CandidateToken("castle", 0.3, 2, -1.20),
            CandidateToken("key", 0.3, 3, -1.20),
        ]
        scores = head.score_candidates(context=("bright",), candidates=candidates)

        # 'forest' reinforces current setting -> 0.85
        assert scores["forest"].raw_score == 0.85
        assert "Reinforces current setting" in scores["forest"].explanation

        # 'castle' is another core setting -> shift bonus 0.40
        assert scores["castle"].raw_score == 0.40
        assert "transition to new core setting" in scores["castle"].explanation

    def test_reset_state(self):
        head = SettingPainter()
        head.update_state("cave")
        assert head.current_setting == "cave"
        head.reset_state()
        assert head.current_setting is None
        assert head.setting_history == []
