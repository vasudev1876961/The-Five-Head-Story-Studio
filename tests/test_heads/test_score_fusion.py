"""Unit tests for ScoreFusion."""

import pytest

from five_head_nlp.heads.head_result import HeadScore
from five_head_nlp.heads.score_fusion import ScoreFusion
from five_head_nlp.prediction.prediction_result import CandidateToken


class TestScoreFusion:
    """Test score fusion normalization, ranking, and deterministic tie-breaking."""

    def test_negative_weight_raises_error(self):
        fusion = ScoreFusion()
        with pytest.raises(ValueError, match="non-negative"):
            fusion.normalize_weights({"H1": -0.1, "H2": 1.0})

    def test_weight_normalization(self):
        fusion = ScoreFusion()
        norm = fusion.normalize_weights({"H1": 2.0, "H2": 3.0})
        assert norm["H1"] == 0.4
        assert norm["H2"] == 0.6

    def test_fused_scores_and_ranking(self):
        fusion = ScoreFusion()
        candidates = [
            CandidateToken("castle", 0.7, 1, -0.35),
            CandidateToken("cave", 0.3, 2, -1.20),
        ]
        head_scores = {
            "Grammar Scout": {
                "castle": HeadScore("Grammar Scout", "castle", 0.7, 0.7, "g"),
                "cave": HeadScore("Grammar Scout", "cave", 0.3, 0.3, "g"),
            },
            "Setting Painter": {
                "castle": HeadScore("Setting Painter", "castle", 0.9, 0.9, "s"),
                "cave": HeadScore("Setting Painter", "cave", 0.5, 0.5, "s"),
            },
        }
        weights = {"Grammar Scout": 0.5, "Setting Painter": 0.5}

        # castle: 0.5 * 0.7 + 0.5 * 0.9 = 0.35 + 0.45 = 0.80
        # cave:   0.5 * 0.3 + 0.5 * 0.5 = 0.15 + 0.25 = 0.40
        ranked = fusion.fuse_scores(candidates, head_scores, weights)

        assert len(ranked) == 2
        assert ranked[0].token == "castle"
        assert pytest.approx(ranked[0].fused_score, rel=1e-5) == 0.80
        assert ranked[0].rank == 1

        assert ranked[1].token == "cave"
        assert pytest.approx(ranked[1].fused_score, rel=1e-5) == 0.40
        assert ranked[1].rank == 2

    def test_deterministic_tie_breaking(self):
        fusion = ScoreFusion()
        candidates = [
            CandidateToken("zebra", 0.5, 1, -0.69),
            CandidateToken("apple", 0.5, 2, -0.69),
        ]
        head_scores = {
            "Grammar Scout": {
                "zebra": HeadScore("Grammar Scout", "zebra", 0.5, 0.5, "g"),
                "apple": HeadScore("Grammar Scout", "apple", 0.5, 0.5, "g"),
            }
        }
        weights = {"Grammar Scout": 1.0}

        ranked = fusion.fuse_scores(candidates, head_scores, weights)
        assert ranked[0].token == "apple"
        assert ranked[1].token == "zebra"
