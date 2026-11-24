"""Unit tests for Head 1 — Grammar Scout."""

import pytest

from five_head_nlp.heads.grammar_scout import GrammarScout
from five_head_nlp.prediction.prediction_result import CandidateToken


class TestGrammarScout:
    """Test Grammar Scout probability extraction and explanations."""

    def test_score_candidates_reflects_model_probabilities(self):
        head = GrammarScout(weight=0.40)
        candidates = [
            CandidateToken(token="castle", probability=0.9167, rank=1, log_probability=-0.087),
            CandidateToken(token="the", probability=0.0118, rank=2, log_probability=-4.440),
        ]
        scores = head.score_candidates(context=("the", "crystal"), candidates=candidates)

        assert len(scores) == 2
        assert scores["castle"].raw_score == 0.9167
        assert scores["castle"].normalized_score == 0.9167
        assert "Statistically dominant continuation" in scores["castle"].explanation
        assert scores["the"].raw_score == 0.0118

    def test_unseen_low_probability_explanation(self):
        head = GrammarScout()
        candidates = [CandidateToken(token="unknown", probability=0.0005, rank=50, log_probability=-7.6)]
        scores = head.score_candidates(context=(), candidates=candidates)

        assert scores["unknown"].raw_score == 0.0005
        assert "Low probability" in scores["unknown"].explanation
