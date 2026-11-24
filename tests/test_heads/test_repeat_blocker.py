"""Unit tests for Head 5 — Repeat Blocker."""

import math
import pytest

from five_head_nlp.heads.repeat_blocker import RepeatBlocker
from five_head_nlp.prediction.prediction_result import CandidateToken


class TestRepeatBlocker:
    """Test Repeat Blocker loop suppression, decay calculation, and function-word exemptions."""

    def test_exempt_tokens_not_penalized(self):
        head = RepeatBlocker()
        # Even if 'the' appeared multiple times, exempt tokens receive full score 1.0
        head.history = ["the", "the", "the"]
        candidates = [CandidateToken("the", 0.5, 1, -0.69), CandidateToken(".", 0.5, 2, -0.69)]
        scores = head.score_candidates(context=("the",), candidates=candidates)

        assert scores["the"].raw_score == 1.0
        assert scores["."].raw_score == 1.0
        assert "exempt" in scores["the"].explanation

    def test_immediate_token_self_loop_penalized(self):
        head = RepeatBlocker()
        candidates = [CandidateToken("castle", 0.8, 1, -0.22)]
        # Context ends in 'castle', proposing 'castle'
        scores = head.score_candidates(context=("the", "castle"), candidates=candidates)

        assert scores["castle"].raw_score == 0.05
        assert "Immediate self-loop detected" in scores["castle"].explanation

    def test_repeated_bigram_penalized(self):
        head = RepeatBlocker()
        # Window contains bigram ('the', 'crystal')
        head.history = ["the", "crystal", "castle", "near", "the"]
        candidates = [CandidateToken("crystal", 0.8, 1, -0.22)]

        # Context ends in 'the', proposing 'crystal' which creates duplicate bigram ('the', 'crystal')
        scores = head.score_candidates(context=("near", "the"), candidates=candidates)
        assert scores["crystal"].raw_score == 0.20
        assert "Repeated bigram loop" in scores["crystal"].explanation

    def test_exponential_frequency_decay(self):
        head = RepeatBlocker()
        head.history = ["key", "silver", "key", "box"]
        candidates = [CandidateToken("key", 0.5, 1, -0.69), CandidateToken("magic", 0.5, 2, -0.69)]

        scores = head.score_candidates(context=("found", "a"), candidates=candidates)

        # 'key' appeared twice in window -> raw score = exp(-0.8 * 2) = exp(-1.6)
        expected_decay = math.exp(-0.8 * 2)
        assert pytest.approx(scores["key"].raw_score, rel=1e-5) == expected_decay
        assert "Frequency penalty" in scores["key"].explanation

        # 'magic' appeared 0 times -> fresh word = 1.0
        assert scores["magic"].raw_score == 1.0
        assert "Fresh unrepeated token" in scores["magic"].explanation
