"""Unit tests for GreedyDecoder."""

import pytest

from five_head_nlp.prediction.greedy import GreedyDecoder
from five_head_nlp.prediction.prediction_result import CandidateToken


class TestGreedyDecoder:
    """Test deterministic argmax greedy decoding."""

    def test_greedy_selection_argmax(self):
        decoder = GreedyDecoder()
        candidates = [
            CandidateToken("castle", 0.65, 1, -0.43),
            CandidateToken("cave", 0.25, 2, -1.38),
            CandidateToken("garden", 0.10, 3, -2.30),
        ]
        assert decoder.select_token(candidates) == "castle"

    def test_greedy_deterministic_identical_output(self):
        decoder = GreedyDecoder()
        candidates = [
            CandidateToken("castle", 0.65, 1, -0.43),
            CandidateToken("cave", 0.25, 2, -1.38),
        ]
        # Multiple calls return identical token
        for _ in range(10):
            assert decoder.select_token(candidates) == "castle"

    def test_greedy_empty_candidates_raises_error(self):
        decoder = GreedyDecoder()
        with pytest.raises(ValueError, match="empty candidates list"):
            decoder.select_token([])
