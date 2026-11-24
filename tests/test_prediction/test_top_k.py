"""Unit tests for TopKDecoder."""

import pytest

from five_head_nlp.prediction.prediction_result import CandidateToken
from five_head_nlp.prediction.top_k import TopKDecoder


class TestTopKDecoder:
    """Test top-k candidate truncation and stochastic sampling."""

    def test_invalid_k_raises_error(self):
        with pytest.raises(ValueError, match="k must be an integer >= 1"):
            TopKDecoder(k=0)
        with pytest.raises(ValueError, match="k must be an integer >= 1"):
            TopKDecoder(k=-5)

    def test_samples_only_from_top_k(self):
        decoder = TopKDecoder(k=2, seed=42)
        candidates = [
            CandidateToken("a", 0.5, 1, -0.69),
            CandidateToken("b", 0.3, 2, -1.20),
            CandidateToken("c", 0.15, 3, -1.89),
            CandidateToken("d", 0.05, 4, -2.99),
        ]

        # Draw 50 samples; every sample must be in {'a', 'b'}
        samples = [decoder.select_token(candidates) for _ in range(50)]
        assert set(samples).issubset({"a", "b"})
        assert "c" not in samples
        assert "d" not in samples

    def test_seed_reproducibility(self):
        candidates = [
            CandidateToken("a", 0.5, 1, -0.69),
            CandidateToken("b", 0.3, 2, -1.20),
            CandidateToken("c", 0.2, 3, -1.60),
        ]

        dec1 = TopKDecoder(k=3, seed=123)
        dec2 = TopKDecoder(k=3, seed=123)

        seq1 = [dec1.select_token(candidates) for _ in range(20)]
        seq2 = [dec2.select_token(candidates) for _ in range(20)]
        assert seq1 == seq2

    def test_k_larger_than_vocabulary_handled_cleanly(self):
        decoder = TopKDecoder(k=100, seed=42)
        candidates = [
            CandidateToken("apple", 0.6, 1, -0.51),
            CandidateToken("banana", 0.4, 2, -0.91),
        ]
        samples = [decoder.select_token(candidates) for _ in range(20)]
        assert set(samples).issubset({"apple", "banana"})

    def test_empty_candidates_raises_error(self):
        decoder = TopKDecoder(k=3)
        with pytest.raises(ValueError, match="empty candidates list"):
            decoder.select_token([])
