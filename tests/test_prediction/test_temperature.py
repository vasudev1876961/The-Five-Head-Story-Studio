"""Unit tests for TemperatureDecoder."""

import math
import pytest

from five_head_nlp.prediction.prediction_result import CandidateToken
from five_head_nlp.prediction.temperature import TemperatureDecoder


class TestTemperatureDecoder:
    """Test temperature-scaled probability sampling and numerical stability."""

    def test_invalid_temperature_raises_error(self):
        with pytest.raises(ValueError, match="strictly positive"):
            TemperatureDecoder(temperature=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            TemperatureDecoder(temperature=-1.5)

    def test_temperature_one_preserves_distribution(self):
        decoder = TemperatureDecoder(temperature=1.0)
        candidates = [
            CandidateToken("a", 0.6, 1, math.log(0.6)),
            CandidateToken("b", 0.4, 2, math.log(0.4)),
        ]
        scaled = decoder.compute_scaled_probabilities(candidates)
        assert pytest.approx(scaled[0], rel=1e-5) == 0.6
        assert pytest.approx(scaled[1], rel=1e-5) == 0.4

    def test_low_temperature_approaches_argmax(self):
        """As T -> 0, probability concentrates entirely on argmax."""
        decoder = TemperatureDecoder(temperature=0.01)
        candidates = [
            CandidateToken("a", 0.6, 1, math.log(0.6)),
            CandidateToken("b", 0.4, 2, math.log(0.4)),
        ]
        scaled = decoder.compute_scaled_probabilities(candidates)
        assert scaled[0] > 0.999
        assert scaled[1] < 0.001

    def test_high_temperature_flattens_distribution(self):
        """As T -> infty, distribution approaches uniform over candidates with P > 0."""
        decoder = TemperatureDecoder(temperature=100.0)
        candidates = [
            CandidateToken("a", 0.7, 1, math.log(0.7)),
            CandidateToken("b", 0.3, 2, math.log(0.3)),
        ]
        scaled = decoder.compute_scaled_probabilities(candidates)
        # Approaches 0.5, 0.5
        assert pytest.approx(scaled[0], abs=0.02) == 0.5
        assert pytest.approx(scaled[1], abs=0.02) == 0.5

    def test_zero_probability_candidates_remain_zero(self):
        decoder = TemperatureDecoder(temperature=2.0)
        candidates = [
            CandidateToken("a", 0.7, 1, math.log(0.7)),
            CandidateToken("b", 0.3, 2, math.log(0.3)),
            CandidateToken("c", 0.0, 3, -math.inf),
        ]
        scaled = decoder.compute_scaled_probabilities(candidates)
        assert scaled[2] == 0.0
        assert pytest.approx(sum(scaled), rel=1e-5) == 1.0

    def test_numerical_stability_extreme_logits(self):
        """Verify that extremely small probabilities do not cause overflow or crash."""
        decoder = TemperatureDecoder(temperature=0.5)
        candidates = [
            CandidateToken("a", 0.999999, 1, math.log(0.999999)),
            CandidateToken("b", 1e-12, 2, math.log(1e-12)),
            CandidateToken("c", 1e-15, 3, math.log(1e-15)),
        ]
        scaled = decoder.compute_scaled_probabilities(candidates)
        assert pytest.approx(sum(scaled), rel=1e-5) == 1.0
        assert math.isfinite(scaled[0])
        assert math.isfinite(scaled[1])

    def test_seed_reproducibility(self):
        candidates = [
            CandidateToken("a", 0.5, 1, math.log(0.5)),
            CandidateToken("b", 0.5, 2, math.log(0.5)),
        ]
        dec1 = TemperatureDecoder(temperature=1.0, seed=42)
        dec2 = TemperatureDecoder(temperature=1.0, seed=42)

        seq1 = [dec1.select_token(candidates) for _ in range(20)]
        seq2 = [dec2.select_token(candidates) for _ in range(20)]
        assert seq1 == seq2
