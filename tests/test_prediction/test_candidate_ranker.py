"""Unit tests for CandidateRanker."""

import math
import pytest

from five_head_nlp.prediction.candidate_ranker import CandidateRanker
from five_head_nlp.prediction.prediction_result import CandidateToken


class TestCandidateRanker:
    """Test candidate ranking, sorting, and tie-breaking."""

    def test_rank_candidates_descending_order(self):
        ranker = CandidateRanker()
        dist = {"castle": 0.5, "cave": 0.3, "garden": 0.2}
        ranked = ranker.rank_candidates(dist)

        assert len(ranked) == 3
        assert [c.token for c in ranked] == ["castle", "cave", "garden"]
        assert [c.rank for c in ranked] == [1, 2, 3]
        assert [c.probability for c in ranked] == [0.5, 0.3, 0.2]

    def test_rank_candidates_deterministic_tie_breaking(self):
        """Audit requirement: equal probabilities must break ties alphabetically."""
        ranker = CandidateRanker()
        dist = {"zebra": 0.2, "apple": 0.4, "banana": 0.4}
        ranked = ranker.rank_candidates(dist)

        assert [c.token for c in ranked] == ["apple", "banana", "zebra"]
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[2].rank == 3

    def test_rank_candidates_top_n(self):
        ranker = CandidateRanker()
        dist = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
        ranked = ranker.rank_candidates(dist, top_n=2)

        assert len(ranked) == 2
        assert [c.token for c in ranked] == ["a", "b"]

    def test_rank_candidates_empty_distribution(self):
        ranker = CandidateRanker()
        assert ranker.rank_candidates({}) == []

    def test_preserves_original_probabilities_and_logs(self):
        ranker = CandidateRanker()
        dist = {"castle": 0.65}
        ranked = ranker.rank_candidates(dist)

        assert ranked[0].probability == 0.65
        assert math.isclose(ranked[0].log_probability, math.log(0.65), rel_tol=1e-9)

    def test_normalize_distribution(self):
        ranker = CandidateRanker()
        unnormalized = {"a": 2.0, "b": 3.0}
        norm = ranker.normalize_distribution(unnormalized)

        assert math.isclose(norm["a"], 0.4, rel_tol=1e-9)
        assert math.isclose(norm["b"], 0.6, rel_tol=1e-9)
        assert math.isclose(sum(norm.values()), 1.0, rel_tol=1e-9)

    def test_normalize_distribution_invalid_sum_raises_error(self):
        ranker = CandidateRanker()
        with pytest.raises(ValueError, match="non-positive sum"):
            ranker.normalize_distribution({"a": 0.0, "b": 0.0})
