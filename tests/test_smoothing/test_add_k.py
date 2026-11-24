"""Unit tests for Add-k probability smoothing across multiple k values."""

import pytest

from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.smoothing.add_k import AddKSmoother


@pytest.fixture
def synthetic_corpus():
    # 3 sentences, 11 total tokens. |V| = 6
    # a: 3, b: 2, c: 1, d: 1, e: 1, .: 3
    return [
        ["a", "b", "c", "."],
        ["a", "b", "d", "."],
        ["a", "e", "."],
    ]


class TestAddKSmootherSynthetic:
    """Mathematical verification of Add-k smoothing."""

    def test_invalid_k_raises_value_error(self, synthetic_corpus):
        model = BigramModel()
        model.fit(synthetic_corpus)
        with pytest.raises(ValueError, match="k must be strictly positive"):
            AddKSmoother(k=0.0, model=model)
        with pytest.raises(ValueError, match="k must be strictly positive"):
            AddKSmoother(k=-0.5, model=model)

    def test_k_one_matches_laplace_exactly(self, synthetic_corpus):
        model = BigramModel()
        model.fit(synthetic_corpus)
        add_one = AddKSmoother(k=1.0, model=model)

        # Context 'a': C(a)=3, C(a, b)=2. (2 + 1) / (3 + 6) = 3/9 = 1/3
        assert pytest.approx(add_one.probability("b", context="a"), rel=1e-5) == 1 / 3
        # Unseen transition: (0 + 1) / (3 + 6) = 1/9
        assert pytest.approx(add_one.probability("c", context="a"), rel=1e-5) == 1 / 9

    def test_add_k_math_for_k_01(self, synthetic_corpus):
        model = BigramModel()
        model.fit(synthetic_corpus)
        add_01 = AddKSmoother(k=0.1, model=model)

        # Context 'a': C(a)=3, |V|=6. Normalizer = 3 + 0.1 * 6 = 3.6
        # C(a, b) = 2 -> (2 + 0.1) / 3.6 = 2.1 / 3.6 = 7 / 12
        assert pytest.approx(add_01.probability("b", context="a"), rel=1e-5) == 2.1 / 3.6
        # C(a, e) = 1 -> (1 + 0.1) / 3.6 = 1.1 / 3.6 = 11 / 36
        assert pytest.approx(add_01.probability("e", context="a"), rel=1e-5) == 1.1 / 3.6
        # C(a, c) = 0 -> (0 + 0.1) / 3.6 = 0.1 / 3.6 = 1 / 36
        assert pytest.approx(add_01.probability("c", context="a"), rel=1e-5) == 0.1 / 3.6

    def test_comparison_across_k_values(self, synthetic_corpus):
        model = BigramModel()
        model.fit(synthetic_corpus)

        k_01 = AddKSmoother(k=0.1, model=model)
        k_05 = AddKSmoother(k=0.5, model=model)
        k_10 = AddKSmoother(k=1.0, model=model)

        # As k decreases, observed events retain higher probability, closer to MLE
        p_01_seen = k_01.probability("b", context="a")  # 2.1 / 3.6 = 0.5833
        p_05_seen = k_05.probability("b", context="a")  # 2.5 / 6.0 = 0.4167
        p_10_seen = k_10.probability("b", context="a")  # 3.0 / 9.0 = 0.3333

        assert p_01_seen > p_05_seen > p_10_seen

        # For unseen events, smaller k allocates less probability mass
        p_01_unseen = k_01.probability("c", context="a")  # 0.1 / 3.6 = 0.0278
        p_05_unseen = k_05.probability("c", context="a")  # 0.5 / 6.0 = 0.0833
        p_10_unseen = k_10.probability("c", context="a")  # 1.0 / 9.0 = 0.1111

        assert p_01_unseen < p_05_unseen < p_10_unseen

    def test_probability_simplex_normalization(self, synthetic_corpus):
        model = BigramModel()
        model.fit(synthetic_corpus)

        for k in [0.01, 0.1, 0.5, 1.0, 2.0]:
            smoother = AddKSmoother(k=k, model=model)
            dist = smoother.get_distribution(context="a")
            assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0
