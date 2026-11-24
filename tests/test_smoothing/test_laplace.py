"""Unit tests for Laplace (Add-One) probability smoothing."""

import pytest

from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.smoothing.laplace import LaplaceSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


@pytest.fixture
def synthetic_corpus():
    # 3 sentences, 11 total tokens
    # a: 3, b: 2, c: 1, d: 1, e: 1, .: 3. |V| = 6
    return [
        ["a", "b", "c", "."],
        ["a", "b", "d", "."],
        ["a", "e", "."],
    ]


@pytest.fixture
def trained_laplace(synthetic_corpus):
    model = BigramModel()
    model.fit(synthetic_corpus)
    return LaplaceSmoother(model=model)


class TestLaplaceSmootherSynthetic:
    """Hand-calculated mathematical verification of Laplace smoothing."""

    def test_unigram_laplace_math(self, trained_laplace):
        # N = 11, |V| = 6. Normalizer = 11 + 6 = 17
        # C(a) = 3 -> (3 + 1) / 17 = 4 / 17
        # C(b) = 2 -> (2 + 1) / 17 = 3 / 17
        # C(c) = 1 -> (1 + 1) / 17 = 2 / 17
        assert pytest.approx(trained_laplace.probability("a", context=()), rel=1e-5) == 4 / 17
        assert pytest.approx(trained_laplace.probability("b", context=()), rel=1e-5) == 3 / 17
        assert pytest.approx(trained_laplace.probability("c", context=()), rel=1e-5) == 2 / 17

        # Unknown token outside vocabulary returns 0.0
        assert trained_laplace.probability("unknown") == 0.0

    def test_conditional_laplace_math(self, trained_laplace):
        # Context h = 'a'. C(a) = 3. |V| = 6. Normalizer = 3 + 6 = 9
        # C(a, b) = 2 -> (2 + 1) / 9 = 3 / 9 = 1 / 3
        # C(a, e) = 1 -> (1 + 1) / 9 = 2 / 9
        # C(a, c) = 0 -> (0 + 1) / 9 = 1 / 9 (unseen transition gets positive probability!)
        assert pytest.approx(trained_laplace.probability("b", context="a"), rel=1e-5) == 3 / 9
        assert pytest.approx(trained_laplace.probability("e", context="a"), rel=1e-5) == 2 / 9
        assert pytest.approx(trained_laplace.probability("c", context="a"), rel=1e-5) == 1 / 9
        assert pytest.approx(trained_laplace.probability("d", context="a"), rel=1e-5) == 1 / 9
        assert pytest.approx(trained_laplace.probability("a", context="a"), rel=1e-5) == 1 / 9
        assert pytest.approx(trained_laplace.probability(".", context="a"), rel=1e-5) == 1 / 9

    def test_distribution_sums_to_one(self, trained_laplace):
        # Sum of probabilities across vocabulary for context 'a' must be exactly 1.0
        dist_a = trained_laplace.get_distribution(context="a")
        assert len(dist_a) == 6
        assert pytest.approx(sum(dist_a.values()), rel=1e-5) == 1.0

        # Also for unigram
        dist_uni = trained_laplace.get_distribution(context=())
        assert pytest.approx(sum(dist_uni.values()), rel=1e-5) == 1.0

    def test_unseen_context_handling(self, trained_laplace):
        # Unseen context 'x' has count 0.
        # For every token in V: (0 + 1) / (0 + 6) = 1/6 (uniform distribution)
        assert pytest.approx(trained_laplace.probability("a", context="x"), rel=1e-5) == 1 / 6
        dist_x = trained_laplace.get_distribution(context="x")
        assert pytest.approx(sum(dist_x.values()), rel=1e-5) == 1.0


class TestLaplaceSmootherCorpusIntegration:
    """Integration test with the 34-sentence story corpus."""

    @pytest.fixture(scope="class")
    def corpus_laplace(self):
        loader = CorpusLoader()
        model = TrigramModel()
        model.fit_corpus(loader)
        return LaplaceSmoother(model=model)

    def test_zero_probability_eliminated(self, corpus_laplace):
        # In unsmoothed MLE, ("crystal", "cave") has probability 0.0
        # In Laplace, it has non-zero probability
        p_unseen_trans = corpus_laplace.probability("cave", context="crystal")
        assert p_unseen_trans > 0.0

    def test_simplex_normalization_on_real_corpus(self, corpus_laplace):
        for ctx in [(), ("crystal",), ("the", "dark"), ("near", "the")]:
            dist = corpus_laplace.get_distribution(context=ctx)
            assert len(dist) == 120
            assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0
