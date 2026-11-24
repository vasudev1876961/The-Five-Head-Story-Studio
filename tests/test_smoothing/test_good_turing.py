"""Unit tests for Good-Turing frequency estimation and probability smoothing."""

import pytest

from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.smoothing.good_turing import GoodTuringSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


@pytest.fixture
def synthetic_corpus():
    # Sentence 1: a b c .
    # Sentence 2: a b d .
    # Sentence 3: a e .
    return [
        ["a", "b", "c", "."],
        ["a", "b", "d", "."],
        ["a", "e", "."],
    ]


class TestGoodTuringSmootherSynthetic:
    """Mathematical verification of Good-Turing estimator."""

    def test_frequency_of_frequencies_table(self, synthetic_corpus):
        model = BigramModel()
        model.fit(synthetic_corpus)
        gt = GoodTuringSmoother(k_threshold=5, model=model)

        # Unigrams: a:3, b:2, c:1, d:1, e:1, .:3
        # Counts: 3 occurs 2 times (a, .); 2 occurs 1 time (b); 1 occurs 3 times (c, d, e)
        fof_uni = gt.get_frequency_of_frequencies(order=1)
        assert fof_uni == {1: 3, 2: 1, 3: 2}

    def test_adjusted_counts_calculation(self, synthetic_corpus):
        model = BigramModel()
        model.fit(synthetic_corpus)
        gt = GoodTuringSmoother(k_threshold=5, model=model)

        # r=1: N_1 = 3, N_2 = 1.
        # r* = (1 + 1) * N_2 / N_1 = 2 * 1 / 3 = 2/3
        adj_1 = gt.get_adjusted_count(1, order=1)
        assert pytest.approx(adj_1, rel=1e-5) == 2 / 3

        # r=2: N_2 = 1, N_3 = 2.
        # r* = (2 + 1) * N_3 / N_2 = 3 * 2 / 1 = 6
        adj_2 = gt.get_adjusted_count(2, order=1)
        assert pytest.approx(adj_2, rel=1e-5) == 6.0

    def test_unseen_mass_calculation(self, synthetic_corpus):
        model = BigramModel()
        model.fit(synthetic_corpus)
        gt = GoodTuringSmoother(k_threshold=5, model=model)

        # N_total = 11. N_1 = 3.
        # P_0 = N_1 / N = 3 / 11
        p_0 = gt.get_unseen_mass(order=1)
        assert pytest.approx(p_0, rel=1e-5) == 3 / 11

    def test_distribution_sums_to_one(self, synthetic_corpus):
        model = BigramModel()
        model.fit(synthetic_corpus)
        gt = GoodTuringSmoother(k_threshold=5, model=model)

        dist_a = gt.get_distribution(context="a")
        assert len(dist_a) == 6
        assert pytest.approx(sum(dist_a.values()), rel=1e-5) == 1.0


class TestGoodTuringSmootherCorpusIntegration:
    """Integration test with the 34-sentence story corpus."""

    @pytest.fixture(scope="class")
    def corpus_gt(self):
        loader = CorpusLoader()
        model = TrigramModel()
        model.fit_corpus(loader)
        return GoodTuringSmoother(k_threshold=5, model=model)

    def test_fof_on_real_corpus(self, corpus_gt):
        # From Phase 2.4 / 3.1: 57 hapax legomena unigrams (occur exactly once)
        fof_uni = corpus_gt.get_frequency_of_frequencies(order=1)
        assert fof_uni[1] == 57

    def test_probability_mass_conservation_real_corpus(self, corpus_gt):
        for ctx in [(), ("crystal",), ("the", "dark")]:
            dist = corpus_gt.get_distribution(context=ctx)
            assert len(dist) == 120
            assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0
