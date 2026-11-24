"""Unit tests for UnigramModel implementation and mathematical verification."""

import math
import pytest

from five_head_nlp.models.unigram import UnigramModel
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


@pytest.fixture
def synthetic_corpus():
    # 3 sentences, 11 total tokens
    # a: 3, b: 2, c: 1, d: 1, e: 1, .: 3
    return [
        ["a", "b", "c", "."],
        ["a", "b", "d", "."],
        ["a", "e", "."],
    ]


@pytest.fixture
def trained_unigram(synthetic_corpus):
    model = UnigramModel()
    model.fit(synthetic_corpus)
    return model


class TestUnigramModelSynthetic:
    """Test UnigramModel against hand-calculated synthetic expected values."""

    def test_basic_properties(self, trained_unigram):
        assert trained_unigram.name == "UnigramModel"
        assert trained_unigram.n == 1
        assert trained_unigram.is_fitted
        assert trained_unigram.total_tokens == 11
        assert trained_unigram.vocabulary_size == 6

    def test_token_probabilities(self, trained_unigram):
        assert pytest.approx(trained_unigram.probability("a"), rel=1e-5) == 3 / 11
        assert pytest.approx(trained_unigram.probability("b"), rel=1e-5) == 2 / 11
        assert pytest.approx(trained_unigram.probability("c"), rel=1e-5) == 1 / 11
        assert pytest.approx(trained_unigram.probability("d"), rel=1e-5) == 1 / 11
        assert pytest.approx(trained_unigram.probability("e"), rel=1e-5) == 1 / 11
        assert pytest.approx(trained_unigram.probability("."), rel=1e-5) == 3 / 11

        # Unknown token returns 0.0
        assert trained_unigram.probability("unknown") == 0.0

    def test_context_independence(self, trained_unigram):
        # Unigram probability ignores context
        p_base = trained_unigram.probability("a")
        p_ctx = trained_unigram.probability("a", context="b")
        p_multi_ctx = trained_unigram.probability("a", context=["b", "c"])
        assert p_base == p_ctx == p_multi_ctx

    def test_candidate_ordering(self, trained_unigram):
        candidates = trained_unigram.get_candidates()
        # a (3), . (3) -> tie break alphabetical -> . then a
        # b (2)
        # c (1), d (1), e (1) -> tie break -> c, d, e
        assert candidates[0:2] == [".", "a"]
        assert candidates[2] == "b"
        assert candidates[3:6] == ["c", "d", "e"]

    def test_candidate_distribution_sums_to_one(self, trained_unigram):
        dist = trained_unigram.get_candidate_distribution()
        assert len(dist) == 6
        assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0

    def test_sequence_probability(self, trained_unigram):
        # P("a", "b", "c") = (3/11) * (2/11) * (1/11) = 6 / 1331
        prob = trained_unigram.sequence_probability(["a", "b", "c"])
        assert pytest.approx(prob, rel=1e-5) == 6 / 1331

        # Sequence with unknown token has probability 0.0
        assert trained_unigram.sequence_probability(["a", "unknown"]) == 0.0
        # Empty sequence has probability 1.0
        assert trained_unigram.sequence_probability([]) == 1.0

    def test_log_probability(self, trained_unigram):
        log_p = trained_unigram.log_probability("a", base=2.0)
        assert pytest.approx(log_p, rel=1e-5) == math.log2(3 / 11)

        assert trained_unigram.log_probability("unknown") == -math.inf


class TestUnigramModelCorpusIntegration:
    """Test UnigramModel with the reconstructed 34-sentence story corpus."""

    @pytest.fixture(scope="class")
    def corpus_unigram(self):
        loader = CorpusLoader()
        model = UnigramModel()
        model.fit_corpus(loader)
        return model

    def test_corpus_dimensions(self, corpus_unigram):
        assert corpus_unigram.total_tokens == 415
        assert corpus_unigram.vocabulary_size == 120

    def test_distribution_sums_to_one_on_real_corpus(self, corpus_unigram):
        dist = corpus_unigram.get_candidate_distribution()
        assert len(dist) == 120
        assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0

    def test_most_frequent_word_is_the(self, corpus_unigram):
        assert corpus_unigram.get_candidates()[0] == "the"
        assert pytest.approx(corpus_unigram.probability("the"), rel=1e-5) == 63 / 415
