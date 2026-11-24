"""Unit tests for BigramModel implementation and mathematical verification."""

import math
import pytest

from five_head_nlp.models.bigram import BigramModel
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


@pytest.fixture
def trained_bigram(synthetic_corpus):
    model = BigramModel()
    model.fit(synthetic_corpus)
    return model


class TestBigramModelSynthetic:
    """Test BigramModel against synthetic expectations."""

    def test_basic_properties(self, trained_bigram):
        assert trained_bigram.name == "BigramModel"
        assert trained_bigram.n == 2
        assert trained_bigram.is_fitted
        assert trained_bigram.total_tokens == 11
        assert trained_bigram.vocabulary_size == 6

    def test_conditional_probabilities(self, trained_bigram):
        # 'a' followed by 'b' (2 times) and 'e' (1 time). Total context = 3
        assert pytest.approx(trained_bigram.probability("b", context="a"), rel=1e-5) == 2 / 3
        assert pytest.approx(trained_bigram.probability("e", context="a"), rel=1e-5) == 1 / 3
        assert trained_bigram.probability("c", context="a") == 0.0

        # Also accepts tuple context
        assert pytest.approx(trained_bigram.probability("b", context=("a",)), rel=1e-5) == 2 / 3

        # Fallback to unigram when context is empty
        assert pytest.approx(trained_bigram.probability("a"), rel=1e-5) == 3 / 11

    def test_sentence_boundary_isolation(self, trained_bigram):
        # Period '.' has zero outgoing transitions
        assert trained_bigram.probability("a", context=".") == 0.0
        assert trained_bigram.get_candidates(context=".") == []
        assert trained_bigram.get_candidate_distribution(context=".") == {}

    def test_candidate_retrieval(self, trained_bigram):
        candidates = trained_bigram.get_candidates(context="a")
        # 'b' (count 2), 'e' (count 1)
        assert candidates == ["b", "e"]

        # Unknown context returns empty list
        assert trained_bigram.get_candidates(context="unknown") == []

    def test_candidate_distribution_sums_to_one(self, trained_bigram):
        dist = trained_bigram.get_candidate_distribution(context="a")
        assert len(dist) == 2
        assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0

        dist_b = trained_bigram.get_candidate_distribution(context="b")
        assert len(dist_b) == 2
        assert pytest.approx(sum(dist_b.values()), rel=1e-5) == 1.0

    def test_unknown_tokens_and_contexts(self, trained_bigram):
        assert trained_bigram.probability("unknown", context="a") == 0.0
        assert trained_bigram.probability("b", context="unknown") == 0.0

    def test_sequence_probability(self, trained_bigram):
        # P("a", "b", "c") = P("a") * P("b"|"a") * P("c"|"b")
        # = (3/11) * (2/3) * (1/2) = 1/11
        prob = trained_bigram.sequence_probability(["a", "b", "c"])
        assert pytest.approx(prob, rel=1e-5) == 1 / 11

        # Impossible transition gives 0.0
        assert trained_bigram.sequence_probability(["b", "a"]) == 0.0
        assert trained_bigram.sequence_probability([]) == 1.0


class TestBigramModelCorpusIntegration:
    """Test BigramModel with the reconstructed 34-sentence story corpus."""

    @pytest.fixture(scope="class")
    def corpus_bigram(self):
        loader = CorpusLoader()
        model = BigramModel()
        model.fit_corpus(loader)
        return model

    def test_known_bigram_transitions(self, corpus_bigram):
        # Deterministic settings:
        assert corpus_bigram.probability("castle", context="crystal") == 1.0
        assert corpus_bigram.probability("cave", context="dark") == 1.0
        assert corpus_bigram.probability("garden", context="secret") == 1.0

        assert corpus_bigram.get_candidates(context="crystal") == ["castle"]
        assert corpus_bigram.get_candidates(context="dark") == ["cave"]
        assert corpus_bigram.get_candidates(context="secret") == ["garden"]

    def test_period_never_has_outgoing_transitions(self, corpus_bigram):
        assert corpus_bigram.get_candidates(context=".") == []
        assert corpus_bigram.get_candidate_distribution(context=".") == {}
        assert corpus_bigram.probability("the", context=".") == 0.0
