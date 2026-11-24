"""Unit tests for TrigramModel implementation and mathematical verification."""

import math
import pytest

from five_head_nlp.models.trigram import TrigramModel
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
def trained_trigram(synthetic_corpus):
    model = TrigramModel()
    model.fit(synthetic_corpus)
    return model


class TestTrigramModelSynthetic:
    """Test TrigramModel against synthetic expectations."""

    def test_basic_properties(self, trained_trigram):
        assert trained_trigram.name == "TrigramModel"
        assert trained_trigram.n == 3
        assert trained_trigram.is_fitted
        assert trained_trigram.total_tokens == 11
        assert trained_trigram.vocabulary_size == 6

    def test_trigram_conditional_probabilities(self, trained_trigram):
        # Context ('a', 'b') followed by 'c' (1 time), 'd' (1 time). Total = 2
        assert pytest.approx(trained_trigram.probability("c", context=("a", "b")), rel=1e-5) == 1 / 2
        assert pytest.approx(trained_trigram.probability("d", context=("a", "b")), rel=1e-5) == 1 / 2
        assert trained_trigram.probability("e", context=("a", "b")) == 0.0

        # Context ('a', 'e') followed by '.' (1 time)
        assert pytest.approx(trained_trigram.probability(".", context=("a", "e")), rel=1e-5) == 1.0

    def test_fallback_on_shorter_contexts(self, trained_trigram):
        # Length 1 context falls back to bigram
        assert pytest.approx(trained_trigram.probability("b", context="a"), rel=1e-5) == 2 / 3
        # Length 0 context falls back to unigram
        assert pytest.approx(trained_trigram.probability("a"), rel=1e-5) == 3 / 11

    def test_candidate_retrieval(self, trained_trigram):
        candidates = trained_trigram.get_candidates(context=("a", "b"))
        assert candidates == ["c", "d"]

        # Unknown context returns []
        assert trained_trigram.get_candidates(context=("x", "y")) == []

    def test_candidate_distribution_sums_to_one(self, trained_trigram):
        dist = trained_trigram.get_candidate_distribution(context=("a", "b"))
        assert len(dist) == 2
        assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0
        assert pytest.approx(dist["c"], rel=1e-5) == 0.5
        assert pytest.approx(dist["d"], rel=1e-5) == 0.5

    def test_sequence_probability(self, trained_trigram):
        # P("a", "b", "c") = P("a") * P("b"|"a") * P("c"|"a", "b")
        # = (3/11) * (2/3) * (1/2) = 1/11
        prob = trained_trigram.sequence_probability(["a", "b", "c"])
        assert pytest.approx(prob, rel=1e-5) == 1 / 11

        assert trained_trigram.sequence_probability(["a", "b", "e"]) == 0.0
        assert trained_trigram.sequence_probability([]) == 1.0


class TestTrigramModelCorpusIntegration:
    """Test TrigramModel with the reconstructed 34-sentence story corpus."""

    @pytest.fixture(scope="class")
    def corpus_trigram(self):
        loader = CorpusLoader()
        model = TrigramModel()
        model.fit_corpus(loader)
        return model

    def test_known_trigram_transitions(self, corpus_trigram):
        # "the crystal castle" -> ("the", "crystal") followed deterministically by "castle"
        p_castle = corpus_trigram.probability("castle", context=("the", "crystal"))
        assert p_castle == 1.0

        p_cave = corpus_trigram.probability("cave", context=("the", "dark"))
        assert p_cave == 1.0

        candidates = corpus_trigram.get_candidates(context=("the", "crystal"))
        assert candidates == ["castle"]

    def test_period_never_has_outgoing_transitions(self, corpus_trigram):
        # Period ending context e.g. ("castle", ".")
        assert corpus_trigram.get_candidates(context=("castle", ".")) == []
        assert corpus_trigram.probability("milo", context=("castle", ".")) == 0.0
