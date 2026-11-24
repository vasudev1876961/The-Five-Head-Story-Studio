"""Comprehensive unit and integration tests for the N-Gram Probability Engine."""

import math
import pytest
from pathlib import Path

from five_head_nlp.probability.mle import (
    NGramProbabilityEngine,
    extract_bigrams,
    extract_ngrams,
    extract_sentence_ngrams,
    extract_trigrams,
    extract_unigrams,
)
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


class TestNGramExtraction:
    """Tests for sequence-level and corpus-level N-gram extraction."""

    def test_extract_ngrams_basic(self):
        tokens = ["pip", "flew", "to", "forest", "."]
        assert extract_ngrams(tokens, 1) == [
            ("pip",),
            ("flew",),
            ("to",),
            ("forest",),
            (".",),
        ]
        assert extract_ngrams(tokens, 2) == [
            ("pip", "flew"),
            ("flew", "to"),
            ("to", "forest"),
            ("forest", "."),
        ]
        assert extract_ngrams(tokens, 3) == [
            ("pip", "flew", "to"),
            ("flew", "to", "forest"),
            ("to", "forest", "."),
        ]

    def test_extract_ngrams_shorter_than_n(self):
        tokens = ["pip", "flew"]
        assert extract_ngrams(tokens, 3) == []
        assert extract_ngrams([], 1) == []

    def test_extract_ngrams_invalid_n(self):
        with pytest.raises(ValueError, match="order must be an integer >= 1"):
            extract_ngrams(["a", "b"], 0)
        with pytest.raises(ValueError, match="order must be an integer >= 1"):
            extract_sentence_ngrams([["a", "b"]], -1)

    def test_sentence_boundary_isolation_strictly_enforced(self):
        """Crucial test: ensure no n-gram bridges across sentence boundaries."""
        sentences = [
            ["pip", "slept", "."],
            ["milo", "jumped", "."],
        ]

        bigrams = extract_sentence_ngrams(sentences, 2)
        assert len(bigrams) == 4
        assert bigrams == [
            ("pip", "slept"),
            ("slept", "."),
            ("milo", "jumped"),
            ("jumped", "."),
        ]

        # Verify cross-sentence bigram does NOT exist
        assert (".", "milo") not in bigrams

        trigrams = extract_sentence_ngrams(sentences, 3)
        assert len(trigrams) == 2
        assert trigrams == [
            ("pip", "slept", "."),
            ("milo", "jumped", "."),
        ]

        # Verify cross-sentence trigrams do NOT exist
        assert ("slept", ".", "milo") not in trigrams
        assert (".", "milo", "jumped") not in trigrams

    def test_extraction_convenience_helpers(self):
        sentences = [
            ["pip", "flew", "."],
            ["zara", "smiled", "."],
        ]
        unigrams = extract_unigrams(sentences)
        assert len(unigrams) == 6
        assert unigrams[0] == ("pip",)

        bigrams = extract_bigrams(sentences)
        assert len(bigrams) == 4
        assert bigrams == [
            ("pip", "flew"),
            ("flew", "."),
            ("zara", "smiled"),
            ("smiled", "."),
        ]

        trigrams = extract_trigrams(sentences)
        assert len(trigrams) == 2
        assert trigrams == [
            ("pip", "flew", "."),
            ("zara", "smiled", "."),
        ]


class TestNGramProbabilityEngineSynthetic:
    """Mathematically verified tests on synthetic toy corpora."""

    @pytest.fixture
    def synthetic_corpus(self):
        # 3 sentences, 11 total tokens:
        # Sentence 1: a b c . (4 tokens, 3 bigrams, 2 trigrams)
        # Sentence 2: a b d . (4 tokens, 3 bigrams, 2 trigrams)
        # Sentence 3: a e .   (3 tokens, 2 bigrams, 1 trigram)
        return [
            ["a", "b", "c", "."],
            ["a", "b", "d", "."],
            ["a", "e", "."],
        ]

    @pytest.fixture
    def engine(self, synthetic_corpus):
        engine = NGramProbabilityEngine(n=3, include_lower_orders=True)
        engine.fit(synthetic_corpus)
        return engine

    def test_basic_dimensions(self, engine):
        assert engine.is_fitted
        assert engine.total_tokens == 11
        assert engine.vocabulary == {"a", "b", "c", "d", "e", "."}
        assert engine.vocabulary_size == 6
        assert engine.get_total_ngrams(order=1) == 11
        assert engine.get_total_ngrams(order=2) == 8
        assert engine.get_total_ngrams(order=3) == 5

    def test_unigram_probabilities(self, engine):
        # Counts: a:3, b:2, c:1, d:1, e:1, .:3. Total: 11
        assert engine.get_unigram_count("a") == 3
        assert engine.get_unigram_count("b") == 2
        assert engine.get_unigram_count("c") == 1
        assert engine.get_unigram_count(".") == 3
        assert engine.get_unigram_count("unseen") == 0

        assert pytest.approx(engine.p_unigram("a"), rel=1e-5) == 3 / 11
        assert pytest.approx(engine.p_unigram("b"), rel=1e-5) == 2 / 11
        assert pytest.approx(engine.p_unigram("c"), rel=1e-5) == 1 / 11
        assert pytest.approx(engine.p_unigram("."), rel=1e-5) == 3 / 11
        assert engine.p_unigram("unseen") == 0.0

    def test_bigram_conditional_probabilities(self, engine):
        # Context 'a' followed by: 'b' (2 times), 'e' (1 time). Total context count = 3
        assert engine.get_context_count("a", order=2) == 3
        assert engine.get_count(("a", "b")) == 2
        assert engine.get_count(("a", "e")) == 1
        assert engine.get_count(("a", "c")) == 0

        assert pytest.approx(engine.p_bigram("b", "a"), rel=1e-5) == 2 / 3
        assert pytest.approx(engine.p_bigram("e", "a"), rel=1e-5) == 1 / 3
        assert engine.p_bigram("c", "a") == 0.0

        # Candidates for 'a'
        candidates = engine.get_candidates("a")
        assert candidates == ["b", "e"]

        # Distribution sums to 1.0
        dist = engine.get_candidate_distribution("a")
        assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0
        assert pytest.approx(dist["b"], rel=1e-5) == 2 / 3
        assert pytest.approx(dist["e"], rel=1e-5) == 1 / 3

        # Context 'b' followed by: 'c' (1 time), 'd' (1 time). Total context count = 2
        assert engine.get_context_count("b", order=2) == 2
        assert pytest.approx(engine.p_bigram("c", "b"), rel=1e-5) == 1 / 2
        assert pytest.approx(engine.p_bigram("d", "b"), rel=1e-5) == 1 / 2

    def test_period_has_zero_outgoing_transitions(self, engine):
        # Sentence-ending period is never followed by any token intra-sentence
        assert engine.get_context_count(".", order=2) == 0
        assert engine.get_candidates(".") == []
        assert engine.get_candidate_distribution(".") == {}
        assert engine.p_bigram("a", ".") == 0.0

    def test_trigram_conditional_probabilities(self, engine):
        # Context ('a', 'b') followed by: 'c' (1 time), 'd' (1 time). Total context count = 2
        assert engine.get_context_count(("a", "b"), order=3) == 2
        assert engine.get_count(("a", "b", "c")) == 1
        assert engine.get_count(("a", "b", "d")) == 1
        assert engine.get_count(("a", "b", "e")) == 0

        assert pytest.approx(engine.p_trigram("c", "a", "b"), rel=1e-5) == 1 / 2
        assert pytest.approx(engine.p_trigram("d", "a", "b"), rel=1e-5) == 1 / 2
        assert engine.p_trigram("e", "a", "b") == 0.0

        candidates = engine.get_candidates(("a", "b"))
        assert candidates == ["c", "d"]
        dist = engine.get_candidate_distribution(("a", "b"))
        assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0

    def test_sequence_probability_chain_rule(self, engine):
        # P("a", "b", "c") = P("a") * P("b"|"a") * P("c"|"b") [under bigram assumption]
        # P("a") = 3/11, P("b"|"a") = 2/3, P("c"|"b") = 1/2
        # Product = (3/11) * (2/3) * (1/2) = 1/11
        prob_bi = engine.sequence_probability(["a", "b", "c"], order=2)
        assert pytest.approx(prob_bi, rel=1e-5) == 1 / 11

        # Trigram sequence probability:
        # P("a", "b", "c") = P("a") * P("b"|"a") * P("c"|"a", "b")
        # = (3/11) * (2/3) * (1/2) = 1/11
        prob_tri = engine.sequence_probability(["a", "b", "c"], order=3)
        assert pytest.approx(prob_tri, rel=1e-5) == 1 / 11

        # Impossible sequence has probability 0.0
        assert engine.sequence_probability(["b", "a"]) == 0.0
        assert engine.sequence_probability([]) == 1.0

    def test_log_probability(self, engine):
        # P("b" | "a") = 2/3 -> log2(2/3)
        log_p = engine.log_probability("b", context="a", base=2.0)
        assert pytest.approx(log_p, rel=1e-5) == math.log2(2 / 3)

        # Zero probability gives -inf
        assert engine.log_probability("c", context="a") == -math.inf

    def test_branching_factors(self, engine):
        # 'a' has 2 branches ('b', 'e')
        assert engine.branching_factor("a") == 2
        # 'b' has 2 branches ('c', 'd')
        assert engine.branching_factor("b") == 2
        # 'c' has 1 branch ('.')
        assert engine.branching_factor("c") == 1
        # '.' has 0 branches
        assert engine.branching_factor(".") == 0

        # Active contexts for order 2 are: 'a', 'b', 'c', 'd', 'e' (5 contexts)
        # Branches: a:2, b:2, c:1, d:1, e:1. Total branches = 7, average = 7/5 = 1.4
        assert pytest.approx(engine.average_branching_factor(order=2), rel=1e-5) == 1.4


class TestNGramProbabilityEngineCorpusIntegration:
    """Integration tests on the actual 34-sentence reconstructed story corpus."""

    @pytest.fixture(scope="class")
    def corpus_engine(self):
        loader = CorpusLoader()
        engine = NGramProbabilityEngine(n=3, include_lower_orders=True)
        engine.fit_corpus(loader)
        return engine

    def test_exact_corpus_dimensions(self, corpus_engine):
        assert corpus_engine.total_tokens == 415
        assert corpus_engine.vocabulary_size == 120
        assert corpus_engine.get_total_ngrams(order=1) == 415
        assert corpus_engine.get_total_ngrams(order=2) == 381
        assert corpus_engine.get_total_ngrams(order=3) == 347

    def test_period_never_has_outgoing_transitions(self, corpus_engine):
        assert corpus_engine.get_unigram_count(".") == 34
        assert corpus_engine.get_context_count(".", order=2) == 0
        assert corpus_engine.get_candidates(".") == []
        assert corpus_engine.branching_factor(".") == 0

    def test_known_bigram_transitions(self, corpus_engine):
        # "crystal" is always followed by "castle" in the corpus (6 times)
        assert corpus_engine.get_count(("crystal", "castle")) == 6
        assert corpus_engine.get_context_count("crystal", order=2) == 6
        assert corpus_engine.p_bigram("castle", "crystal") == 1.0

        # "dark" is always followed by "cave"
        assert corpus_engine.get_count(("dark", "cave")) == 6
        assert corpus_engine.get_context_count("dark", order=2) == 6
        assert corpus_engine.p_bigram("cave", "dark") == 1.0

        # "secret" is always followed by "garden" (5 times)
        assert corpus_engine.get_count(("secret", "garden")) == 5
        assert corpus_engine.get_context_count("secret", order=2) == 5
        assert corpus_engine.p_bigram("garden", "secret") == 1.0

    def test_known_trigram_transitions(self, corpus_engine):
        # ("the", "crystal", "castle") occurs 6 times
        assert corpus_engine.get_count(("the", "crystal", "castle")) == 6
        assert corpus_engine.get_context_count(("the", "crystal"), order=3) == 6
        assert corpus_engine.p_trigram("castle", "the", "crystal") == 1.0

        # ("the", "dark", "cave") occurs 5 times (out of 6 "dark cave" occurrences)
        dark_cave_tri = corpus_engine.get_count(("the", "dark", "cave"))
        assert dark_cave_tri in (5, 6)
        assert corpus_engine.p_trigram("cave", "the", "dark") == 1.0

    def test_candidate_distributions_sum_to_one(self, corpus_engine):
        # Sample frequent contexts
        sample_contexts = ["the", "milo", "pip", "zara", "in", "near"]
        for ctx in sample_contexts:
            dist = corpus_engine.get_candidate_distribution(ctx)
            assert len(dist) > 0
            assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0

    def test_average_bigram_branching_factor_matches_audit(self, corpus_engine):
        # From Phase 2.1.1 / 2.4 audit: 259 unique bigrams across 119 active word contexts
        # Branching factor = 259 / 119 = 2.17647...
        avg_bf = corpus_engine.average_branching_factor(order=2)
        assert pytest.approx(avg_bf, rel=1e-4) == 259 / 119


class TestNGramProbabilityEngineErrorHandling:
    """Error handling and boundary condition tests."""

    def test_unfitted_engine_raises_runtime_error(self):
        engine = NGramProbabilityEngine(n=2)
        with pytest.raises(RuntimeError, match="must be fitted"):
            engine.probability("milo")

    def test_fit_empty_raises_value_error(self):
        engine = NGramProbabilityEngine(n=2)
        with pytest.raises(ValueError, match="empty sentence list"):
            engine.fit([])

    def test_fit_corpus_invalid_object_raises_type_error(self):
        engine = NGramProbabilityEngine(n=2)
        with pytest.raises(TypeError, match="must provide a get_sentences"):
            engine.fit_corpus(object())

    def test_invalid_order_init(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            NGramProbabilityEngine(n=0)
