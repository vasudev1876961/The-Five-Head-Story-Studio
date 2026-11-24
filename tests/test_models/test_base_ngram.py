"""Unit tests for BaseNGramModel abstract base and common interface."""

import pytest
from pathlib import Path

from five_head_nlp.models.base_ngram import BaseNGramModel
from five_head_nlp.models.unigram import UnigramModel
from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


class DummyNGramModel(BaseNGramModel):
    """Concrete minimal subclass for testing BaseNGramModel base behaviors."""

    def __init__(self, n: int = 1) -> None:
        super().__init__(n=n)

    def probability(self, token, context=None):
        self._check_fitted()
        return self.engine.probability(token, context=context)

    def get_candidates(self, context=None):
        self._check_fitted()
        return self.engine.get_candidates(context)

    def get_candidate_distribution(self, context=None):
        self._check_fitted()
        return self.engine.get_candidate_distribution(context)

    def sequence_probability(self, tokens):
        self._check_fitted()
        return self.engine.sequence_probability(tokens)


class TestBaseNGramModelCore:
    """Test base class initialization, validation, and training states."""

    def test_order_validation(self):
        with pytest.raises(ValueError, match="order must be an integer >= 1"):
            DummyNGramModel(n=0)
        with pytest.raises(ValueError, match="order must be an integer >= 1"):
            DummyNGramModel(n=-2)

    def test_unfitted_state(self):
        model = DummyNGramModel(n=2)
        assert not model.is_fitted
        assert model.vocabulary_size == 0
        assert model.total_tokens == 0
        assert model.metadata == {}

        with pytest.raises(RuntimeError, match="must be trained"):
            model.probability("milo")

        with pytest.raises(RuntimeError, match="must be trained"):
            model.to_dict()

    def test_fit_empty_raises_value_error(self):
        model = DummyNGramModel(n=2)
        with pytest.raises(ValueError, match="empty sentences collection"):
            model.fit([])

    def test_fit_corpus_invalid_loader_raises_type_error(self):
        model = DummyNGramModel(n=2)
        with pytest.raises(TypeError, match="must provide a get_sentences"):
            model.fit_corpus(object())

    def test_fit_success_and_metadata(self):
        sentences = [
            ["pip", "flew", "."],
            ["milo", "jumped", "."],
        ]
        model = DummyNGramModel(n=2)
        model.fit(sentences)

        assert model.is_fitted
        assert model.vocabulary_size == 5  # pip, flew, ., milo, jumped
        assert model.total_tokens == 6
        info = model.get_model_info()
        assert info["model_type"] == "DummyNGramModel"
        assert info["order"] == 2
        assert info["is_fitted"] is True
        assert info["vocabulary_size"] == 5
        assert info["total_tokens"] == 6

        # Check metadata
        assert model.metadata["sentence_count"] == 2
        assert model.metadata["total_tokens"] == 6
        assert model.metadata["vocabulary_size"] == 5

    def test_is_known_token_and_context(self):
        sentences = [["pip", "flew", "to", "forest", "."]]
        model = DummyNGramModel(n=2)
        model.fit(sentences)

        assert model.is_known_token("pip") is True
        assert model.is_known_token("forest") is True
        assert model.is_known_token(".") is True
        assert model.is_known_token("unseen") is False

        assert model.is_known_context("pip") is True
        assert model.is_known_context("to") is True
        # period has zero outgoing transitions
        assert model.is_known_context(".") is False
        assert model.is_known_context("unseen") is False

    def test_interface_polymorphism_across_all_models(self):
        """Verify that Unigram, Bigram, and Trigram models all implement uniform interface."""
        models = [UnigramModel(), BigramModel(), TrigramModel()]
        sentences = [["pip", "flew", "."], ["zara", "smiled", "."]]

        for m in models:
            m.fit(sentences)
            assert m.is_fitted
            assert isinstance(m.probability("pip"), float)
            assert isinstance(m.conditional_probability("pip"), float)
            assert isinstance(m.log_probability("pip"), float)
            assert isinstance(m.get_candidates(), list)
            assert isinstance(m.get_candidate_distribution(), dict)
            assert isinstance(m.sequence_probability(["pip", "flew"]), float)
            assert isinstance(m.sequence_log_probability(["pip", "flew"]), float)
            assert isinstance(m.get_model_info(), dict)
