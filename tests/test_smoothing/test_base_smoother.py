"""Unit tests for BaseSmoother abstract class and shared utilities."""

import math
import pytest

from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.unigram import UnigramModel
from five_head_nlp.probability.mle import NGramProbabilityEngine
from five_head_nlp.smoothing.base_smoother import BaseSmoother
from five_head_nlp.smoothing.laplace import LaplaceSmoother
from five_head_nlp.tokenization.vocabulary import Vocabulary


class TestBaseSmootherInitialization:
    """Test initialization and error handling of base smoother."""

    def test_init_from_unfitted_model_raises_value_error(self):
        model = BigramModel()
        with pytest.raises(ValueError, match="unfitted model"):
            LaplaceSmoother(model=model)

    def test_init_from_unfitted_engine_raises_value_error(self):
        engine = NGramProbabilityEngine(n=2)
        vocab = Vocabulary(["a", "b"])
        with pytest.raises(ValueError, match="unfitted probability engine"):
            LaplaceSmoother(engine=engine, vocabulary=vocab)

    def test_init_missing_arguments_raises_value_error(self):
        with pytest.raises(ValueError, match="requires either a trained BaseNGramModel"):
            LaplaceSmoother()

    def test_init_success_properties(self):
        model = BigramModel()
        model.fit([["pip", "flew", "."]])
        smoother = LaplaceSmoother(model=model)

        assert smoother.name == "LaplaceSmoother"
        assert smoother.order == 2
        assert smoother.vocabulary_size == 3
        assert smoother.total_tokens == 3

    def test_perplexity_empty_sentences_returns_one(self):
        model = BigramModel()
        model.fit([["pip", "flew", "."]])
        smoother = LaplaceSmoother(model=model)
        assert smoother.perplexity([]) == 1.0
        assert smoother.perplexity([[]]) == 1.0

    def test_cross_entropy_perplexity_exact_exponential_relationship(self):
        """Audit requirement: verify H = -1/M sum ln P and PPL = exp(H) to floating tolerance."""
        model = BigramModel()
        sentences = [
            ["pip", "flew", "."],
            ["milo", "smiled", "."],
        ]
        model.fit(sentences)
        smoother = LaplaceSmoother(model=model)

        eval_sentences = [["pip", "smiled", "."]]
        total_nll, mean_h, ppl, zero_events, total_tokens = smoother.evaluate_metrics(eval_sentences)

        assert zero_events == 0
        assert total_tokens == 3
        assert total_nll > 0.0
        assert math.isclose(mean_h, total_nll / total_tokens, rel_tol=1e-9)
        assert math.isclose(ppl, math.exp(mean_h), rel_tol=1e-9)
        # Also verify base-smoother perplexity() method matches evaluate_metrics()
        assert math.isclose(smoother.perplexity(eval_sentences), ppl, rel_tol=1e-9)
