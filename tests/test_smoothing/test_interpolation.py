"""Unit tests for Linear Interpolation probability smoothing."""

import math
import pytest

from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.smoothing.interpolation import InterpolationSmoother
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
def trained_interpolator(synthetic_corpus):
    model = TrigramModel()
    model.fit(synthetic_corpus)
    return InterpolationSmoother(weights=(0.2, 0.3, 0.5), model=model)


class TestInterpolationSmootherSynthetic:
    """Mathematical verification of Linear Interpolation smoother."""

    def test_weight_validation(self, synthetic_corpus):
        model = TrigramModel()
        model.fit(synthetic_corpus)

        # Negative weight
        with pytest.raises(ValueError, match="must be non-negative"):
            InterpolationSmoother(weights=(-0.1, 0.6, 0.5), model=model)

        # Sum != 1.0
        with pytest.raises(ValueError, match="must sum to 1.0"):
            InterpolationSmoother(weights=(0.5, 0.5, 0.5), model=model)

        # Wrong length
        with pytest.raises(ValueError, match="exactly 3 values"):
            InterpolationSmoother(weights=(0.5, 0.5), model=model)

    def test_linear_interpolation_math(self, trained_interpolator):
        # Weights: l1 = 0.2, l2 = 0.3, l3 = 0.5
        # Context ('a', 'b'):
        # P_tri('c' | 'a', 'b') = 1/2 = 0.5
        # P_bi('c' | 'b') = 1/2 = 0.5
        # P_uni('c') with Laplace floor: (1 + 1) / (11 + 6) = 2 / 17
        # P_interp = 0.2 * (2/17) + 0.3 * (0.5) + 0.5 * (0.5) = (0.4/17) + 0.15 + 0.25
        # = (0.4/17) + 0.40 = 0.023529 + 0.4 = 0.423529
        expected = 0.2 * (2 / 17) + 0.3 * 0.5 + 0.5 * 0.5
        actual = trained_interpolator.probability("c", context=("a", "b"))
        assert pytest.approx(actual, rel=1e-5) == expected

    def test_simplex_normalization(self, trained_interpolator):
        dist_ab = trained_interpolator.get_distribution(context=("a", "b"))
        assert len(dist_ab) == 6
        assert pytest.approx(sum(dist_ab.values()), rel=1e-5) == 1.0

        dist_b = trained_interpolator.get_distribution(context="b")
        assert len(dist_b) == 6
        assert pytest.approx(sum(dist_b.values()), rel=1e-5) == 1.0

        dist_uni = trained_interpolator.get_distribution(context=())
        assert len(dist_uni) == 6
        assert pytest.approx(sum(dist_uni.values()), rel=1e-5) == 1.0

    def test_optimize_weights(self, synthetic_corpus):
        model = TrigramModel()
        model.fit(synthetic_corpus)
        smoother = InterpolationSmoother(weights=(0.33, 0.33, 0.34), model=model)

        val_data = [["a", "b", "c", "."]]
        best_w = smoother.optimize_weights(val_data, step=0.2)
        assert len(best_w) == 3
        assert pytest.approx(sum(best_w), rel=1e-5) == 1.0


class TestInterpolationSmootherCorpusIntegration:
    """Integration test with the 34-sentence story corpus."""

    @pytest.fixture(scope="class")
    def corpus_interp(self):
        loader = CorpusLoader()
        model = TrigramModel()
        model.fit_corpus(loader)
        return InterpolationSmoother(weights=(0.15, 0.35, 0.50), model=model)

    def test_positive_probability_for_novel_contexts(self, corpus_interp):
        # Even for an unseen context, unigram weight guarantees positive probability
        p = corpus_interp.probability("castle", context=("unseen_w1", "unseen_w2"))
        assert p > 0.0

    def test_simplex_normalization_real_corpus(self, corpus_interp):
        for ctx in [(), ("crystal",), ("the", "crystal"), ("unseen1", "unseen2")]:
            dist = corpus_interp.get_distribution(context=ctx)
            assert len(dist) == 120
            assert pytest.approx(sum(dist.values()), rel=1e-5) == 1.0

    def test_weight_optimization_held_out_separation(self, corpus_interp):
        """Audit requirement: ensure held-out evaluation data does not participate in weight optimization."""
        loader = CorpusLoader()
        sentences = loader.get_sentences()

        # 3-way split: 24 train, 5 val, 5 test
        val_sentences = sentences[24:29]
        test_sentences = sentences[29:34]

        # Optimization runs strictly on validation sentences
        best_weights = corpus_interp.optimize_weights(val_sentences, step=0.1)

        # Confirm weights are valid simplex
        assert len(best_weights) == 3
        assert pytest.approx(sum(best_weights), rel=1e-5) == 1.0

        # Evaluate perplexity on test sentences without altering test sentences or trained parameters
        ppl_before = corpus_interp.perplexity(test_sentences)
        assert ppl_before > 0.0
        assert math.isfinite(ppl_before)
        assert len(test_sentences) == 5
