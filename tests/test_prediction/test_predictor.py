"""Unit tests for NextTokenPredictor."""

import pytest

from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.models.unigram import UnigramModel
from five_head_nlp.prediction.predictor import NextTokenPredictor
from five_head_nlp.smoothing.backoff import BackoffSmoother
from five_head_nlp.smoothing.laplace import LaplaceSmoother


@pytest.fixture
def synthetic_corpus():
    return [
        ["milo", "found", "the", "silver", "key", "."],
        ["milo", "liked", "the", "crystal", "castle", "."],
        ["pip", "flew", "near", "the", "castle", "."],
    ]


class TestNextTokenPredictor:
    """Test NextTokenPredictor across model orders, smoothers, and edge contexts."""

    def test_unfitted_model_raises_error(self):
        model = TrigramModel()
        with pytest.raises(ValueError, match="unfitted model"):
            NextTokenPredictor(model=model)

    def test_unigram_prediction(self, synthetic_corpus):
        model = UnigramModel().fit(synthetic_corpus)
        predictor = NextTokenPredictor(model=model)

        assert predictor.model_order == 1
        res = predictor.predict_next(context=("milo",), strategy="greedy")
        assert res.selected_token is not None
        assert res.strategy == "greedy"
        assert res.distribution_sum > 0.0

    def test_bigram_prediction(self, synthetic_corpus):
        model = BigramModel().fit(synthetic_corpus)
        predictor = NextTokenPredictor(model=model)

        assert predictor.model_order == 2
        # Context 'milo' was followed by 'found' and 'liked'
        candidates = predictor.rank_candidates(context=("milo",))
        cand_tokens = [c.token for c in candidates]
        assert "found" in cand_tokens
        assert "liked" in cand_tokens

    def test_trigram_prediction(self, synthetic_corpus):
        model = TrigramModel().fit(synthetic_corpus)
        predictor = NextTokenPredictor(model=model)

        assert predictor.model_order == 3
        # Context ('the', 'crystal') was followed by 'castle'
        candidates = predictor.rank_candidates(context=("the", "crystal"))
        assert candidates[0].token == "castle"
        assert candidates[0].probability == 1.0

    def test_smoothing_integration(self, synthetic_corpus):
        model = TrigramModel().fit(synthetic_corpus)
        smoother = LaplaceSmoother(model=model)
        predictor = NextTokenPredictor(model=model, smoother=smoother)

        assert predictor.smoother_name == "LaplaceSmoother"
        # Even for unseen context, smoother covers full vocabulary
        res = predictor.predict_next(context=("unseen1", "unseen2"), strategy="greedy")
        assert res.selected_token is not None
        assert res.distribution_sum == pytest.approx(1.0, rel=1e-5)
        assert len(res.candidates) == model.vocabulary_size

    def test_context_ending_with_period_is_terminal(self, synthetic_corpus):
        model = TrigramModel().fit(synthetic_corpus)
        predictor = NextTokenPredictor(model=model)

        res = predictor.predict_next(context=("castle", "."))
        assert res.is_terminal
        assert res.selected_token is None
        assert res.candidates == []

    def test_unknown_context_unsmoothed_returns_empty_candidates(self, synthetic_corpus):
        model = TrigramModel().fit(synthetic_corpus)
        predictor = NextTokenPredictor(model=model)

        res = predictor.predict_next(context=("unknown_word",))
        assert res.selected_token is None
        assert res.candidates == []
        assert res.distribution_sum == 0.0

    def test_generate_sequence_rollout_terminates_on_period(self, synthetic_corpus):
        model = TrigramModel().fit(synthetic_corpus)
        smoother = BackoffSmoother(discount=0.5, model=model)
        predictor = NextTokenPredictor(model=model, smoother=smoother)

        tokens, steps = predictor.generate(
            prompt=["the", "crystal"],
            max_tokens=10,
            strategy="greedy",
        )
        assert tokens[:2] == ["the", "crystal"]
        assert len(tokens) > 2
        # Sequence should terminate with period or reach max_tokens
        assert tokens[-1] == "." or len(tokens) == 12
