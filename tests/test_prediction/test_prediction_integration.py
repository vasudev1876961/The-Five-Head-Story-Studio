"""Integration tests for next-token prediction with actual corpus models and evaluators."""

import pytest

from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.prediction.evaluator import PredictionEvaluator
from five_head_nlp.prediction.predictor import NextTokenPredictor
from five_head_nlp.smoothing.backoff import BackoffSmoother
from five_head_nlp.smoothing.interpolation import InterpolationSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


class TestPredictionIntegration:
    """Integration verification using the 34-sentence children's story corpus."""

    @pytest.fixture(scope="class")
    def corpus_predictor(self):
        loader = CorpusLoader()
        model = TrigramModel()
        model.fit_corpus(loader)
        smoother = BackoffSmoother(discount=0.5, model=model)
        return NextTokenPredictor(model=model, smoother=smoother)

    def test_real_corpus_candidate_distribution(self, corpus_predictor):
        # 'the crystal' in training was followed by 'castle'
        candidates = corpus_predictor.rank_candidates(context=("the", "crystal"))
        assert len(candidates) == 120  # Backoff smoother provides full support over |V|=120
        assert candidates[0].token == "castle"
        assert candidates[0].probability > 0.85

    def test_decoding_strategies_produce_valid_tokens(self, corpus_predictor):
        ctx = ("near", "the")

        res_greedy = corpus_predictor.predict_next(context=ctx, strategy="greedy")
        assert res_greedy.selected_token in corpus_predictor.vocabulary

        res_topk = corpus_predictor.predict_next(context=ctx, strategy="top_k", k=5, seed=42)
        assert res_topk.selected_token in corpus_predictor.vocabulary

        res_temp = corpus_predictor.predict_next(context=ctx, strategy="temperature", temperature=0.8, seed=42)
        assert res_temp.selected_token in corpus_predictor.vocabulary

    def test_reproducibility_across_identical_seeds(self, corpus_predictor):
        ctx = ("in", "the")
        dec1 = corpus_predictor.create_decoder(strategy="top_k", k=5, seed=999)
        dec2 = corpus_predictor.create_decoder(strategy="top_k", k=5, seed=999)

        seq1 = [corpus_predictor.predict_next(context=ctx, decoder=dec1).selected_token for _ in range(15)]
        seq2 = [corpus_predictor.predict_next(context=ctx, decoder=dec2).selected_token for _ in range(15)]
        assert seq1 == seq2

    def test_evaluator_benchmark_execution(self, corpus_predictor):
        evaluator = PredictionEvaluator(predictor=corpus_predictor)
        metrics = evaluator.evaluate_strategy(strategy="greedy")

        assert metrics["total_events_evaluated"] == 58
        assert metrics["coverage_pct"] == 100.0
        assert metrics["invalid_distribution_count"] == 0
        assert 0.0 <= metrics["top1_accuracy"] <= 1.0
        assert 0.0 <= metrics["mean_reciprocal_rank_mrr"] <= 1.0
        assert metrics["latency_mean_ms"] >= 0.0

    def test_rollout_stops_on_period(self, corpus_predictor):
        tokens, steps = corpus_predictor.generate(
            prompt=["pip", "flew"],
            max_tokens=20,
            strategy="greedy",
        )
        assert len(tokens) >= 3
        # If finished naturally, last token must be '.'
        if len(tokens) < 22:
            assert tokens[-1] == "."
