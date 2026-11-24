"""Integration tests for five-head scoring engine with NextTokenPredictor and real corpus."""

import pytest

from five_head_nlp.heads.head_orchestrator import HeadOrchestrator
from five_head_nlp.heads.scoring_config import ScoringConfig
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.prediction.predictor import NextTokenPredictor
from five_head_nlp.smoothing.backoff import BackoffSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


class TestHeadIntegration:
    """Integration verification of five-head candidate scoring with predictor."""

    @pytest.fixture(scope="class")
    def fitted_predictor(self):
        loader = CorpusLoader()
        sentences = loader.get_sentences()
        train_sentences = sentences[:24]

        model = TrigramModel().fit(train_sentences)
        smoother = BackoffSmoother(discount=0.5, model=model)
        predictor = NextTokenPredictor(model=model, smoother=smoother)
        predictor.orchestrator = HeadOrchestrator()
        return predictor

    def test_baseline_mode_unaffected(self, fitted_predictor):
        # Baseline mode must produce identical output as Phase 3.4
        res_baseline = fitted_predictor.predict_next(context=("the", "crystal"), mode="baseline")
        assert res_baseline.selected_token == "castle"
        assert res_baseline.metadata["mode"] == "baseline"

    def test_five_head_mode_produces_fused_candidates(self, fitted_predictor):
        res_five_head = fitted_predictor.predict_next(context=("the", "crystal"), mode="five_head")
        assert res_five_head.selected_token == "castle"
        assert res_five_head.metadata["mode"] == "five_head"
        assert "orchestrator_result" in res_five_head.metadata

        top_cand = res_five_head.candidates[0]
        assert top_cand.token == "castle"
        assert top_cand.probability > 0.0

    def test_repeat_blocker_breaks_immediate_loop(self, fitted_predictor):
        """Verify Repeat Blocker penalizes self-loop."""
        # Propose continuation given context 'castle'
        res = fitted_predictor.predict_next(context=("the", "castle"), mode="five_head")
        # 'castle' followed by 'castle' must receive low Repeat Blocker score
        orch_res = res.metadata["orchestrator_result"]
        castle_cand = next((c for c in orch_res["top_candidates"] if c["token"] == "castle"), None)
        if castle_cand:
            assert castle_cand["head_breakdown"]["Repeat Blocker"]["raw_score"] == 0.05

    def test_generate_five_head_rollout(self, fitted_predictor):
        tokens, steps = fitted_predictor.generate(
            prompt=["pip", "flew"],
            max_tokens=15,
            strategy="greedy",
            mode="five_head",
        )
        assert len(tokens) >= 3
        assert tokens[:2] == ["pip", "flew"]
        assert all(isinstance(t, str) for t in tokens)
