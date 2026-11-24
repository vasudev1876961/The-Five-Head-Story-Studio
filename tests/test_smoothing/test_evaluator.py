"""Unit tests for SmoothingEvaluator, 3-way partition, and artifact generation."""

import math
import pytest
from pathlib import Path

from five_head_nlp.smoothing.evaluator import SmoothingEvaluator


class TestSmoothingEvaluatorPartition:
    """Test data split and leak-free properties."""

    def test_three_way_split_counts_and_separation(self):
        evaluator = SmoothingEvaluator(train_count=24, val_count=5, random_seed=42)

        # 34 total sentences in corpus
        assert len(evaluator.sentences) == 34
        assert len(evaluator.train_sentences) == 24
        assert len(evaluator.val_sentences) == 5
        assert len(evaluator.test_sentences) == 5

        # Verify pairwise disjoint sentence indices
        train_tuples = [tuple(s) for s in evaluator.train_sentences]
        val_tuples = [tuple(s) for s in evaluator.val_sentences]
        test_tuples = [tuple(s) for s in evaluator.test_sentences]

        # Sets of sentence identities
        assert len(set(train_tuples).intersection(set(val_tuples))) == 0
        assert len(set(val_tuples).intersection(set(test_tuples))) == 0
        assert len(set(train_tuples).intersection(set(test_tuples))) == 0

    def test_token_counts(self):
        evaluator = SmoothingEvaluator(train_count=24, val_count=5, random_seed=42)
        total_tokens = sum(len(s) for s in evaluator.sentences)
        train_tokens = sum(len(s) for s in evaluator.train_sentences)
        val_tokens = sum(len(s) for s in evaluator.val_sentences)
        test_tokens = sum(len(s) for s in evaluator.test_sentences)

        assert total_tokens == 415
        assert train_tokens + val_tokens + test_tokens == 415
        assert train_tokens == 295
        assert val_tokens == 62
        assert test_tokens == 58

    def test_evaluation_metric_relationships(self):
        evaluator = SmoothingEvaluator(train_count=24, val_count=5, random_seed=42)
        results = evaluator.run_full_comparison()

        assert "test_results" in results
        assert "validation_results" in results
        assert "training_results" in results

        # Verify for every smoother on test set:
        for r in results["test_results"]:
            if r["perplexity"] != "inf":
                ppl = float(r["perplexity"])
                mean_h = float(r["mean_cross_entropy_nats"])
                mean_h_bits = float(r["mean_cross_entropy_bits"])
                total_nll = float(r["total_negative_log_likelihood"])
                m = r["evaluated_prediction_events"]

                assert m == 58
                # H = NLL / M
                assert pytest.approx(mean_h, rel=1e-3) == total_nll / m
                # PPL = exp(H)
                assert pytest.approx(ppl, rel=1e-3) == math.exp(mean_h)
                # H_bits = H / ln(2)
                assert pytest.approx(mean_h_bits, rel=1e-3) == mean_h / math.log(2.0)
                # PPL = 2^(H_bits)
                assert pytest.approx(ppl, rel=1e-3) == (2.0 ** mean_h_bits)
