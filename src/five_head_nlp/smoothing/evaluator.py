"""Evaluation and comparative benchmarking module for probability smoothing."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.models.unigram import UnigramModel
from five_head_nlp.probability.mle import NGramProbabilityEngine
from five_head_nlp.smoothing.add_k import AddKSmoother
from five_head_nlp.smoothing.backoff import BackoffSmoother
from five_head_nlp.smoothing.good_turing import GoodTuringSmoother
from five_head_nlp.smoothing.interpolation import InterpolationSmoother
from five_head_nlp.smoothing.laplace import LaplaceSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader
from five_head_nlp.tokenization.vocabulary import Vocabulary


class SmoothingEvaluator:
    """Evaluates and compares unsmoothed MLE and various smoothing techniques.

    Measures perplexity, negative log-likelihood (NLL), zero-probability event count,
    unseen mass distribution, and probability simplex normalization across
    a strictly separated train/validation split.
    Measures total negative log-likelihood (NLL), mean cross-entropy (H = NLL / M),
    perplexity (PPL = exp(H)), zero-probability event count, and probability simplex
    normalization across a strictly separated 3-way split:
        - Train (24 sentences, ~70.6%): Used for model fitting.
        - Validation (5 sentences, ~14.7%): Used for hyperparameter tuning / weight optimization.
        - Test (5 sentences, ~14.7%): Strictly held-out; used only for final benchmark reporting.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        train_count: int = 24,
        val_count: int = 5,
        random_seed: int = 42,
        corpus_loader: Optional[CorpusLoader] = None,
    ) -> None:
        """Initialize SmoothingEvaluator."""
        self.train_ratio: float = train_ratio
        """Initialize SmoothingEvaluator with 3-way partition."""
        self.train_count: int = train_count
        self.val_count: int = val_count
        self.random_seed: int = random_seed
        self.loader: CorpusLoader = corpus_loader or CorpusLoader()

        self.sentences: List[List[str]] = self.loader.get_sentences()
        self.train_sentences: List[List[str]] = []
        self.val_sentences: List[List[str]] = []
        self.test_sentences: List[List[str]] = []

        self._split_data()

        # Full closed vocabulary of the domain (|V| = 120) with training transitions from train_sentences
        self.vocabulary: Vocabulary = Vocabulary(self.sentences)
        self.engine: NGramProbabilityEngine = NGramProbabilityEngine(n=3, include_lower_orders=True)
        self.engine.fit(self.train_sentences)

        self.mle_bigram: BigramModel = BigramModel()
        self.mle_bigram.engine = self.engine
        self.mle_bigram.vocabulary = self.vocabulary
        self.mle_bigram._is_fitted = True

        self.mle_trigram: TrigramModel = TrigramModel()
        self.mle_trigram.engine = self.engine
        self.mle_trigram.vocabulary = self.vocabulary
        self.mle_trigram._is_fitted = True

        # Interpolation with optimization on validation set only
        interp_opt = InterpolationSmoother(
            weights=(0.15, 0.35, 0.50), model=self.mle_trigram
        )
        self.optimized_weights = interp_opt.optimize_weights(self.val_sentences, step=0.05)

        # Initialize smoothers
        self.smoothers: Dict[str, Any] = {
            "Unsmoothed MLE (Trigram)": self.mle_trigram,
            "Laplace (Add-1)": LaplaceSmoother(model=self.mle_trigram),
            "Add-k (k=0.1)": AddKSmoother(k=0.1, model=self.mle_trigram),
            "Add-k (k=0.5)": AddKSmoother(k=0.5, model=self.mle_trigram),
            "Good-Turing": GoodTuringSmoother(k_threshold=5, model=self.mle_trigram),
            "Absolute Discounting Backoff": BackoffSmoother(discount=0.5, model=self.mle_trigram),
            "Linear Interpolation (Fixed: 0.15, 0.35, 0.50)": InterpolationSmoother(
                weights=(0.15, 0.35, 0.50), model=self.mle_trigram
            ),
            f"Linear Interpolation (Optimized on Val: {self.optimized_weights})": interp_opt,
        }

    def _split_data(self) -> None:
        """Partition 34 sentences deterministically into train and validation sets."""
        """Partition 34 sentences deterministically into train (24), val (5), and test (5)."""
        indices = list(range(len(self.sentences)))
        rng = random.Random(self.random_seed)
        rng.shuffle(indices)

        split_point = int(len(indices) * self.train_ratio)
        train_idx = sorted(indices[:split_point])
        val_idx = sorted(indices[split_point:])
        train_idx = sorted(indices[:self.train_count])
        val_idx = sorted(indices[self.train_count : self.train_count + self.val_count])
        test_idx = sorted(indices[self.train_count + self.val_count :])

        self.train_sentences = [self.sentences[i] for i in train_idx]
        self.val_sentences = [self.sentences[i] for i in val_idx]
        self.test_sentences = [self.sentences[i] for i in test_idx]

    def evaluate_model(
        self,
        name: str,
        model_or_smoother: Any,
        eval_sentences: Sequence[Sequence[str]],
    ) -> Dict[str, Any]:
        """Compute perplexity, NLL, zero-count, and normalization metrics."""
        """Compute exact NLL, mean cross-entropy, and perplexity (PPL = exp(H)).

        Uses natural logarithms consistently:
            NLL_total = - sum ln P(w | context)
            H = NLL_total / M
            PPL = exp(H)
        """
        total_tokens = sum(len(s) for s in eval_sentences)
        zero_events = 0
        total_log_p = 0.0
        total_nll = 0.0

        for sentence in eval_sentences:
            for i, token in enumerate(sentence):
                if i == 0:
                    ctx: Tuple[str, ...] = ()
                elif i == 1:
                    ctx = (sentence[0],)
                else:
                    ctx = (sentence[i - 2], sentence[i - 1])

                p = model_or_smoother.probability(token, context=ctx)
                if p <= 0.0:
                    zero_events += 1
                else:
                    total_log_p += math.log2(p)
                    total_nll += -math.log(p)

        if zero_events > 0:
            perplexity = float("inf")
            nll = float("inf")
            avg_log_p = -float("inf")
            mean_h = float("inf")
            mean_h_bits = float("inf")
            total_nll_val = float("inf")
        else:
            avg_log_p = total_log_p / total_tokens if total_tokens > 0 else 0.0
            nll = -avg_log_p
            perplexity = math.pow(2.0, nll)
            total_nll_val = total_nll
            mean_h = total_nll / total_tokens if total_tokens > 0 else 0.0
            perplexity = math.exp(mean_h)
            mean_h_bits = mean_h / math.log(2.0)

        # Check normalization on sample contexts
        sample_contexts = [(), ("the",), ("milo",), ("the", "crystal"), ("near", "the")]
        norm_checks: List[float] = []
        for c in sample_contexts:
            if hasattr(model_or_smoother, "get_distribution"):
                dist = model_or_smoother.get_distribution(context=c)
                norm_checks.append(sum(dist.values()))
            elif hasattr(model_or_smoother, "get_candidate_distribution"):
                dist = model_or_smoother.get_candidate_distribution(context=c)
                norm_checks.append(sum(dist.values()) if dist else 0.0)

        avg_norm = sum(norm_checks) / len(norm_checks) if norm_checks else 1.0

        return {
            "name": name,
            "total_tokens_evaluated": total_tokens,
            "evaluated_prediction_events": total_tokens,
            "zero_probability_events": zero_events,
            "zero_event_rate_pct": round((zero_events / total_tokens) * 100, 2) if total_tokens > 0 else 0.0,
            "cross_entropy_nll": round(nll, 4) if nll != float("inf") else "inf",
            "total_negative_log_likelihood": round(total_nll_val, 4) if total_nll_val != float("inf") else "inf",
            "mean_cross_entropy_nats": round(mean_h, 4) if mean_h != float("inf") else "inf",
            "mean_cross_entropy_bits": round(mean_h_bits, 4) if mean_h_bits != float("inf") else "inf",
            "perplexity": round(perplexity, 4) if perplexity != float("inf") else "inf",
            "avg_probability_simplex_sum": round(avg_norm, 6),
        }

    def run_full_comparison(self) -> Dict[str, Any]:
        """Execute evaluation across all smoothers on test, validation, and training sets."""
        test_results = []
        val_results = []
        train_results = []

        for name, smoother in self.smoothers.items():
            test_results.append(self.evaluate_model(name, smoother, self.test_sentences))
            val_results.append(self.evaluate_model(name, smoother, self.val_sentences))
            train_results.append(self.evaluate_model(name, smoother, self.train_sentences))

        return {
            "split_info": {
                "total_sentences": len(self.sentences),
                "train_sentences": len(self.train_sentences),
                "validation_sentences": len(self.val_sentences),
                "test_sentences": len(self.test_sentences),
                "random_seed": self.random_seed,
                "train_tokens": sum(len(s) for s in self.train_sentences),
                "val_tokens": sum(len(s) for s in self.val_sentences),
                "test_tokens": sum(len(s) for s in self.test_sentences),
            },
            "test_results": test_results,
            "validation_results": val_results,
            "training_results": train_results,
        }

    def export_artifacts(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Export JSON and CSV benchmark reports."""
        target_dir = Path(output_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        results = self.run_full_comparison()

        # 1. JSON Report
        json_path = target_dir / "smoothing_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # 2. Comparison CSV (Benchmarked strictly on held-out test data)
        csv_path = target_dir / "smoothing_comparison.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Method",
                "Test_Tokens_M",
                "Zero_Events",
                "Zero_Event_Pct",
                "Total_NLL_nats",
                "Mean_CrossEntropy_nats",
                "Mean_CrossEntropy_bits",
                "Perplexity",
                "Simplex_Sum",
            ])
            for r in results["test_results"]:
                writer.writerow([
                    r["name"],
                    r["evaluated_prediction_events"],
                    r["zero_probability_events"],
                    f"{r['zero_event_rate_pct']}%",
                    r["total_negative_log_likelihood"],
                    r["mean_cross_entropy_nats"],
                    r["mean_cross_entropy_bits"],
                    r["perplexity"],
                    r["avg_probability_simplex_sum"],
                ])

        # 3. Good-Turing Frequency Counts CSV
        gt_smoother: GoodTuringSmoother = self.smoothers["Good-Turing"]
        fof_path = target_dir / "good_turing_frequency_counts.csv"
        with open(fof_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Order",
                "Frequency_r",
                "Type_Count_N_r",
                "Total_Occurrences_r_times_N_r",
                "Adjusted_Count_r_star",
            ])
            for order in [1, 2, 3]:
                fof = gt_smoother.get_frequency_of_frequencies(order=order)
                for r, n_r in sorted(fof.items()):
                    adj_r = gt_smoother.get_adjusted_count(r, order=order)
                    writer.writerow([order, r, n_r, r * n_r, round(adj_r, 4)])

        # 4. Context Probability Distribution CSV
        prob_dist_path = target_dir / "probability_distribution.csv"
        contexts_to_export = [
            ("crystal",),
            ("dark",),
            ("secret",),
            ("near",),
            ("the",),
        ]
        sample_tokens = ["castle", "cave", "garden", "forest", "the", "silver", "milo", "pip"]

        with open(prob_dist_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = ["Context", "Candidate_Token"] + list(self.smoothers.keys())
            writer.writerow(headers)

            for ctx in contexts_to_export:
                ctx_str = " ".join(ctx)
                for tok in sample_tokens:
                    row = [ctx_str, tok]
                    for name, sm in self.smoothers.items():
                        prob = sm.probability(tok, context=ctx)
                        row.append(round(prob, 6))
                    writer.writerow(row)

        return {
            "json": json_path,
            "comparison_csv": csv_path,
            "fof_csv": fof_path,
            "prob_csv": prob_dist_path,
        }
