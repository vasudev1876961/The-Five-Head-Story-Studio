"""Comparative evaluation module for next-token prediction and decoding strategies."""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.prediction.decoding import BaseDecoder
from five_head_nlp.prediction.predictor import NextTokenPredictor
from five_head_nlp.smoothing.backoff import BackoffSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


class PredictionEvaluator:
    """Evaluates next-token prediction accuracy, MRR, latency, coverage, and reproducibility.

    Evaluates across held-out test prediction events using the established leak-free
    3-way split (24 train, 5 validation, 5 test).
    """

    def __init__(
        self,
        predictor: Optional[NextTokenPredictor] = None,
        test_sentences: Optional[Sequence[Sequence[str]]] = None,
        corpus_loader: Optional[CorpusLoader] = None,
    ) -> None:
        """Initialize PredictionEvaluator."""
        self.loader: CorpusLoader = corpus_loader or CorpusLoader()
        sentences = self.loader.get_sentences()

        # Deterministic 3-way partition matching Phase 3.3 audit (seed 42)
        rng = random.Random(42)
        indices = list(range(len(sentences)))
        rng.shuffle(indices)
        train_idx = sorted(indices[:24])
        val_idx = sorted(indices[24:29])
        test_idx = sorted(indices[29:])

        self.train_sentences: List[List[str]] = [sentences[i] for i in train_idx]
        self.val_sentences: List[List[str]] = [sentences[i] for i in val_idx]
        self.test_sentences: List[List[str]] = (
            [list(s) for s in test_sentences] if test_sentences is not None
            else [sentences[i] for i in test_idx]
        )

        if predictor is not None:
            self.predictor: NextTokenPredictor = predictor
        else:
            # Fit trigram model on train_sentences and attach BackoffSmoother
            model = TrigramModel()
            model.fit(self.train_sentences)
            smoother = BackoffSmoother(discount=0.5, model=model)
            self.predictor = NextTokenPredictor(model=model, smoother=smoother)

    def extract_prediction_events(
        self, sentences: Sequence[Sequence[str]]
    ) -> List[Tuple[Tuple[str, ...], str]]:
        """Extract all (context, gold_token) prediction events from sentences.

        Sentence boundaries are strictly observed: no cross-sentence transitions.
        """
        events: List[Tuple[Tuple[str, ...], str]] = []
        for sentence in sentences:
            for i, gold_token in enumerate(sentence):
                if i == 0:
                    ctx: Tuple[str, ...] = ()
                elif i == 1:
                    ctx = (sentence[0],)
                else:
                    ctx = (sentence[i - 2], sentence[i - 1])
                events.append((ctx, gold_token))
        return events

    def evaluate_strategy(
        self,
        strategy: str = "greedy",
        k: int = 5,
        temperature: float = 1.0,
        seed: int = 42,
        trials: int = 1,
    ) -> Dict[str, Any]:
        """Evaluate a decoding strategy across all held-out test prediction events.

        Parameters
        ----------
        strategy : str, default='greedy'
            Strategy to evaluate ('greedy', 'top_k', 'temperature').
        k : int, default=5
            Parameter for top_k.
        temperature : float, default=1.0
            Parameter for temperature.
        seed : int, default=42
            Seed for reproducibility.
        trials : int, default=1
            Number of repeated seeded trials for stochastic decoders.

        Returns
        -------
        Dict[str, Any]
            Performance metrics including top-1 accuracy, top-k accuracy, MRR, latency.
        """
        events = self.extract_prediction_events(self.test_sentences)
        total_events = len(events)

        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        reciprocal_ranks: List[float] = []
        latencies_ms: List[float] = []
        valid_distributions = 0
        invalid_distributions = 0

        # Run repeated trials for stochastic sampling if trials > 1
        trial_seeds = [seed + t for t in range(trials)]

        for trial_seed in trial_seeds:
            decoder = self.predictor.create_decoder(
                strategy=strategy, k=k, temperature=temperature, seed=trial_seed
            )

            for ctx, gold_token in events:
                t0 = time.perf_counter()
                res = self.predictor.predict_next(context=ctx, decoder=decoder)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000)

                if res.candidates:
                    valid_distributions += 1
                else:
                    invalid_distributions += 1

                # Top-1 accuracy (based on selected token)
                if res.selected_token == gold_token:
                    top1_correct += 1

                # Rank metrics from candidate list
                gold_cand = res.get_candidate(gold_token)
                if gold_cand is not None:
                    rank = gold_cand.rank
                    reciprocal_ranks.append(1.0 / rank)
                    if rank <= 3:
                        top3_correct += 1
                    if rank <= 5:
                        top5_correct += 1
                else:
                    reciprocal_ranks.append(0.0)

        total_predictions = total_events * trials
        mean_top1 = top1_correct / total_predictions if total_predictions > 0 else 0.0
        mean_top3 = top3_correct / total_predictions if total_predictions > 0 else 0.0
        mean_top5 = top5_correct / total_predictions if total_predictions > 0 else 0.0
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        latencies_ms.sort()
        mean_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        med_lat = latencies_ms[len(latencies_ms) // 2] if latencies_ms else 0.0
        p95_lat = latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0

        return {
            "strategy": strategy,
            "params": {"k": k, "temperature": temperature, "seed": seed, "trials": trials},
            "total_events_evaluated": total_events,
            "total_predictions": total_predictions,
            "coverage_pct": round((valid_distributions / total_predictions) * 100, 2),
            "invalid_distribution_count": invalid_distributions,
            "top1_accuracy": round(mean_top1, 4),
            "top3_accuracy": round(mean_top3, 4),
            "top5_accuracy": round(mean_top5, 4),
            "mean_reciprocal_rank_mrr": round(mrr, 4),
            "latency_mean_ms": round(mean_lat, 4),
            "latency_median_ms": round(med_lat, 4),
            "latency_p95_ms": round(p95_lat, 4),
        }

    def verify_reproducibility(
        self,
        strategy: str,
        k: int = 5,
        temperature: float = 1.0,
        seed: int = 42,
        num_runs: int = 3,
    ) -> bool:
        """Verify that identical seeds produce 100% identical prediction sequences."""
        runs_output: List[List[str]] = []

        for _ in range(num_runs):
            seq, _ = self.predictor.generate(
                prompt=["the", "crystal"],
                max_tokens=15,
                strategy=strategy,
                k=k,
                temperature=temperature,
                seed=seed,
            )
            runs_output.append(seq)

        # All runs must match run 0 identically
        first_run = runs_output[0]
        return all(run == first_run for run in runs_output)

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Execute comparative benchmark across Greedy, Top-k, and Temperature configurations."""
        configs = [
            {"strategy": "greedy"},
            {"strategy": "top_k", "k": 3, "seed": 42},
            {"strategy": "top_k", "k": 5, "seed": 42},
            {"strategy": "top_k", "k": 10, "seed": 42},
            {"strategy": "temperature", "temperature": 0.1, "seed": 42},
            {"strategy": "temperature", "temperature": 0.5, "seed": 42},
            {"strategy": "temperature", "temperature": 1.0, "seed": 42},
            {"strategy": "temperature", "temperature": 1.5, "seed": 42},
            {"strategy": "temperature", "temperature": 2.0, "seed": 42},
        ]

        benchmarks = []
        for cfg in configs:
            benchmarks.append(
                self.evaluate_strategy(
                    strategy=cfg["strategy"],
                    k=cfg.get("k", 5),
                    temperature=cfg.get("temperature", 1.0),
                    seed=cfg.get("seed", 42),
                    trials=3 if cfg["strategy"] != "greedy" else 1,
                )
            )

        # Reproducibility checks
        reproducibility = {
            "greedy": self.verify_reproducibility("greedy"),
            "top_k_k5": self.verify_reproducibility("top_k", k=5, seed=42),
            "temp_1_0": self.verify_reproducibility("temperature", temperature=1.0, seed=42),
        }

        return {
            "dataset": {
                "train_sentences": len(self.train_sentences),
                "val_sentences": len(self.val_sentences),
                "test_sentences": len(self.test_sentences),
                "test_events": len(self.extract_prediction_events(self.test_sentences)),
            },
            "predictor": {
                "model": self.predictor.model.name,
                "order": self.predictor.model_order,
                "smoother": self.predictor.smoother_name,
            },
            "benchmarks": benchmarks,
            "reproducibility": reproducibility,
        }
