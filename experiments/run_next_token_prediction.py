"""Reproducible demonstration and artifact generation for next-token prediction and decoding."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.models.unigram import UnigramModel
from five_head_nlp.prediction.evaluator import PredictionEvaluator
from five_head_nlp.prediction.generation_state import GenerationState
from five_head_nlp.prediction.predictor import NextTokenPredictor
from five_head_nlp.prediction.temperature import TemperatureDecoder
from five_head_nlp.smoothing.backoff import BackoffSmoother
from five_head_nlp.smoothing.interpolation import InterpolationSmoother
from five_head_nlp.smoothing.laplace import LaplaceSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


def main() -> None:
    print("=" * 70)
    print("Phase 3.4 — Next-Token Prediction & Decoding Engine Demonstration")
    print("=" * 70)

    output_dir = Path("experiments/results/next_token_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Corpus and Establish Leak-Free 3-Way Partition
    loader = CorpusLoader()
    sentences = loader.get_sentences()

    rng = random.Random(42)
    indices = list(range(len(sentences)))
    rng.shuffle(indices)
    train_idx = sorted(indices[:24])
    val_idx = sorted(indices[24:29])
    test_idx = sorted(indices[29:])

    train_sentences = [sentences[i] for i in train_idx]
    val_sentences = [sentences[i] for i in val_idx]
    test_sentences = [sentences[i] for i in test_idx]

    print(f"Corpus: {len(sentences)} sentences")
    print(f"Train split: {len(train_sentences)} sentences (295 tokens)")
    print(f"Val split:   {len(val_sentences)} sentences (62 tokens)")
    print(f"Test split:  {len(test_sentences)} sentences (58 tokens)")

    # 2. Train Models
    uni_model = UnigramModel().fit(train_sentences)
    bi_model = BigramModel().fit(train_sentences)
    tri_model = TrigramModel().fit(train_sentences)

    # Attach Best Performing Smoother (BackoffSmoother, d=0.5)
    backoff_smoother = BackoffSmoother(discount=0.5, model=tri_model)
    predictor = NextTokenPredictor(model=tri_model, smoother=backoff_smoother)

    print(f"\nPredictor initialized: Model={tri_model.name} (order={tri_model.n}), Smoother={backoff_smoother.name}")

    # 3. Demonstrate Predictions Across Contexts
    contexts = [
        (),
        ("the",),
        ("milo",),
        ("pip",),
        ("the", "crystal"),
        ("in", "the"),
        ("near", "the"),
        ("dark", "cave"),
        ("secret", "garden"),
        ("unknown_word",),
    ]

    prediction_examples: List[Dict[str, Any]] = []
    print("\n--- Next-Token Prediction Examples across Contexts ---")
    for ctx in contexts:
        greedy_res = predictor.predict_next(context=ctx, strategy="greedy")
        topk_res = predictor.predict_next(context=ctx, strategy="top_k", k=5, seed=42)
        temp_res = predictor.predict_next(context=ctx, strategy="temperature", temperature=0.8, seed=42)

        top3_tokens = [f"{c.token} ({c.probability:.3f})" for c in greedy_res.candidates[:3]]
        ctx_str = " ".join(ctx) if ctx else "<empty>"
        print(f"Context: [{ctx_str:18}] -> Top 3: {', '.join(top3_tokens)}")

        prediction_examples.append({
            "context": list(ctx),
            "greedy_selection": greedy_res.selected_token,
            "top_k_selection": topk_res.selected_token,
            "temperature_selection": temp_res.selected_token,
            "top_candidates": [
                {"rank": c.rank, "token": c.token, "probability": round(c.probability, 4)}
                for c in greedy_res.candidates[:5]
            ],
        })

    # Save prediction_examples.json
    examples_json_path = output_dir / "prediction_examples.json"
    with open(examples_json_path, "w", encoding="utf-8") as f:
        json.dump(prediction_examples, f, indent=2)
    print(f"Saved: {examples_json_path}")

    # 4. Export Candidate Rankings CSV for Sample Contexts
    rankings_csv_path = output_dir / "candidate_rankings.csv"
    with open(rankings_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Context", "Rank", "Token", "Probability", "Log_Probability"])

        export_contexts = [(), ("the",), ("milo",), ("the", "crystal"), ("near", "the")]
        for ctx in export_contexts:
            ctx_str = " ".join(ctx) if ctx else "<empty>"
            candidates = predictor.rank_candidates(context=ctx, top_n=10)
            for c in candidates:
                writer.writerow([ctx_str, c.rank, c.token, round(c.probability, 6), round(c.log_probability, 4)])
    print(f"Saved: {rankings_csv_path}")

    # 5. Full Evaluation Benchmark
    print("\n--- Running Prediction Benchmark on Held-Out Test Set (M=58) ---")
    evaluator = PredictionEvaluator(predictor=predictor, test_sentences=test_sentences)
    benchmark_data = evaluator.run_full_benchmark()

    # Save prediction_metrics.json
    metrics_json_path = output_dir / "prediction_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Saved: {metrics_json_path}")

    # Save decoding_comparison.csv
    decoding_csv_path = output_dir / "decoding_comparison.csv"
    with open(decoding_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Strategy",
            "Parameters",
            "Coverage_Pct",
            "Top1_Accuracy",
            "Top3_Accuracy",
            "Top5_Accuracy",
            "MRR",
            "Latency_Mean_ms",
            "Latency_P95_ms",
        ])
        for b in benchmark_data["benchmarks"]:
            writer.writerow([
                b["strategy"],
                str(b["params"]),
                f"{b['coverage_pct']}%",
                b["top1_accuracy"],
                b["top3_accuracy"],
                b["top5_accuracy"],
                b["mean_reciprocal_rank_mrr"],
                b["latency_mean_ms"],
                b["latency_p95_ms"],
            ])
    print(f"Saved: {decoding_csv_path}")

    # 6. Latency Benchmark CSV across Models and Smoothers
    latency_csv_path = output_dir / "prediction_latency.csv"
    with open(latency_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Smoother", "Context", "Strategy", "Trials", "Mean_Latency_ms", "P95_Latency_ms"])

        test_ctx = ("the", "crystal")
        models = [
            ("UnigramModel", uni_model, None),
            ("BigramModel", bi_model, None),
            ("TrigramModel (MLE)", tri_model, None),
            ("Trigram + Laplace", tri_model, LaplaceSmoother(model=tri_model)),
            ("Trigram + Backoff", tri_model, backoff_smoother),
        ]
        for name, m, sm in models:
            pred = NextTokenPredictor(model=m, smoother=sm)
            for strat in ["greedy", "top_k", "temperature"]:
                latencies = []
                for _ in range(50):
                    t0 = time.perf_counter()
                    pred.predict_next(context=test_ctx, strategy=strat, seed=42)
                    latencies.append((time.perf_counter() - t0) * 1000)
                latencies.sort()
                mean_l = sum(latencies) / len(latencies)
                p95_l = latencies[int(len(latencies) * 0.95)]
                writer.writerow([name, sm.name if sm else "None", "the crystal", strat, 50, round(mean_l, 4), round(p95_l, 4)])
    print(f"Saved: {latency_csv_path}")

    # 7. Generate Visualizations (Matplotlib only)
    print("\n--- Generating Visualizations ---")

    # Plot 1: Top Candidate Probability Distribution for Context "the crystal"
    ctx_sample = ("the", "crystal")
    sample_candidates = predictor.rank_candidates(context=ctx_sample, top_n=8)
    cand_tokens = [c.token for c in sample_candidates]
    cand_probs = [c.probability for c in sample_candidates]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(cand_tokens, cand_probs, color="#1f77b4", width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    ax.set_ylabel("Probability P(w | 'the', 'crystal')")
    ax.set_title("Next-Token Candidate Probability Distribution: Context 'the crystal'")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p1_path = output_dir / "top_candidate_distribution.png"
    fig.savefig(p1_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {p1_path}")

    # Plot 2: Top-k Candidate Comparison across k in {3, 5, 10}
    top10_cands = predictor.rank_candidates(context=ctx_sample, top_n=10)
    tokens_10 = [c.token for c in top10_cands]
    probs_raw = [c.probability for c in top10_cands]

    # Renormalized probabilities under k=3, k=5, k=10
    sum_3 = sum(probs_raw[:3])
    p_k3 = [p / sum_3 if i < 3 else 0.0 for i, p in enumerate(probs_raw)]

    sum_5 = sum(probs_raw[:5])
    p_k5 = [p / sum_5 if i < 5 else 0.0 for i, p in enumerate(probs_raw)]

    sum_10 = sum(probs_raw[:10])
    p_k10 = [p / sum_10 for p in probs_raw]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(tokens_10))
    w = 0.25
    ax.bar([i - w for i in x], p_k3, width=w, label="Top-k (k=3)", color="#2ca02c")
    ax.bar(x, p_k5, width=w, label="Top-k (k=5)", color="#1f77b4")
    ax.bar([i + w for i in x], p_k10, width=w, label="Top-k (k=10)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(tokens_10, rotation=25)
    ax.set_ylabel("Renormalized Sampling Probability")
    ax.set_title("Top-k Candidate Filtering & Renormalization Comparison: Context 'the crystal'")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p2_path = output_dir / "top_k_candidate_comparison.png"
    fig.savefig(p2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {p2_path}")

    # Plot 3: Temperature Effect on Probability Distribution
    temps = [0.1, 0.5, 1.0, 1.5, 2.0]
    top6_cands = predictor.rank_candidates(context=ctx_sample, top_n=6)
    tokens_6 = [c.token for c in top6_cands]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(tokens_6))
    w = 0.16
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

    for idx, (T, col) in enumerate(zip(temps, colors)):
        t_decoder = TemperatureDecoder(temperature=T)
        t_probs = t_decoder.compute_scaled_probabilities(top6_cands)
        offset = (idx - 2) * w
        ax.bar([i + offset for i in x], t_probs, width=w, label=f"T = {T}", color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(tokens_6)
    ax.set_ylabel("Temperature-Scaled Probability P_T(w)")
    ax.set_title("Temperature Scaling Effect on Candidate Probabilities: Context 'the crystal'")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p3_path = output_dir / "temperature_effect_distribution.png"
    fig.savefig(p3_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {p3_path}")

    # Plot 4: Decoding Strategy Comparison (Accuracy and Latency)
    b_data = benchmark_data["benchmarks"]
    labels = ["Greedy", "Top-k (k=3)", "Top-k (k=5)", "Top-k (k=10)", "T=0.1", "T=0.5", "T=1.0", "T=1.5", "T=2.0"]
    top1_acc = [b["top1_accuracy"] for b in b_data]
    mrr_vals = [b["mean_reciprocal_rank_mrr"] for b in b_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = range(len(labels))
    w = 0.35
    ax1.bar([i - w/2 for i in x], top1_acc, width=w, label="Top-1 Accuracy", color="#1f77b4")
    ax1.bar([i + w/2 for i in x], mrr_vals, width=w, label="MRR", color="#2ca02c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("Score [0.0 - 1.0]")
    ax1.set_title("Decoding Accuracy & MRR on Held-Out Test Set (M=58)")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    latencies = [b["latency_mean_ms"] for b in b_data]
    ax2.bar(labels, latencies, color="#ff7f0e", width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_ylabel("Mean Latency (ms)")
    ax2.set_title("Prediction Latency per Step across Decoding Strategies")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    p4_path = output_dir / "decoding_strategy_comparison.png"
    fig.savefig(p4_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {p4_path}")

    print("\nDemonstration and artifact generation complete.")


if __name__ == "__main__":
    main()
