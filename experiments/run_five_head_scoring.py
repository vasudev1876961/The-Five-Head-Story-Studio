"""Demonstration and artifact generation script for Phase 4 — Five Transparent Scoring Heads."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from five_head_nlp.heads.evaluator import HeadEvaluator
from five_head_nlp.heads.hamilton_apportionment import HamiltonApportionment
from five_head_nlp.heads.head_orchestrator import HeadOrchestrator
from five_head_nlp.heads.scoring_config import ScoringConfig
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.prediction.predictor import NextTokenPredictor
from five_head_nlp.smoothing.backoff import BackoffSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


def main() -> None:
    print("=" * 70)
    print("Phase 4 — The Five Transparent Scoring Heads Demonstration")
    print("=" * 70)

    output_dir = Path("experiments/results/five_head_scoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Corpus and Initialize Trained Predictor
    loader = CorpusLoader()
    sentences = loader.get_sentences()

    # Leak-free 24-sentence training partition
    rng = random.Random(42)
    indices = list(range(len(sentences)))
    rng.shuffle(indices)
    train_idx = sorted(indices[:24])
    train_sentences = [sentences[i] for i in train_idx]

    model = TrigramModel().fit(train_sentences)
    smoother = BackoffSmoother(discount=0.5, model=model)
    predictor = NextTokenPredictor(model=model, smoother=smoother)

    # Initialize Head Orchestrator
    config = ScoringConfig()
    orchestrator = HeadOrchestrator(config=config)
    predictor.orchestrator = orchestrator

    print(f"Loaded {len(train_sentences)} training sentences.")
    print("Orchestrator initialized with 5 scoring heads:")
    for name, w in orchestrator.get_active_weights().items():
        print(f"  - {name:18}: weight = {w:.2f}")

    # 2. Candidate Evaluation & Score Breakdown
    ctx_sample = ("the", "crystal")
    base_candidates = predictor.rank_candidates(context=ctx_sample, top_n=8)
    orch_result = orchestrator.evaluate_candidates(context=ctx_sample, candidates=base_candidates)

    print(f"\n--- Context: {list(ctx_sample)} Candidate Multi-Head Scores ---")
    for fc in orch_result.fused_candidates:
        g = fc.grammar_score
        c = fc.character_score
        s = fc.setting_score
        a = fc.action_score
        r = fc.repeat_score
        print(f"Rank {fc.rank:2d}: [{fc.token:10}] Fused={fc.fused_score:.3f} | G={g:.3f} C={c:.3f} S={s:.3f} A={a:.3f} R={r:.3f}")

    # Save head_score_breakdown.csv
    breakdown_csv = output_dir / "head_score_breakdown.csv"
    with open(breakdown_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Context",
            "Token",
            "Fused_Rank",
            "Fused_Score",
            "Grammar_Score",
            "Grammar_Explanation",
            "Character_Score",
            "Character_Explanation",
            "Setting_Score",
            "Setting_Explanation",
            "Action_Score",
            "Action_Explanation",
            "Repeat_Score",
            "Repeat_Explanation",
        ])
        ctx_str = " ".join(ctx_sample)
        for fc in orch_result.fused_candidates:
            writer.writerow([
                ctx_str,
                fc.token,
                fc.rank,
                round(fc.fused_score, 4),
                round(fc.grammar_score, 4),
                fc.head_scores["Grammar Scout"].explanation,
                round(fc.character_score, 4),
                fc.head_scores["Character Keeper"].explanation,
                round(fc.setting_score, 4),
                fc.head_scores["Setting Painter"].explanation,
                round(fc.action_score, 4),
                fc.head_scores["Action Booster"].explanation,
                round(fc.repeat_score, 4),
                fc.head_scores["Repeat Blocker"].explanation,
            ])
    print(f"Saved: {breakdown_csv}")

    # 3. Candidate Fusion under Different Weight Profiles
    profiles = [
        ("Balanced (Default)", {"Grammar Scout": 0.40, "Character Keeper": 0.15, "Setting Painter": 0.15, "Action Booster": 0.10, "Repeat Blocker": 0.20}),
        ("Grammar-Dominated", {"Grammar Scout": 0.80, "Character Keeper": 0.05, "Setting Painter": 0.05, "Action Booster": 0.05, "Repeat Blocker": 0.05}),
        ("Character-Heavy", {"Grammar Scout": 0.30, "Character Keeper": 0.45, "Setting Painter": 0.10, "Action Booster": 0.05, "Repeat Blocker": 0.10}),
        ("Action-Heavy", {"Grammar Scout": 0.30, "Character Keeper": 0.10, "Setting Painter": 0.10, "Action Booster": 0.40, "Repeat Blocker": 0.10}),
        ("Setting-Heavy", {"Grammar Scout": 0.30, "Character Keeper": 0.10, "Setting Painter": 0.45, "Action Booster": 0.05, "Repeat Blocker": 0.10}),
    ]

    fusion_csv = output_dir / "candidate_fusion_results.csv"
    sample_ctx_list = [("the", "crystal"), ("near", "the"), ("milo",), ("pip",)]
    with open(fusion_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Profile", "Context", "Rank", "Token", "Fused_Score", "Original_Probability"])

        for prof_name, w_dict in profiles:
            temp_config = ScoringConfig(
                grammar_weight=w_dict["Grammar Scout"],
                character_weight=w_dict["Character Keeper"],
                setting_weight=w_dict["Setting Painter"],
                action_weight=w_dict["Action Booster"],
                repeat_weight=w_dict["Repeat Blocker"],
            )
            temp_orch = HeadOrchestrator(config=temp_config)
            for c_tuple in sample_ctx_list:
                c_str = " ".join(c_tuple)
                cands = predictor.rank_candidates(context=c_tuple, top_n=5)
                fused = temp_orch.evaluate_candidates(context=c_tuple, candidates=cands)
                for fc in fused.fused_candidates:
                    writer.writerow([prof_name, c_str, fc.rank, round(fc.fused_score, 4), round(fc.original_probability, 4)])
    print(f"Saved: {fusion_csv}")

    # 4. Hamilton d10 Allocation Examples
    apportioner = HamiltonApportionment(total_points=10)
    hamilton_examples = {}
    for prof_name, w_dict in profiles:
        alloc = apportioner.apportion(w_dict)
        hamilton_examples[prof_name] = {
            "weights": {k: round(v, 3) for k, v in w_dict.items()},
            "d10_allocation": alloc,
            "total_points": sum(alloc.values()),
        }

    hamilton_json = output_dir / "hamilton_allocation.json"
    with open(hamilton_json, "w", encoding="utf-8") as f:
        json.dump(hamilton_examples, f, indent=2)
    print(f"Saved: {hamilton_json}")

    # 5. Scoring Configuration JSON
    config_json = output_dir / "scoring_configuration.json"
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Saved: {config_json}")

    # 6. Evaluation Ablation Benchmark
    print("\n--- Running Head Ablation Benchmark on Test Split ---")
    evaluator = HeadEvaluator(predictor=predictor)
    ablation_results = evaluator.run_full_ablation()

    metrics_json = output_dir / "head_evaluation_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Saved: {metrics_json}")

    # 7. Generate Visualizations (Matplotlib only)
    print("\n--- Generating Visualizations ---")

    # Plot 1: Individual Head Scores Comparison
    top_cands = orch_result.fused_candidates[:6]
    cand_labels = [c.token for c in top_cands]
    x = range(len(cand_labels))
    w = 0.16

    fig, ax = plt.subplots(figsize=(12, 5))
    heads_meta = [
        ("Grammar Scout", [c.grammar_score for c in top_cands], "#1f77b4"),
        ("Character Keeper", [c.character_score for c in top_cands], "#2ca02c"),
        ("Setting Painter", [c.setting_score for c in top_cands], "#ff7f0e"),
        ("Action Booster", [c.action_score for c in top_cands], "#d62728"),
        ("Repeat Blocker", [c.repeat_score for c in top_cands], "#9467bd"),
    ]

    for idx, (h_name, scores, col) in enumerate(heads_meta):
        offset = (idx - 2) * w
        ax.bar([i + offset for i in x], scores, width=w, label=h_name, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(cand_labels)
    ax.set_ylabel("Normalized Head Score [0.0 - 1.0]")
    ax.set_title("Individual Head Scores for Top Candidates: Context 'the crystal'")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p1 = output_dir / "individual_head_scores.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"Saved: {p1}")

    # Plot 2: Weighted Fusion Contribution Stacked Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    weights = orchestrator.fusion.normalize_weights(orchestrator.get_active_weights())

    bottom = [0.0] * len(cand_labels)
    for h_name, scores, col in heads_meta:
        weighted_scores = [s * weights.get(h_name, 0.0) for s in scores]
        ax.bar(cand_labels, weighted_scores, bottom=bottom, label=f"{h_name} (w={weights[h_name]:.2f})", color=col, width=0.5)
        bottom = [b + ws for b, ws in zip(bottom, weighted_scores)]

    for i, total_h in enumerate(bottom):
        ax.annotate(f"{total_h:.3f}", xy=(i, total_h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Fused Score Contribution")
    ax.set_title("Stacked Multi-Head Contribution to Fused Score: Context 'the crystal'")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p2 = output_dir / "weighted_fusion_contributions.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"Saved: {p2}")

    # Plot 3: Hamilton d10 Integer Allocation Chart
    d10_default = hamilton_examples["Balanced (Default)"]["d10_allocation"]
    head_names = list(d10_default.keys())
    points = [d10_default[h] for h in head_names]
    h_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(head_names, points, color=h_colors, width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h} pts", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("d10 Integer Die Points (Sum = 10)")
    ax.set_title("Hamilton Largest-Remainder d10 Allocation (Balanced Profile)")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, 6)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p3 = output_dir / "hamilton_d10_allocation.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"Saved: {p3}")

    # Plot 4: Baseline vs Five-Head Candidate Rankings
    cands_comp = [c.token for c in base_candidates[:6]]
    baseline_ranks = list(range(1, len(cands_comp) + 1))
    five_head_ranks = [orch_result.get_candidate(tok).rank for tok in cands_comp]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(cands_comp))
    w = 0.35
    ax.bar([i - w/2 for i in x], baseline_ranks, width=w, label="Baseline Rank (Pure N-gram)", color="#7f7f7f")
    ax.bar([i + w/2 for i in x], five_head_ranks, width=w, label="Five-Head Fused Rank", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(cands_comp)
    ax.set_ylabel("Rank Position (1 = Highest Priority)")
    ax.set_title("Candidate Rank Shift: Baseline N-Gram vs. Five-Head Scoring")
    ax.legend()
    ax.invert_yaxis()  # Rank 1 at top
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p4 = output_dir / "baseline_vs_five_head_rankings.png"
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    print(f"Saved: {p4}")

    # Plot 5: Repetition Penalty Decay Curve
    counts = list(range(6))
    gammas = [0.4, 0.8, 1.2]
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in gammas:
        decay_vals = [math.exp(-g * c) for c in counts]
        ax.plot(counts, decay_vals, marker="o", label=f"Gamma = {g}")
    ax.set_xlabel("Recent Occurrence Count in Window")
    ax.set_ylabel("Repeat Blocker Multiplier Score R(w)")
    ax.set_title("Repeat Blocker Non-Linear Frequency Decay: R(w) = exp(-gamma * count)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    p5 = output_dir / "repetition_penalty_decay.png"
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    print(f"Saved: {p5}")

    # Plot 6: Character and Setting Tracking Across Rollout
    rollout_tokens, _ = predictor.generate(
        prompt=["milo", "walked", "into", "the"],
        max_tokens=15,
        strategy="greedy",
        mode="five_head",
    )
    tokens_x = list(range(1, len(rollout_tokens) + 1))
    has_char = [1 if t in orchestrator.character_head.all_character_tokens else 0 for t in rollout_tokens]
    has_setting = [1 if t in orchestrator.setting_head.all_setting_tokens else 0 for t in rollout_tokens]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.step(tokens_x, has_char, where="mid", label="Character Mention", color="#2ca02c", linewidth=2)
    ax.step(tokens_x, has_setting, where="mid", label="Setting Mention", color="#ff7f0e", linewidth=2, linestyle="--")
    ax.set_xticks(tokens_x)
    ax.set_xticklabels(rollout_tokens, rotation=35, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Absent", "Present"])
    ax.set_title("Thematic Entity Tracking Along Sequence Rollout")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p6 = output_dir / "character_setting_tracking.png"
    fig.savefig(p6, dpi=300)
    plt.close(fig)
    print(f"Saved: {p6}")

    print("\nDemonstration and artifact generation complete.")


if __name__ == "__main__":
    main()
