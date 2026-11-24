"""Visualization generator for probability smoothing analysis (matplotlib only)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from five_head_nlp.smoothing.evaluator import SmoothingEvaluator


class SmoothingPlotter:
    """Generates comparative visualizations for smoothing techniques using Matplotlib."""

    def __init__(self, evaluator: Optional[SmoothingEvaluator] = None) -> None:
        """Initialize SmoothingPlotter."""
        self.evaluator: SmoothingEvaluator = evaluator or SmoothingEvaluator()

    def plot_all(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Generate all 6 required analytical charts."""
        target_dir = Path(output_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        results = self.evaluator.run_full_comparison()

        p1 = self.plot_mle_vs_laplace(target_dir / "mle_vs_laplace.png")
        p2 = self.plot_add_k_comparison(target_dir / "add_k_comparison.png")
        p3 = self.plot_good_turing_adjusted_counts(target_dir / "good_turing_adjusted_counts.png")
        p4 = self.plot_context_distributions(target_dir / "context_distributions.png")
        p5 = self.plot_perplexity_comparison(target_dir / "perplexity_comparison.png", results)
        p6 = self.plot_zero_probability_counts(target_dir / "zero_probability_counts.png", results)

        return {
            "mle_vs_laplace": p1,
            "add_k_comparison": p2,
            "good_turing_adjusted_counts": p3,
            "context_distributions": p4,
            "perplexity_comparison": p5,
            "zero_probability_counts": p6,
        }

    def plot_mle_vs_laplace(self, save_path: Path) -> Path:
        """Plot comparison of MLE vs Laplace probabilities for frequent tokens."""
        tokens = ["the", "crystal", "castle", "milo", "dark", "cave", "pip", "flew"]
        mle_probs = [self.evaluator.mle_trigram.probability(t) for t in tokens]
        laplace_smoother = self.evaluator.smoothers["Laplace (Add-1)"]
        laplace_probs = [laplace_smoother.probability(t) for t in tokens]

        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(len(tokens))
        width = 0.35

        ax.bar([i - width / 2 for i in x], mle_probs, width, label="Unsmoothed MLE", color="#1f77b4")
        ax.bar([i + width / 2 for i in x], laplace_probs, width, label="Laplace (Add-1)", color="#ff7f0e")

        ax.set_ylabel("Probability P(w)")
        ax.set_title("Unigram Probability: Unsmoothed MLE vs. Laplace Smoothing")
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=30)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_add_k_comparison(self, save_path: Path) -> Path:
        """Plot Add-k probabilities across k in {0.1, 0.5, 1.0}."""
        tokens = ["the", "crystal", "castle", "milo", "unseen_token"]
        add_01 = self.evaluator.smoothers["Add-k (k=0.1)"]
        add_05 = self.evaluator.smoothers["Add-k (k=0.5)"]
        add_10 = self.evaluator.smoothers["Laplace (Add-1)"]

        p_01 = [add_01.probability(t, context="near") for t in tokens]
        p_05 = [add_05.probability(t, context="near") for t in tokens]
        p_10 = [add_10.probability(t, context="near") for t in tokens]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(tokens))
        width = 0.25

        ax.bar([i - width for i in x], p_01, width, label="Add-k (k=0.1)", color="#2ca02c")
        ax.bar(x, p_05, width, label="Add-k (k=0.5)", color="#1f77b4")
        ax.bar([i + width for i in x], p_10, width, label="Add-k (k=1.0, Laplace)", color="#d62728")

        ax.set_ylabel("Probability P(w | 'near')")
        ax.set_title("Add-k Smoothing Comparison Given Context 'near'")
        ax.set_xticks(x)
        ax.set_xticklabels(tokens)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_good_turing_adjusted_counts(self, save_path: Path) -> Path:
        """Plot raw frequency r vs Good-Turing adjusted count r*."""
        gt_smoother = self.evaluator.smoothers["Good-Turing"]
        fof = gt_smoother.get_frequency_of_frequencies(order=2)
        raw_r = sorted(fof.keys())[:8]
        adj_r = [gt_smoother.get_adjusted_count(r, order=2) for r in raw_r]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(raw_r, raw_r, "k--", label="Raw Count (r)", marker="o")
        ax.plot(raw_r, adj_r, "r-", label="Good-Turing Adjusted Count (r*)", marker="s", linewidth=2)

        for x_val, y_val in zip(raw_r, adj_r):
            ax.annotate(f"{y_val:.2f}", (x_val, y_val), textcoords="offset points", xytext=(0, 7), ha="center")

        ax.set_xlabel("Empirical Frequency (r)")
        ax.set_ylabel("Adjusted Count (r*)")
        ax.set_title("Good-Turing Discounting on Bigram Counts")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_context_distributions(self, save_path: Path) -> Path:
        """Plot probability distribution for context 'crystal' across smoothers."""
        ctx = "crystal"
        candidates = ["castle", "garden", "cave", "forest", "the"]

        interp_key = next((k for k in self.evaluator.smoothers if "Fixed: 0.15, 0.35, 0.50" in k or "0.15, 0.35, 0.50" in k), None)
        methods = ["Unsmoothed MLE (Trigram)", "Add-k (k=0.5)", "Absolute Discounting Backoff"]
        if interp_key:
            methods.append(interp_key)

        labels = ["MLE", "Add-0.5", "Backoff", "Interpolation"][:len(methods)]
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"][:len(methods)]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(candidates))
        width = 0.2

        for idx, (method_name, short_label, color) in enumerate(zip(methods, labels, colors)):
            sm = self.evaluator.smoothers[method_name]
            probs = [sm.probability(tok, context=ctx) for tok in candidates]
            offset = (idx - 1.5) * width
            ax.bar([i + offset for i in x], probs, width, label=short_label, color=color)

        ax.set_ylabel("Probability P(w | 'crystal')")
        ax.set_title("Probability Distributions Given Context 'crystal'")
        ax.set_xticks(x)
        ax.set_xticklabels(candidates)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_perplexity_comparison(self, save_path: Path, results: Dict[str, Any]) -> Path:
        """Bar chart comparing held-out test perplexity across smoothers."""
        test_res = results.get("test_results") or results.get("validation_results", [])
        names = []
        perplexities = []

        for r in test_res:
            if r["perplexity"] != "inf":
                short_name = (
                    r["name"]
                    .replace("Linear Interpolation (Fixed: 0.15, 0.35, 0.50)", "Interp (Fixed)")
                    .replace("Absolute Discounting Backoff", "Backoff")
                    .replace("Laplace (Add-1)", "Laplace")
                )
                if "Optimized on Val" in short_name:
                    short_name = "Interp (Opt Val)"
                names.append(short_name)
                perplexities.append(float(r["perplexity"]))

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#8c564b", "#e377c2", "#9467bd", "#17becf"]
        bars = ax.bar(names, perplexities, color=colors[:len(names)], width=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom")

        test_tokens = results.get("split_info", {}).get("test_tokens", 58)
        ax.set_ylabel("Held-Out Test Perplexity (Lower is Better)")
        ax.set_title(f"Perplexity Comparison on Held-Out Test Sentences (M={test_tokens})")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_zero_probability_counts(self, save_path: Path, results: Dict[str, Any]) -> Path:
        """Bar chart showing zero-probability event count on held-out test set."""
        test_res = results.get("test_results") or results.get("validation_results", [])
        test_tokens = results.get("split_info", {}).get("test_tokens", 58)
        names = []
        for r in test_res:
            raw_name = (
                r["name"]
                .replace("Linear Interpolation (Fixed: 0.15, 0.35, 0.50)", "Interp (Fixed)")
                .replace("Absolute Discounting Backoff", "Backoff")
                .replace("Unsmoothed MLE (Trigram)", "MLE (Trigram)")
                .replace("Laplace (Add-1)", "Laplace")
            )
            if "Optimized on Val" in raw_name:
                raw_name = "Interp (Opt Val)"
            names.append(raw_name)

        zero_counts = [r["zero_probability_events"] for r in test_res]

        fig, ax = plt.subplots(figsize=(11, 5))
        colors = ["#d62728"] + ["#2ca02c"] * (len(names) - 1)
        bars = ax.bar(names, zero_counts, color=colors, width=0.55)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold")

        ax.set_ylabel("Zero-Probability Events Count")
        ax.set_title(f"Zero-Probability Event Count on Held-Out Test Set (Out of {test_tokens} Events)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path
