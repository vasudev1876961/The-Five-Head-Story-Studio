"""Corpus statistical visualization module using matplotlib.

Generates 6 publication-ready charts saved to experiments/results/corpus_analysis/:
1. top_20_words.png
2. top_20_bigrams.png
3. sentence_length_distribution.png
4. character_occurrences.png
5. setting_occurrences.png
6. vocabulary_frequency_distribution.png
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

from five_head_nlp.evaluation.corpus_statistics import CorpusStatistics


class CorpusPlotter:
    """Generates and exports statistical plots for a CorpusStatistics instance."""

    def __init__(self, statistics: CorpusStatistics) -> None:
        """Initialize the plotter.

        Args:
            statistics: CorpusStatistics instance to visualize.
        """
        self.stats = statistics

    def plot_top_20_words(self, output_path: str | Path) -> Path:
        """Horizontal bar chart of top 20 lexical word frequencies."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        analysis = self.stats.get_word_frequency_analysis()
        top_20 = analysis["top_20_lexical_words"]
        words = [item["word"] for item in top_20][::-1]
        freqs = [item["frequency"] for item in top_20][::-1]

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(words, freqs, color="#2b5c8f", edgecolor="#1a365d")
        ax.set_title("Top 20 Lexical Words by Frequency", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Frequency (Occurrences in Corpus)", fontsize=11)
        ax.set_ylabel("Lexical Word", fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        # Value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(width)}",
                    va="center", ha="left", fontsize=9, color="#222")

        ax.set_xlim(0, max(freqs) + 5)
        plt.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_top_20_bigrams(self, output_path: str | Path) -> Path:
        """Horizontal bar chart of top 20 within-sentence bigram frequencies."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        bigram_data = self.stats.get_bigram_statistics()
        top_20 = bigram_data["top_20_bigrams"]
        labels = [item["bigram"] for item in top_20][::-1]
        freqs = [item["frequency"] for item in top_20][::-1]

        fig, ax = plt.subplots(figsize=(10, 7.5))
        bars = ax.barh(labels, freqs, color="#3182ce", edgecolor="#2b6cb0")
        ax.set_title("Top 20 Within-Sentence Bigrams", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Frequency (Transitions in Corpus)", fontsize=11)
        ax.set_ylabel("Bigram Sequence (w1 -> w2)", fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.15, bar.get_y() + bar.get_height() / 2, f"{int(width)}",
                    va="center", ha="left", fontsize=9, color="#222")

        ax.set_xlim(0, max(freqs) + 1.5)
        plt.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_sentence_length_distribution(self, output_path: str | Path) -> Path:
        """Bar chart showing the distribution of sentence token lengths."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        basic = self.stats.get_basic_statistics()
        lengths = basic["sentence_lengths"]
        length_counts = Counter(lengths)
        sorted_lens = sorted(length_counts.keys())
        counts = [length_counts[l] for l in sorted_lens]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar([str(l) for l in sorted_lens], counts, color="#38a169", edgecolor="#276749", width=0.55)
        ax.set_title("Sentence Length Distribution (Tokens per Sentence)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Tokens in Sentence (Words + Period)", fontsize=11)
        ax.set_ylabel("Number of Sentences", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3, f"{int(height)}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylim(0, max(counts) + 3)
        plt.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_character_occurrences(self, output_path: str | Path) -> Path:
        """Bar chart comparing occurrences of the five recurring characters."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        entity_data = self.stats.get_entity_analysis()
        chars = entity_data["characters"]
        names = list(chars.keys())
        counts = list(chars.values())

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#d69e2e", "#dd6b20", "#e53e3e", "#805ad5", "#3182ce"]
        bars = ax.bar(names, counts, color=colors, edgecolor="#2d3748", width=0.55)
        ax.set_title("Character Mentions across Story Corpus", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Unigram Token Count", fontsize=11)
        ax.set_xlabel("Character Entity", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f"{int(height)}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylim(0, max(counts) + 2)
        plt.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_setting_occurrences(self, output_path: str | Path) -> Path:
        """Grouped bar chart comparing setting phrase and noun frequencies."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        entity_data = self.stats.get_entity_analysis()
        settings = ["Crystal Castle", "Dark Cave", "Secret Garden", "Bright Forest"]
        phrase_vals = [entity_data["setting_phrases"][s] for s in settings]
        noun_keys = ["Castle (noun)", "Cave (noun)", "Garden (noun)", "Forest (noun)"]
        noun_vals = [entity_data["setting_nouns"][k] for k in noun_keys]

        import numpy as np
        x = np.arange(len(settings))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5.5))
        rects1 = ax.bar(x - width / 2, phrase_vals, width, label="Compound Phrase (e.g. 'crystal castle')",
                        color="#4fd1c5", edgecolor="#285e61")
        rects2 = ax.bar(x + width / 2, noun_vals, width, label="Location Noun (e.g. 'castle')",
                        color="#319795", edgecolor="#234e52")

        ax.set_title("Setting Mentions: Exact Phrases vs. Location Nouns", fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(settings, fontsize=10)
        ax.set_ylabel("Occurrence Count", fontsize=11)
        ax.legend(frameon=True, facecolor="#f7fafc")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for rect in rects1:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.2, f"{int(h)}", ha="center", va="bottom", fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.2, f"{int(h)}", ha="center", va="bottom", fontsize=9)

        ax.set_ylim(0, max(noun_vals) + 2)
        plt.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_vocabulary_frequency_distribution(self, output_path: str | Path) -> Path:
        """Rank-frequency curve on log-log scale (Zipfian analysis)."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        vocab = self.stats.vocabulary
        # All tokens ranked by frequency
        freqs = [vocab.get_frequency(t) for t in vocab.tokens]
        ranks = list(range(1, len(freqs) + 1))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ranks, freqs, marker="o", markersize=3, color="#e53e3e", linestyle="-", linewidth=1.5, label="Observed Rank-Frequency")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Vocabulary Rank-Frequency Distribution (Log-Log Scale)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Token Rank (log scale)", fontsize=11)
        ax.set_ylabel("Token Frequency (log scale)", fontsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(frameon=True)

        plt.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def generate_all_plots(self, output_dir: str | Path) -> dict[str, Path]:
        """Generate all 6 statistical charts and save to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        return {
            "top_20_words": self.plot_top_20_words(out / "top_20_words.png"),
            "top_20_bigrams": self.plot_top_20_bigrams(out / "top_20_bigrams.png"),
            "sentence_length_distribution": self.plot_sentence_length_distribution(out / "sentence_length_distribution.png"),
            "character_occurrences": self.plot_character_occurrences(out / "character_occurrences.png"),
            "setting_occurrences": self.plot_setting_occurrences(out / "setting_occurrences.png"),
            "vocabulary_frequency_distribution": self.plot_vocabulary_frequency_distribution(out / "vocabulary_frequency_distribution.png"),
        }


def generate_corpus_visualizations(
    corpus_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Convenience function to load corpus, compute stats, and generate all plots."""
    project_root = Path(__file__).resolve().parents[3]
    if corpus_path is None:
        corpus_path = project_root / "data" / "raw" / "story_corpus.txt"
    if output_dir is None:
        output_dir = project_root / "experiments" / "results" / "corpus_analysis"

    loader = CorpusLoader(corpus_path).load()
    stats = CorpusStatistics(loader)
    plotter = CorpusPlotter(stats)
    return plotter.generate_all_plots(output_dir)


if __name__ == "__main__":
    generated = generate_corpus_visualizations()
    print("Generated statistical visualizations:")
    for k, p in generated.items():
        print(f"  - {k}: {p}")
