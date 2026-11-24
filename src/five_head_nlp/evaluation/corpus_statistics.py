"""Corpus statistical analysis and exploratory data analysis module.

Computes comprehensive lexical distributions, within-sentence bigram transitions,
sentence length profiles, and entity metrics for the Five-Head Story-Generation System.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from five_head_nlp.tokenization.corpus_loader import CorpusLoader, is_punctuation_token
from five_head_nlp.tokenization.vocabulary import Vocabulary


class CorpusStatistics:
    """Computes and exports descriptive and structural corpus statistics."""

    def __init__(
        self,
        source: CorpusLoader | list[list[str]],
        vocabulary: Vocabulary | None = None,
    ) -> None:
        """Initialize the statistics engine.

        Args:
            source: A CorpusLoader instance or pre-tokenized list of sentence token lists.
            vocabulary: Optional Vocabulary instance. If None, constructed dynamically
                        from the input sentences.
        """
        if isinstance(source, CorpusLoader):
            self._sentences = source.get_sentences()
            self._corpus_path = str(source.corpus_path)
        elif isinstance(source, list):
            self._sentences = [list(s) for s in source]
            self._corpus_path = "in_memory"
        else:
            raise TypeError(
                f"Expected CorpusLoader or list of sentence token lists, got {type(source).__name__}"
            )

        if not self._sentences:
            raise ValueError("Cannot compute statistics on an empty corpus.")

        self._flat_tokens = [t for s in self._sentences for t in s]

        if vocabulary is not None:
            self.vocabulary = vocabulary
        else:
            self.vocabulary = Vocabulary(self._flat_tokens)

        # Precompute within-sentence bigrams and transitions
        self._compute_within_sentence_bigrams()

    def _compute_within_sentence_bigrams(self) -> None:
        """Calculate adjacent-token bigrams strictly within sentence boundaries.

        Rules:
        - Bigrams are only created between tokens adjacent inside the SAME sentence.
        - Sentence-ending periods do NOT transition to the first token of the next sentence.
        - The period token has 0 outgoing transitions in this model.
        """
        self._bigrams: Counter[tuple[str, str]] = Counter()
        self._transitions: dict[str, Counter[str]] = defaultdict(Counter)

        for sent in self._sentences:
            for i in range(len(sent) - 1):
                prev_tok, next_tok = sent[i], sent[i + 1]
                self._bigrams[(prev_tok, next_tok)] += 1
                self._transitions[prev_tok][next_tok] += 1

    # --- Basic Corpus Statistics ---

    def get_basic_statistics(self) -> dict[str, Any]:
        """Compute basic corpus dimension, length, and vocabulary metrics."""
        sentence_lengths = [len(s) for s in self._sentences]
        num_sentences = len(self._sentences)
        total_tokens = len(self._flat_tokens)
        word_tokens = sum(1 for t in self._flat_tokens if not is_punctuation_token(t))
        punctuation_tokens = sum(1 for t in self._flat_tokens if is_punctuation_token(t))

        avg_length = total_tokens / num_sentences if num_sentences else 0.0
        min_length = min(sentence_lengths) if sentence_lengths else 0
        max_length = max(sentence_lengths) if sentence_lengths else 0
        median_length = statistics.median(sentence_lengths) if sentence_lengths else 0.0

        ttr = self.vocabulary.size / total_tokens if total_tokens else 0.0
        lexical_ttr = self.vocabulary.lexical_size / word_tokens if word_tokens else 0.0

        return {
            "num_sentences": num_sentences,
            "total_tokens": total_tokens,
            "word_tokens": word_tokens,
            "punctuation_tokens": punctuation_tokens,
            "unique_full_vocabulary": self.vocabulary.size,
            "unique_lexical_vocabulary": self.vocabulary.lexical_size,
            "avg_sentence_length": round(avg_length, 2),
            "min_sentence_length": min_length,
            "max_sentence_length": max_length,
            "median_sentence_length": median_length,
            "sentence_lengths": sentence_lengths,
            "type_token_ratio_full": round(ttr, 4),
            "type_token_ratio_lexical": round(lexical_ttr, 4),
        }

    # --- Word-Frequency Analysis ---

    def get_word_frequency_analysis(self) -> dict[str, Any]:
        """Compute lexical word frequency distributions, hapax legomena, and probabilities."""
        total_tokens_n = len(self._flat_tokens)

        # Lexical words only (descending frequency, alphabetical ties)
        lexical_tokens = self.vocabulary.lexical_tokens
        frequencies = {w: self.vocabulary.get_frequency(w) for w in lexical_tokens}

        # Top 20 lexical words
        top_20 = [
            {"word": w, "frequency": frequencies[w], "probability": round(frequencies[w] / total_tokens_n, 6)}
            for w in lexical_tokens[:20]
        ]

        # Hapax legomena: words appearing exactly once
        hapax = [w for w, count in frequencies.items() if count == 1]

        # Least frequent words
        min_freq = min(frequencies.values()) if frequencies else 0
        least_frequent = [w for w, count in frequencies.items() if count == min_freq]

        # Frequency distribution summary (count of words by occurrence count)
        freq_dist: Counter[int] = Counter(frequencies.values())
        freq_dist_sorted = dict(sorted(freq_dist.items()))

        return {
            "top_20_lexical_words": top_20,
            "hapax_legomena_count": len(hapax),
            "hapax_legomena_ratio": round(len(hapax) / len(lexical_tokens), 4) if lexical_tokens else 0.0,
            "hapax_legomena": sorted(hapax),
            "min_frequency": min_freq,
            "least_frequent_words_count": len(least_frequent),
            "frequency_distribution": freq_dist_sorted,
        }

    # --- Bigram Statistics ---

    def get_bigram_statistics(self) -> dict[str, Any]:
        """Compute within-sentence bigram frequencies, transitions, and branching factors."""
        total_transitions = sum(self._bigrams.values())
        unique_bigrams = len(self._bigrams)

        # Sort bigrams descending by frequency, then alphabetical
        sorted_bigrams = sorted(
            self._bigrams.keys(),
            key=lambda bg: (-self._bigrams[bg], bg[0], bg[1]),
        )

        top_20_bigrams = [
            {"bigram": f"{bg[0]} {bg[1]}", "w1": bg[0], "w2": bg[1], "frequency": self._bigrams[bg]}
            for bg in sorted_bigrams[:20]
        ]

        # Outgoing continuations per token
        continuations_per_token = {
            t: len(self._transitions[t]) for t in self.vocabulary.tokens
        }

        # Branching factor calculations
        sum_out_degrees = sum(continuations_per_token.values())
        # Denominator A: Full vocabulary (including period token '.')
        denom_full = self.vocabulary.size
        branching_full = sum_out_degrees / denom_full if denom_full else 0.0

        # Denominator B: Active tokens only (tokens with out-degree > 0)
        active_tokens = [t for t, deg in continuations_per_token.items() if deg > 0]
        denom_active = len(active_tokens)
        branching_active = sum_out_degrees / denom_active if denom_active else 0.0

        return {
            "total_bigram_transitions": total_transitions,
            "unique_bigrams_count": unique_bigrams,
            "top_20_bigrams": top_20_bigrams,
            "continuations_per_token": continuations_per_token,
            "active_tokens_count": denom_active,
            "average_branching_factor_full_vocab": round(branching_full, 4),
            "average_branching_factor_active_vocab": round(branching_active, 4),
            "bigrams_dict": {f"{w1} {w2}": self._bigrams[(w1, w2)] for w1, w2 in sorted_bigrams},
        }

    # --- Pivot-Token Analysis ---

    def get_pivot_token_analysis(self) -> list[dict[str, Any]]:
        """Compute occurrence counts, outgoing transitions, and unique next tokens for key pivots."""
        pivot_tokens = [
            "the", ".", "and", "a", "to", "near", "milo", "pip", "zara", "castle", "cave"
        ]

        results = []
        for p in pivot_tokens:
            unigram_freq = self.vocabulary.get_frequency(p)
            out_trans = sum(self._transitions[p].values())
            uniq_next = len(self._transitions[p])
            next_distribution = [
                {"next_token": nxt, "count": cnt}
                for nxt, cnt in sorted(self._transitions[p].items(), key=lambda x: (-x[1], x[0]))
            ]
            results.append({
                "token": p,
                "unigram_frequency": unigram_freq,
                "outgoing_transitions": out_trans,
                "unique_next_tokens": uniq_next,
                "continuations": next_distribution,
            })
        return results

    # --- Character and Setting Analysis ---

    def get_entity_analysis(self) -> dict[str, Any]:
        """Compute occurrences of recurring characters and settings."""
        # Characters: count canonical unigram occurrences in the text
        characters = {
            "Milo": self.vocabulary.get_frequency("milo"),
            "Pip": self.vocabulary.get_frequency("pip"),
            "Zara": self.vocabulary.get_frequency("zara"),
            "Sleepy Owl": self.vocabulary.get_frequency("owl"),
            "Playful Fox": self.vocabulary.get_frequency("fox"),
        }

        # Settings: measure both exact phrase bigrams and location nouns
        # Phrases: "crystal castle", "dark cave", "secret garden", "bright forest"
        phrase_counts = {
            "Crystal Castle": self._bigrams.get(("crystal", "castle"), 0),
            "Dark Cave": self._bigrams.get(("dark", "cave"), 0),
            "Secret Garden": self._bigrams.get(("secret", "garden"), 0),
            "Bright Forest": self._bigrams.get(("bright", "forest"), 0),
        }

        noun_counts = {
            "Castle (noun)": self.vocabulary.get_frequency("castle"),
            "Cave (noun)": self.vocabulary.get_frequency("cave"),
            "Garden (noun)": self.vocabulary.get_frequency("garden"),
            "Forest (noun)": self.vocabulary.get_frequency("forest"),
        }

        return {
            "characters": characters,
            "setting_phrases": phrase_counts,
            "setting_nouns": noun_counts,
        }

    # --- Comprehensive Aggregation & Export ---

    def compute_all_statistics(self) -> dict[str, Any]:
        """Aggregate all analyses into a single comprehensive dictionary."""
        return {
            "corpus_path": self._corpus_path,
            "basic_statistics": self.get_basic_statistics(),
            "word_frequency_analysis": self.get_word_frequency_analysis(),
            "bigram_statistics": self.get_bigram_statistics(),
            "pivot_token_analysis": self.get_pivot_token_analysis(),
            "entity_analysis": self.get_entity_analysis(),
        }

    def save_artifacts(self, output_dir: str | Path) -> dict[str, Path]:
        """Export JSON report and CSV tables to the designated output directory.

        Generates:
        1. corpus_statistics.json
        2. bigram_frequencies.csv
        3. pivot_token_statistics.csv

        Args:
            output_dir: Target directory for statistical artifacts.

        Returns:
            Dict mapping artifact names to their written Path destinations.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_stats = self.compute_all_statistics()

        # 1. corpus_statistics.json
        json_path = out_dir / "corpus_statistics.json"
        json_path.write_text(
            json.dumps(all_stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 2. bigram_frequencies.csv
        bigram_csv = out_dir / "bigram_frequencies.csv"
        with bigram_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prev_token", "next_token", "frequency"])
            sorted_bgs = sorted(
                self._bigrams.keys(),
                key=lambda bg: (-self._bigrams[bg], bg[0], bg[1]),
            )
            for w1, w2 in sorted_bgs:
                writer.writerow([w1, w2, self._bigrams[(w1, w2)]])

        # 3. pivot_token_statistics.csv
        pivots_csv = out_dir / "pivot_token_statistics.csv"
        with pivots_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["token", "unigram_frequency", "outgoing_transitions", "unique_continuations"])
            for p in self.get_pivot_token_analysis():
                writer.writerow([
                    p["token"],
                    p["unigram_frequency"],
                    p["outgoing_transitions"],
                    p["unique_next_tokens"],
                ])

        return {
            "statistics_json": json_path,
            "bigram_csv": bigram_csv,
            "pivot_csv": pivots_csv,
        }
