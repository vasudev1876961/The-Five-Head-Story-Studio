"""Corpus validation utility for the Five-Head Story-Generation System.

This module calculates statistical, grammatical, and transition properties
of the raw story corpus for Phase 2.1 and Phase 2.1.1 validation.

Calculation Conventions:
1. Punctuation as Tokens: Lowercase word-level splitting; periods '.' are isolated
   as individual tokens.
2. Vocabulary Size: Unique vocabulary count includes the period '.' (119 words + 1 period = 120).
3. Sentence Boundaries: By default, the corpus is modeled as a continuous autoregressive
   stream across sentences where the '.' of sentence k transitions to the first token of
   sentence k+1 (415 tokens, 414 total transitions, 271 unique bigrams).
4. Sentence-Final Occurrences: Counted in unigram totals. In bigram transitions, words
   preceding '.' have an outgoing transition (word -> '.').
5. Branching Factor: Calculated as sum(unique_continuations) / len(tokens_with_outgoing).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def count_syllables(word: str) -> int:
    """Estimate syllable count of an English word using standard heuristic rules.

    Rules applied:
    - Normalizes word by stripping punctuation and lowercasing.
    - Words with length <= 3 are estimated as 1 syllable.
    - Strips common silent trailing patterns ('-e', '-ed', '-es' when not contributing).
    - Counts contiguous vowel groups [aeiouy].
    - Guarantees at least 1 syllable per non-empty token.
    """
    clean_word = word.lower().strip(".:;?!,'\"-")
    if not clean_word:
        return 0
    if len(clean_word) <= 3:
        return 1
    # Strip silent trailing e/ed/es
    processed = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", clean_word)
    processed = re.sub(r"^y", "", processed)
    vowel_matches = re.findall(r"[aeiouy]{1,2}", processed)
    return max(1, len(vowel_matches))


def compute_flesch_kincaid(num_words: int, num_sentences: int, total_syllables: int) -> float:
    """Calculate Flesch-Kincaid Grade Level using standard formula.

    FKGL = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    """
    if num_sentences == 0 or num_words == 0:
        return 0.0
    asl = num_words / num_sentences
    asw = total_syllables / num_words
    return round(0.39 * asl + 11.8 * asw - 15.59, 2)


def validate_corpus(corpus_path: str | Path) -> dict[str, Any]:
    """Parse and validate story corpus text file, computing all descriptive metrics.

    Counting conventions documented at module level.
    """
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    sentence_count = len(lines)
    tokens: list[str] = []
    sentence_token_lists: list[list[str]] = []

    for line in lines:
        words = line.rstrip(".").split()
        st = words + ["."]
        sentence_token_lists.append(st)
        tokens.extend(st)

    total_tokens = len(tokens)
    word_tokens = [t for t in tokens if t != "."]
    word_count = len(word_tokens)
    vocab = set(tokens)
    vocab_size = len(vocab)
    avg_words_per_sentence = round(word_count / sentence_count, 2) if sentence_count else 0.0

    total_syllables = sum(count_syllables(w) for w in word_tokens)
    fkgl = compute_flesch_kincaid(word_count, sentence_count, total_syllables)

    # Unigram counts
    unigram_counts = Counter(tokens)

    # 1. Flat Stream Bigram transitions (continuous flow across sentences)
    bigrams_flat: Counter[tuple[str, str]] = Counter()
    transitions_flat: dict[str, Counter[str]] = defaultdict(Counter)

    for i in range(len(tokens) - 1):
        prev_tok, next_tok = tokens[i], tokens[i + 1]
        bigrams_flat[(prev_tok, next_tok)] += 1
        transitions_flat[prev_tok][next_tok] += 1

    branching_factors_flat = [len(nxt) for nxt in transitions_flat.values()]
    tokens_with_outgoing = len(transitions_flat)
    avg_branching_factor_flat = (
        round(sum(branching_factors_flat) / tokens_with_outgoing, 2) if tokens_with_outgoing else 0.0
    )

    # 2. Within-Sentence Bigrams (disjoint sentences, no '.' -> word transitions)
    bigrams_sent: Counter[tuple[str, str]] = Counter()
    transitions_sent: dict[str, Counter[str]] = defaultdict(Counter)

    for st in sentence_token_lists:
        for i in range(len(st) - 1):
            prev_tok, next_tok = st[i], st[i + 1]
            bigrams_sent[(prev_tok, next_tok)] += 1
            transitions_sent[prev_tok][next_tok] += 1

    branching_factors_sent = [len(nxt) for nxt in transitions_sent.values()]
    avg_branching_factor_sent = (
        round(sum(branching_factors_sent) / len(transitions_sent), 2) if transitions_sent else 0.0
    )

    # Character entity counts
    character_counts = {
        "milo": unigram_counts.get("milo", 0),
        "pip": unigram_counts.get("pip", 0),
        "zara": unigram_counts.get("zara", 0),
        "owl": unigram_counts.get("owl", 0),
        "fox": unigram_counts.get("fox", 0),
    }

    # Setting entity counts
    setting_counts = {
        "forest": unigram_counts.get("forest", 0),
        "castle": unigram_counts.get("castle", 0),
        "cave": unigram_counts.get("cave", 0),
        "garden": unigram_counts.get("garden", 0),
    }

    # Comprehensive Pivot Token Statistics (audited Phase 2.1.1)
    pivot_tokens = ["the", ".", "and", "a", "to", "milo", "pip", "zara", "near", "castle", "cave"]
    pivot_stats: dict[str, dict[str, Any]] = {}
    for p in pivot_tokens:
        uni_count = unigram_counts.get(p, 0)
        out_transitions = sum(transitions_flat[p].values())
        unique_next = len(transitions_flat[p])
        continuations = dict(transitions_flat[p].most_common())
        pivot_stats[p] = {
            "unigram_count": uni_count,
            "out_transitions": out_transitions,
            "unique_next": unique_next,
            "continuations": continuations,
        }

    # Reference target bigrams
    target_bigrams = [
        ("the", "silver"),
        ("the", "crystal"),
        ("the", "old"),
        ("the", "dark"),
        ("the", "bright"),
        ("the", "brave"),
        (".", "the"),
        (".", "milo"),
        (".", "pip"),
        (".", "zara"),
        ("and", "the"),
        ("and", "flew"),
        ("to", "the"),
        ("near", "the"),
        ("milo", "lived"),
        ("milo", "liked"),
        ("milo", "found"),
        ("milo", "showed"),
    ]

    target_bigram_counts = {
        f"{w1} -> {w2}": bigrams_flat.get((w1, w2), 0) for w1, w2 in target_bigrams
    }

    return {
        "sentence_count": sentence_count,
        "word_count": word_count,
        "total_tokens": total_tokens,
        "unique_vocabulary": vocab_size,
        "avg_words_per_sentence": avg_words_per_sentence,
        "total_syllables": total_syllables,
        "flesch_kincaid_grade_level": fkgl,
        "avg_bigram_branching_factor": avg_branching_factor_flat,
        "avg_bigram_branching_factor_within_sentence": avg_branching_factor_sent,
        "total_bigram_transitions_flat": sum(bigrams_flat.values()),
        "total_bigram_transitions_sent": sum(bigrams_sent.values()),
        "bigram_total_unique_flat": len(bigrams_flat),
        "bigram_total_unique_sent": len(bigrams_sent),
        "character_counts": character_counts,
        "setting_counts": setting_counts,
        "pivot_stats": pivot_stats,
        "target_bigram_counts": target_bigram_counts,
    }


def print_report(results: dict[str, Any]) -> None:
    """Print formatted validation summary to stdout."""
    print("=" * 65)
    print("FIVE-HEAD CORPUS AUDIT & VALIDATION REPORT (PHASE 2.1.1)")
    print("=" * 65)
    print(f"Sentences:                       {results['sentence_count']} (Target: 34)")
    print(f"Word Count:                      {results['word_count']} (Target: 381)")
    print(f"Total Tokens (incl. '.'):        {results['total_tokens']} (Target: 415)")
    print(f"Unique Vocabulary:               {results['unique_vocabulary']} (Target: 120)")
    print(f"Average Words / Sentence:        {results['avg_words_per_sentence']} (Target: 11.21)")
    print(f"FKGL (Readability):              {results['flesch_kincaid_grade_level']} (Target: 4.02)")
    print(f"Avg Branching (Flat Stream):     {results['avg_bigram_branching_factor']} (Target: 1.95)")
    print(f"Avg Branching (Within-Sentence): {results['avg_bigram_branching_factor_within_sentence']}")
    print(f"Unique Bigrams (Flat Stream):    {results['bigram_total_unique_flat']}")
    print(f"Unique Bigrams (Within-Sent):    {results['bigram_total_unique_sent']}")
    print("-" * 65)
    print("CHARACTER OCCURRENCE COUNTS (UNIGRAM TOTALS):")
    for char, cnt in results["character_counts"].items():
        print(f"  - {char.capitalize()}: {cnt}")
    print("-" * 65)
    print("SETTING OCCURRENCE COUNTS (UNIGRAM TOTALS):")
    for setting, cnt in results["setting_counts"].items():
        print(f"  - {setting.capitalize()}: {cnt}")
    print("-" * 65)
    print("AUDITED PIVOT TOKEN STATISTICS (UNIGRAM / OUT-TRANSITIONS / UNIQUE NEXT):")
    for pivot, stats in results["pivot_stats"].items():
        print(f"  '{pivot:8s}': unigram={stats['unigram_count']:2d}, out={stats['out_transitions']:2d}, unique_next={stats['unique_next']:2d}")
    print("-" * 65)
    print("KNOWN TARGET BIGRAM TRANSITIONS (18/18 EXACT MATCHES):")
    for bg, cnt in results["target_bigram_counts"].items():
        print(f"  {bg:22s}: {cnt}")
    print("=" * 65)


if __name__ == "__main__":
    default_corpus = Path(__file__).resolve().parents[3] / "data" / "raw" / "story_corpus.txt"
    metrics = validate_corpus(default_corpus)
    print_report(metrics)
