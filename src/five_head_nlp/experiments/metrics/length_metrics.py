"""Sentence and token length metrics for generated stories."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Sequence


def calculate_length_metrics(tokens: Sequence[str]) -> Dict[str, Any]:
    """Calculate token count, sentence count, and sentence length statistics.

    Parameters
    ----------
    tokens : Sequence[str]
        Full token sequence of the story.

    Returns
    -------
    Dict[str, Any]
        Dictionary of length metrics.
    """
    total_tokens = len(tokens)
    words = [t for t in tokens if t not in {".", ",", "!", "?"}]
    total_words = len(words)

    # Segment tokens into sentences based on terminal period '.'
    sentences: List[List[str]] = []
    current_sentence: List[str] = []

    for t in tokens:
        current_sentence.append(t)
        if t == ".":
            sentences.append(current_sentence)
            current_sentence = []

    # If trailing tokens exist without a terminal period, count them as an incomplete sentence
    if current_sentence:
        sentences.append(current_sentence)

    sentence_count = len(sentences)
    sentence_lengths = [len(s) for s in sentences] if sentences else [0]

    mean_len = float(statistics.mean(sentence_lengths)) if sentence_lengths else 0.0
    median_len = float(statistics.median(sentence_lengths)) if sentence_lengths else 0.0
    min_len = int(min(sentence_lengths)) if sentence_lengths else 0
    max_len = int(max(sentence_lengths)) if sentence_lengths else 0
    std_len = float(statistics.stdev(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0

    return {
        "total_tokens": total_tokens,
        "total_words": total_words,
        "sentence_count": sentence_count,
        "sentence_lengths": sentence_lengths,
        "mean_sentence_length": round(mean_len, 4),
        "median_sentence_length": round(median_len, 4),
        "min_sentence_length": min_len,
        "max_sentence_length": max_len,
        "std_sentence_length": round(std_len, 4),
    }
