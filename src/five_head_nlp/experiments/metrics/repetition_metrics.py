"""Repetition metrics for generated stories."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from five_head_nlp.heads.scoring_config import RepeatConfig


def calculate_repetition_metrics(
    tokens: Sequence[str],
    exempt_tokens: Optional[Set[str]] = None,
    window_size: int = 8,
) -> Dict[str, Any]:
    """Calculate token repetition, n-gram duplicates, and window repetition rates.

    Parameters
    ----------
    tokens : Sequence[str]
        Story token sequence.
    exempt_tokens : Optional[Set[str]], default=None
        Tokens exempt from repetition penalties (e.g. articles, prepositions, punctuation).
        Defaults to RepeatConfig.exempt_tokens.
    window_size : int, default=8
        Window size for rolling phrase repetition analysis.

    Returns
    -------
    Dict[str, Any]
        Dictionary of repetition metrics.
    """
    if exempt_tokens is None:
        exempt_tokens = RepeatConfig().exempt_tokens

    n = len(tokens)
    if n == 0:
        return {
            "immediate_token_repeat_count": 0,
            "immediate_token_repeat_rate": 0.0,
            "repeated_bigram_count": 0,
            "repeated_bigram_rate": 0.0,
            "repeated_trigram_count": 0,
            "repeated_trigram_rate": 0.0,
            "window_repetition_rate": 0.0,
            "has_repetition_event": False,
        }

    # 1. Immediate token repetition: w_i == w_{i-1} for content tokens
    immediate_repeats = 0
    immediate_comparisons = 0
    for i in range(1, n):
        t_curr = tokens[i].lower()
        t_prev = tokens[i - 1].lower()
        if t_curr not in exempt_tokens:
            immediate_comparisons += 1
            if t_curr == t_prev:
                immediate_repeats += 1

    imm_rate = (immediate_repeats / immediate_comparisons) if immediate_comparisons > 0 else 0.0

    # 2. Repeated bigrams: (w_i, w_{i+1})
    bigrams: List[Tuple[str, str]] = [
        (tokens[i].lower(), tokens[i + 1].lower()) for i in range(n - 1)
    ]
    total_bigrams = len(bigrams)
    unique_bigrams = len(set(bigrams))
    repeated_bigrams = total_bigrams - unique_bigrams
    bigram_rate = (repeated_bigrams / total_bigrams) if total_bigrams > 0 else 0.0

    # 3. Repeated trigrams: (w_i, w_{i+1}, w_{i+2})
    trigrams: List[Tuple[str, str, str]] = [
        (tokens[i].lower(), tokens[i + 1].lower(), tokens[i + 2].lower())
        for i in range(n - 2)
    ]
    total_trigrams = len(trigrams)
    unique_trigrams = len(set(trigrams))
    repeated_trigrams = total_trigrams - unique_trigrams
    trigram_rate = (repeated_trigrams / total_trigrams) if total_trigrams > 0 else 0.0

    # 4. Rolling window repetition rate (window_size tokens)
    # Fraction of windows containing at least one repeated non-exempt token
    repeated_windows = 0
    total_windows = max(0, n - window_size + 1)
    if total_windows > 0:
        for i in range(total_windows):
            win = [tokens[j].lower() for j in range(i, i + window_size)]
            content_win = [t for t in win if t not in exempt_tokens]
            counts = Counter(content_win)
            if any(cnt > 1 for cnt in counts.values()):
                repeated_windows += 1
        window_rate = repeated_windows / total_windows
    else:
        window_rate = 0.0

    has_event = bool(immediate_repeats > 0 or repeated_bigrams > 0 or repeated_trigrams > 0)

    return {
        "immediate_token_repeat_count": immediate_repeats,
        "immediate_token_repeat_rate": round(imm_rate, 4),
        "repeated_bigram_count": repeated_bigrams,
        "repeated_bigram_rate": round(bigram_rate, 4),
        "repeated_trigram_count": repeated_trigrams,
        "repeated_trigram_rate": round(trigram_rate, 4),
        "window_repetition_rate": round(window_rate, 4),
        "has_repetition_event": has_event,
    }
