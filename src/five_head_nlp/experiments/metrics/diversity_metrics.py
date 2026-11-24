"""Vocabulary diversity and Type-Token Ratio (TTR) metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Sequence


def calculate_diversity_metrics(
    tokens: Sequence[str],
    window_size: int = 20,
) -> Dict[str, Any]:
    """Calculate lexical diversity, Type-Token Ratio (TTR), moving-window TTR, and hapax ratio.

    Parameters
    ----------
    tokens : Sequence[str]
        Full token sequence.
    window_size : int, default=20
        Window size for Moving-Average Type-Token Ratio (MATTR).

    Returns
    -------
    Dict[str, Any]
        Dictionary of diversity metrics.
    """
    # Exclude punctuation for lexical diversity calculation
    words = [t.lower() for t in tokens if t not in {".", ",", "!", "?"}]
    total_words = len(words)

    if total_words == 0:
        return {
            "vocabulary_size": 0,
            "ttr": 0.0,
            "mattr_20": 0.0,
            "hapax_count": 0,
            "hapax_ratio": 0.0,
        }

    word_counts = Counter(words)
    vocab_size = len(word_counts)
    ttr = vocab_size / total_words

    # Hapax legomena: words occurring exactly once
    hapax_count = sum(1 for cnt in word_counts.values() if cnt == 1)
    hapax_ratio = hapax_count / vocab_size if vocab_size > 0 else 0.0

    # Moving-Average TTR (MATTR)
    if total_words < window_size:
        mattr = ttr
    else:
        num_windows = total_words - window_size + 1
        window_ttrs = []
        for i in range(num_windows):
            win_words = words[i : i + window_size]
            win_ttr = len(set(win_words)) / window_size
            window_ttrs.append(win_ttr)
        mattr = sum(window_ttrs) / len(window_ttrs)

    return {
        "vocabulary_size": vocab_size,
        "ttr": round(ttr, 4),
        "mattr_20": round(mattr, 4),
        "hapax_count": hapax_count,
        "hapax_ratio": round(hapax_ratio, 4),
    }
