"""Readability and lexical density proxy metrics for generated stories.

Note:
-----
Standard readability formulas like Flesch-Kincaid require multi-syllabic phoneme counting,
which is uncalibrated for closed micro-vocabularies (120 tokens).
We provide formal statistical proxies:
- Average sentence length (ASL in words).
- Average word length (AWL in characters).
- Lexical density (ratio of non-exempt content words to total word tokens).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from five_head_nlp.heads.scoring_config import RepeatConfig


def calculate_readability_metrics(tokens: Sequence[str]) -> Dict[str, Any]:
    """Calculate statistical readability proxies and lexical density.

    Parameters
    ----------
    tokens : Sequence[str]
        Story token sequence.

    Returns
    -------
    Dict[str, Any]
        Dictionary of readability proxy metrics.
    """
    words = [t.lower() for t in tokens if t not in {".", ",", "!", "?"}]
    total_words = len(words)

    # Sentence count based on terminal period '.'
    num_sentences = max(1, tokens.count("."))

    avg_sentence_length_words = total_words / num_sentences if num_sentences > 0 else 0.0

    # Average word length in characters
    total_chars = sum(len(w) for w in words)
    avg_word_length_chars = total_chars / total_words if total_words > 0 else 0.0

    # Content words (lexical words excluding function words/exemptions)
    exempt = RepeatConfig().exempt_tokens
    content_words = [w for w in words if w not in exempt]
    lexical_density = len(content_words) / total_words if total_words > 0 else 0.0

    return {
        "avg_sentence_length_words": round(avg_sentence_length_words, 4),
        "avg_word_length_chars": round(avg_word_length_chars, 4),
        "content_word_count": len(content_words),
        "lexical_density": round(lexical_density, 4),
    }
