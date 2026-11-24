"""Character tracking, introduction, and continuity metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from five_head_nlp.heads.scoring_config import CharacterConfig


def calculate_character_metrics(
    tokens: Sequence[str],
    config: Optional[CharacterConfig] = None,
) -> Dict[str, Any]:
    """Calculate character introduction, continuity, and mention distribution.

    Parameters
    ----------
    tokens : Sequence[str]
        Story token sequence.
    config : Optional[CharacterConfig], default=None
        Configuration containing character lexicons. Defaults to CharacterConfig().

    Returns
    -------
    Dict[str, Any]
        Dictionary of character metrics.
    """
    cfg = config or CharacterConfig()
    all_chars: Set[str] = cfg.named_characters.union(cfg.character_nouns)

    # Extract ordered character mentions
    char_mentions: List[str] = [t.lower() for t in tokens if t.lower() in all_chars]
    total_tokens = len(tokens)
    total_mentions = len(char_mentions)
    unique_chars = sorted(list(set(char_mentions)))
    unique_char_count = len(unique_chars)

    # Character density (mentions per total tokens)
    char_density = (total_mentions / total_tokens) if total_tokens > 0 else 0.0

    # Continuity: consecutive mentions that refer to the same character
    continuity_events = 0
    continuity_comparisons = max(0, total_mentions - 1)
    for i in range(1, total_mentions):
        if char_mentions[i] == char_mentions[i - 1]:
            continuity_events += 1

    continuity_ratio = (continuity_events / continuity_comparisons) if continuity_comparisons > 0 else 0.0

    # Primary (most frequent) character focus
    if char_mentions:
        from collections import Counter
        counts = Counter(char_mentions)
        primary_char, primary_count = counts.most_common(1)[0]
        focus_ratio = primary_count / total_mentions
    else:
        primary_char = None
        focus_ratio = 0.0

    return {
        "character_mentions": total_mentions,
        "unique_characters": unique_char_count,
        "character_density": round(char_density, 4),
        "character_continuity_events": continuity_events,
        "character_continuity_ratio": round(continuity_ratio, 4),
        "primary_character": primary_char,
        "character_focus_ratio": round(focus_ratio, 4),
        "characters_found": unique_chars,
    }
