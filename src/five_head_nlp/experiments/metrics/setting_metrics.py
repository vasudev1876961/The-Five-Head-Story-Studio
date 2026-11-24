"""Setting consistency, presence, and transition metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from five_head_nlp.heads.scoring_config import SettingConfig


def calculate_setting_metrics(
    tokens: Sequence[str],
    config: Optional[SettingConfig] = None,
) -> Dict[str, Any]:
    """Calculate setting introductions, presence, and consistency metrics.

    Parameters
    ----------
    tokens : Sequence[str]
        Story token sequence.
    config : Optional[SettingConfig], default=None
        Configuration containing setting lexicons. Defaults to SettingConfig().

    Returns
    -------
    Dict[str, Any]
        Dictionary of setting metrics.
    """
    cfg = config or SettingConfig()
    all_settings: Set[str] = cfg.core_settings.union(cfg.setting_nouns)

    setting_mentions: List[str] = [t.lower() for t in tokens if t.lower() in all_settings]
    total_tokens = len(tokens)
    total_mentions = len(setting_mentions)
    unique_settings = sorted(list(set(setting_mentions)))
    unique_setting_count = len(unique_settings)

    setting_density = (total_mentions / total_tokens) if total_tokens > 0 else 0.0

    if setting_mentions:
        from collections import Counter
        counts = Counter(setting_mentions)
        dominant_setting, dom_count = counts.most_common(1)[0]
        consistency_ratio = dom_count / total_mentions
    else:
        dominant_setting = None
        consistency_ratio = 0.0

    return {
        "setting_mentions": total_mentions,
        "unique_settings": unique_setting_count,
        "setting_density": round(setting_density, 4),
        "dominant_setting": dominant_setting,
        "setting_consistency_ratio": round(consistency_ratio, 4),
        "settings_found": unique_settings,
    }
