"""Action verb density and dynamic-to-stative ratio metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from five_head_nlp.heads.scoring_config import ActionConfig


def calculate_action_metrics(
    tokens: Sequence[str],
    config: Optional[ActionConfig] = None,
) -> Dict[str, Any]:
    """Calculate action verb density and proportion of dynamic actions.

    Parameters
    ----------
    tokens : Sequence[str]
        Story token sequence.
    config : Optional[ActionConfig], default=None
        Configuration containing action lexicons. Defaults to ActionConfig().

    Returns
    -------
    Dict[str, Any]
        Dictionary of action metrics.
    """
    cfg = config or ActionConfig()
    dyn_verbs: Set[str] = cfg.dynamic_actions
    stative_verbs: Set[str] = cfg.state_verbs

    dyn_found: List[str] = [t.lower() for t in tokens if t.lower() in dyn_verbs]
    stative_found: List[str] = [t.lower() for t in tokens if t.lower() in stative_verbs]

    total_tokens = len(tokens)
    dynamic_count = len(dyn_found)
    stative_count = len(stative_found)
    total_action_count = dynamic_count + stative_count

    action_density = (total_action_count / total_tokens) if total_tokens > 0 else 0.0
    dynamic_ratio = (dynamic_count / total_action_count) if total_action_count > 0 else 0.0

    return {
        "dynamic_action_count": dynamic_count,
        "stative_action_count": stative_count,
        "total_action_count": total_action_count,
        "action_density": round(action_density, 4),
        "dynamic_action_ratio": round(dynamic_ratio, 4),
    }
