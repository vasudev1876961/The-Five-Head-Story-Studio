"""Story metrics aggregation module."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from five_head_nlp.experiments.metrics.action_metrics import calculate_action_metrics
from five_head_nlp.experiments.metrics.character_metrics import calculate_character_metrics
from five_head_nlp.experiments.metrics.diversity_metrics import calculate_diversity_metrics
from five_head_nlp.experiments.metrics.latency_metrics import calculate_latency_metrics
from five_head_nlp.experiments.metrics.length_metrics import calculate_length_metrics
from five_head_nlp.experiments.metrics.readability_metrics import calculate_readability_metrics
from five_head_nlp.experiments.metrics.repetition_metrics import calculate_repetition_metrics
from five_head_nlp.experiments.metrics.setting_metrics import calculate_setting_metrics


@dataclass
class StoryMetrics:
    """Consolidated metrics for a single generated story."""

    length: Dict[str, Any]
    repetition: Dict[str, Any]
    diversity: Dict[str, Any]
    character: Dict[str, Any]
    setting: Dict[str, Any]
    action: Dict[str, Any]
    readability: Dict[str, Any]
    latency: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to hierarchical dictionary."""
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert metrics to flat dictionary suitable for CSV serialization."""
        flat: Dict[str, Any] = {}
        for category, cat_dict in asdict(self).items():
            for key, val in cat_dict.items():
                if isinstance(val, (list, tuple, set)):
                    continue  # Omit complex list fields from tabular flat view
                flat[f"{category}_{key}"] = val
        return flat


def compute_story_metrics(
    tokens: Sequence[str],
    generation_time_sec: float = 0.0,
    token_latencies_ms: Optional[Sequence[float]] = None,
) -> StoryMetrics:
    """Compute all metric categories for a given story.

    Parameters
    ----------
    tokens : Sequence[str]
        Full token sequence.
    generation_time_sec : float, default=0.0
        Total generation time in seconds.
    token_latencies_ms : Optional[Sequence[float]], default=None
        Per-token prediction latencies in milliseconds.

    Returns
    -------
    StoryMetrics
        Consolidated metrics object.
    """
    token_lats = list(token_latencies_ms) if token_latencies_ms is not None else []

    return StoryMetrics(
        length=calculate_length_metrics(tokens),
        repetition=calculate_repetition_metrics(tokens),
        diversity=calculate_diversity_metrics(tokens),
        character=calculate_character_metrics(tokens),
        setting=calculate_setting_metrics(tokens),
        action=calculate_action_metrics(tokens),
        readability=calculate_readability_metrics(tokens),
        latency=calculate_latency_metrics(generation_time_sec, token_lats),
    )


__all__ = [
    "StoryMetrics",
    "compute_story_metrics",
    "calculate_length_metrics",
    "calculate_repetition_metrics",
    "calculate_diversity_metrics",
    "calculate_character_metrics",
    "calculate_setting_metrics",
    "calculate_action_metrics",
    "calculate_readability_metrics",
    "calculate_latency_metrics",
]
