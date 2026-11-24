"""Configuration specifications for Phase 5 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from five_head_nlp.heads.scoring_config import ScoringConfig


@dataclass
class ExperimentConfig:
    """Specification of an experimental story generation condition."""

    config_id: str  # e.g., 'E1', 'E2', ... 'E8'
    name: str  # e.g., 'Baseline MLE'
    description: str
    model_type: str = "trigram"
    smoother_type: str = "none"  # 'none', 'laplace', 'backoff'
    smoother_params: Dict[str, Any] = field(default_factory=dict)
    mode: str = "baseline"  # 'baseline' or 'five_head'
    scoring_config: Optional[ScoringConfig] = None
    decoder_type: str = "greedy"  # 'greedy', 'top_k', 'temperature'
    decoder_params: Dict[str, Any] = field(default_factory=dict)
    target_min_tokens: int = 80
    target_max_tokens: int = 120
    min_sentences: int = 3
    max_total_tokens: int = 150
    default_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        return {
            "config_id": self.config_id,
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type,
            "smoother_type": self.smoother_type,
            "smoother_params": self.smoother_params,
            "mode": self.mode,
            "decoder_type": self.decoder_type,
            "decoder_params": self.decoder_params,
            "target_min_tokens": self.target_min_tokens,
            "target_max_tokens": self.target_max_tokens,
            "min_sentences": self.min_sentences,
            "max_total_tokens": self.max_total_tokens,
            "default_seed": self.default_seed,
            "scoring_config": self.scoring_config.to_dict() if self.scoring_config else None,
        }
