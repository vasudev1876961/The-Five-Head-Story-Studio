"""Standardized result structures for individual and fused scoring heads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class HeadScore:
    """Individual score emitted by a specific head for a single candidate token."""

    head_name: str
    token: str
    raw_score: float
    normalized_score: float
    explanation: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert head score to dictionary."""
        return {
            "head_name": self.head_name,
            "token": self.token,
            "raw_score": round(self.raw_score, 4),
            "normalized_score": round(self.normalized_score, 4),
            "explanation": self.explanation,
            "evidence": self.evidence,
        }


@dataclass
class FusedCandidate:
    """Represents a candidate token with combined multi-head scores."""

    token: str
    fused_score: float
    rank: int
    head_scores: Dict[str, HeadScore]
    original_probability: float

    @property
    def grammar_score(self) -> float:
        return self.head_scores["Grammar Scout"].normalized_score if "Grammar Scout" in self.head_scores else 0.0

    @property
    def character_score(self) -> float:
        return self.head_scores["Character Keeper"].normalized_score if "Character Keeper" in self.head_scores else 0.0

    @property
    def setting_score(self) -> float:
        return self.head_scores["Setting Painter"].normalized_score if "Setting Painter" in self.head_scores else 0.0

    @property
    def action_score(self) -> float:
        return self.head_scores["Action Booster"].normalized_score if "Action Booster" in self.head_scores else 0.0

    @property
    def repeat_score(self) -> float:
        return self.head_scores["Repeat Blocker"].normalized_score if "Repeat Blocker" in self.head_scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert fused candidate to dictionary."""
        return {
            "rank": self.rank,
            "token": self.token,
            "fused_score": round(self.fused_score, 4),
            "original_probability": round(self.original_probability, 4),
            "head_breakdown": {k: v.to_dict() for k, v in self.head_scores.items()},
        }


@dataclass
class OrchestratorResult:
    """Standardized output of the multi-head candidate orchestration pipeline."""

    selected_token: Optional[str]
    context: List[str]
    fused_candidates: List[FusedCandidate]
    head_weights: Dict[str, float]
    d10_allocation: Dict[str, int]
    mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_candidate(self) -> Optional[FusedCandidate]:
        """Return the highest-ranked fused candidate."""
        return self.fused_candidates[0] if self.fused_candidates else None

    def get_candidate(self, token: str) -> Optional[FusedCandidate]:
        """Retrieve candidate by token string."""
        for c in self.fused_candidates:
            if c.token == token:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert orchestrator result to dictionary."""
        return {
            "selected_token": self.selected_token,
            "context": self.context,
            "mode": self.mode,
            "head_weights": {k: round(v, 4) for k, v in self.head_weights.items()},
            "d10_allocation": self.d10_allocation,
            "num_candidates": len(self.fused_candidates),
            "top_candidates": [c.to_dict() for c in self.fused_candidates[:10]],
            "metadata": self.metadata,
        }
