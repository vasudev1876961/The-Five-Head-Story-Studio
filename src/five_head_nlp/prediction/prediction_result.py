"""Standardized data structures for next-token prediction and candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CandidateToken:
    """Represents a single candidate token with its probability and rank."""

    token: str
    probability: float
    rank: int
    log_probability: float

    def __repr__(self) -> str:
        return f"CandidateToken(rank={self.rank}, token={self.token!r}, prob={self.probability:.4f})"


@dataclass
class PredictionResult:
    """Standardized output of a next-token prediction step."""

    selected_token: Optional[str]
    context: Tuple[str, ...]
    candidates: List[CandidateToken]
    strategy: str
    model_order: int
    smoother_name: Optional[str]
    is_terminal: bool
    distribution_sum: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_candidate(self) -> Optional[CandidateToken]:
        """Return the highest-ranked candidate token if available."""
        return self.candidates[0] if self.candidates else None

    @property
    def num_candidates(self) -> int:
        """Return total number of candidates evaluated."""
        return len(self.candidates)

    def get_candidate(self, token: str) -> Optional[CandidateToken]:
        """Find candidate token by string identity."""
        for c in self.candidates:
            if c.token == token:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return {
            "selected_token": self.selected_token,
            "context": list(self.context),
            "strategy": self.strategy,
            "model_order": self.model_order,
            "smoother_name": self.smoother_name,
            "is_terminal": self.is_terminal,
            "distribution_sum": round(self.distribution_sum, 6),
            "num_candidates": len(self.candidates),
            "top_candidates": [
                {
                    "rank": c.rank,
                    "token": c.token,
                    "probability": round(c.probability, 6),
                    "log_probability": round(c.log_probability, 6),
                }
                for c in self.candidates[:10]
            ],
            "metadata": self.metadata,
        }
