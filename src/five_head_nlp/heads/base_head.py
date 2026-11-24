"""Abstract Base Class for all individual scoring heads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from five_head_nlp.heads.head_result import HeadScore
from five_head_nlp.prediction.generation_state import GenerationState
from five_head_nlp.prediction.prediction_result import CandidateToken


class BaseHead(ABC):
    """Abstract Base Class defining the scoring head contract.

    Principles:
        1. Pure Evaluation: score_candidates() MUST NOT mutate head state.
        2. Explicit State Updates: state mutations occur ONLY when update_state(token)
           is explicitly called after decoding selection.
        3. Explainability: every scored candidate must include an interpretable reason.
    """

    def __init__(self, name: str, weight: float = 1.0, enabled: bool = True) -> None:
        """Initialize BaseHead."""
        if weight < 0.0:
            raise ValueError(f"Head weight must be non-negative, got {weight}")
        self.name: str = name
        self.weight: float = float(weight)
        self.enabled: bool = enabled

    @abstractmethod
    def score_candidates(
        self,
        context: Tuple[str, ...],
        candidates: List[CandidateToken],
        state: Optional[GenerationState] = None,
    ) -> Dict[str, HeadScore]:
        """Score each candidate token based on context and generation state.

        MUST NOT mutate head state during scoring.

        Parameters
        ----------
        context : Tuple[str, ...]
            Recent context tokens.
        candidates : List[CandidateToken]
            List of candidate tokens with their underlying model probabilities.
        state : Optional[GenerationState], default=None
            Active generation state tracker.

        Returns
        -------
        Dict[str, HeadScore]
            Map of {token: HeadScore}.
        """
        pass

    def update_state(self, selected_token: str) -> None:
        """Update internal head tracking state after a token has been officially chosen.

        Parameters
        ----------
        selected_token : str
            The token selected by the decoding strategy.
        """
        pass

    def reset_state(self) -> None:
        """Reset internal state for a new generation sequence."""
        pass
