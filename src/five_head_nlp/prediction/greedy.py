"""Greedy decoding implementation."""

from __future__ import annotations

from typing import List, Optional

from five_head_nlp.prediction.decoding import BaseDecoder
from five_head_nlp.prediction.prediction_result import CandidateToken


class GreedyDecoder(BaseDecoder):
    """Greedy next-token selection: selects token with highest probability.

    Mathematical formulation:
        w_t = argmax_{w in V} P(w | context)

    Properties:
        - Strictly deterministic: identical inputs yield identical outputs.
        - Inherits alphabetical tie-breaking from the candidate ranking pipeline.
        - Zero randomness.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize GreedyDecoder (seed parameter preserved for interface consistency)."""
        super().__init__(seed=seed)

    @property
    def strategy_name(self) -> str:
        return "greedy"

    def select_token(self, candidates: List[CandidateToken]) -> str:
        """Select the highest-probability candidate.

        Parameters
        ----------
        candidates : List[CandidateToken]
            Ranked candidate tokens.

        Returns
        -------
        str
            Token with maximum probability (rank 1).

        Raises
        ------
        ValueError
            If candidates list is empty.
        """
        if not candidates:
            raise ValueError("Cannot perform greedy decoding on empty candidates list.")

        # Candidates are pre-sorted in descending probability with deterministic tie-breaking
        return candidates[0].token
