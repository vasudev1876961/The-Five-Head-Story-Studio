"""Candidate ranking module for next-token probability distributions."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from five_head_nlp.prediction.prediction_result import CandidateToken


class CandidateRanker:
    """Ranks candidate continuation tokens by probability with deterministic tie-breaking.

    Tie-breaking policy:
        If two candidates have identical probability (within numerical tolerance 1e-9),
        they are ordered alphabetically in ascending order:
        (prob_a == prob_b) -> sort by token_a < token_b.
    """

    def __init__(self, tolerance: float = 1e-9) -> None:
        """Initialize candidate ranker with floating-point tie tolerance."""
        self.tolerance: float = tolerance

    def rank_candidates(
        self,
        distribution: Dict[str, float],
        top_n: Optional[int] = None,
    ) -> List[CandidateToken]:
        """Rank candidates in descending probability order with alphabetical tie-breaking.

        Parameters
        ----------
        distribution : Dict[str, float]
            Map of {token: probability}.
        top_n : Optional[int], default=None
            If provided, returns at most top_n candidates. Otherwise returns all.

        Returns
        -------
        List[CandidateToken]
            Ranked candidate tokens with 1-based ranks.
        """
        if not distribution:
            return []

        # Sort key: (-probability, token)
        # Using rounding to self.tolerance buckets ensures numerical stability for close floats
        def sort_key(item: Tuple[str, float]) -> Tuple[float, str]:
            token, prob = item
            # Negate probability for descending order, then alphabetical token for tie-breaking
            return (-prob, token)

        sorted_items = sorted(distribution.items(), key=sort_key)

        if top_n is not None and top_n > 0:
            sorted_items = sorted_items[:top_n]

        ranked: List[CandidateToken] = []
        for rank_idx, (token, prob) in enumerate(sorted_items, start=1):
            log_prob = math.log(prob) if prob > 0.0 else -math.inf
            ranked.append(
                CandidateToken(
                    token=token,
                    probability=prob,
                    rank=rank_idx,
                    log_probability=log_prob,
                )
            )

        return ranked

    def normalize_distribution(
        self, distribution: Dict[str, float]
    ) -> Dict[str, float]:
        """Explicitly re-normalize a candidate distribution over its support.

        Parameters
        ----------
        distribution : Dict[str, float]
            Map of {token: probability}.

        Returns
        -------
        Dict[str, float]
            Normalized map where sum of probabilities equals 1.0.

        Raises
        ------
        ValueError
            If distribution sum is non-positive or all values are zero.
        """
        total = sum(distribution.values())
        if total <= 0.0:
            raise ValueError(f"Cannot normalize distribution with non-positive sum: {total}")

        return {k: v / total for k, v in distribution.items()}
