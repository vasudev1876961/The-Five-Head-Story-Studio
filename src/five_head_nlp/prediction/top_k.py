"""Top-k candidate filtering and sampling decoding."""

from __future__ import annotations

from typing import List, Optional

from five_head_nlp.prediction.decoding import BaseDecoder
from five_head_nlp.prediction.prediction_result import CandidateToken


class TopKDecoder(BaseDecoder):
    """Top-k candidate filtering and stochastic sampling decoder.

    Algorithm:
        1. Select the top k highest-probability candidates.
        2. Renormalize their probabilities over the truncated subset:
           P_{top-k}(w_i) = P(w_i) / sum_{j=1}^k P(w_j)
        3. Sample one candidate from this normalized distribution using an isolated RNG.

    Note on Top-k Filtering vs. Top-k Probability Estimation:
        - Top-k filtering is a decoding-time truncation that discards the tail of the
          distribution to avoid low-probability sampling artifacts.
        - It does not modify underlying language model parameters or N-gram estimates.
    """

    def __init__(self, k: int = 5, seed: Optional[int] = None) -> None:
        """Initialize TopKDecoder.

        Parameters
        ----------
        k : int, default=5
            Number of top candidates to retain before sampling. Must be >= 1.
        seed : Optional[int], default=None
            Seed for deterministic reproducibility.

        Raises
        ------
        ValueError
            If k < 1.
        """
        if k < 1:
            raise ValueError(f"k must be an integer >= 1, got {k}")

        self.k: int = k
        super().__init__(seed=seed)

    @property
    def strategy_name(self) -> str:
        return f"top_k(k={self.k})"

    def select_token(self, candidates: List[CandidateToken]) -> str:
        """Filter top-k candidates, renormalize, and sample one token.

        Parameters
        ----------
        candidates : List[CandidateToken]
            Ranked candidate tokens.

        Returns
        -------
        str
            Sampled token from top-k subset.

        Raises
        ------
        ValueError
            If candidates list is empty.
        """
        if not candidates:
            raise ValueError("Cannot perform top-k decoding on empty candidates list.")

        # Truncate to top k
        top_subset = candidates[: self.k]
        tokens = [c.token for c in top_subset]
        probs = [c.probability for c in top_subset]
        total_p = sum(probs)

        # If all candidates in top subset have 0 probability, sample uniformly among them
        if total_p <= 0.0:
            return self.rng.choice(tokens)

        # Normalized weights for sampling
        norm_weights = [p / total_p for p in probs]
        chosen = self.rng.choices(tokens, weights=norm_weights, k=1)[0]
        return chosen
