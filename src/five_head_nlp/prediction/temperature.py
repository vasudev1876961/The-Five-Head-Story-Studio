"""Temperature-scaled probability sampling decoder."""

from __future__ import annotations

import math
from typing import List, Optional

from five_head_nlp.prediction.decoding import BaseDecoder
from five_head_nlp.prediction.prediction_result import CandidateToken


class TemperatureDecoder(BaseDecoder):
    """Temperature-scaled probability sampling decoder.

    Mathematical formulation:
        P_T(w_i) = exp( ln(P(w_i)) / T ) / Z
        where Z = sum_{j : P(w_j) > 0} exp( ln(P(w_j)) / T )

    Properties:
        - T > 0.
        - T = 1.0 preserves the original probability distribution.
        - T -> 0+ concentrates probability onto the argmax (approaching greedy selection).
        - Higher T flattens the distribution towards uniform exploration among candidates with P > 0.
        - Zero-probability candidates (P = 0) strictly remain zero probability.
        - Implemented with numerically stable softmax (subtracting max logit to prevent overflow).
    """

    def __init__(self, temperature: float = 1.0, seed: Optional[int] = None) -> None:
        """Initialize TemperatureDecoder.

        Parameters
        ----------
        temperature : float, default=1.0
            Sampling temperature. Must be strictly positive (T > 0).
        seed : Optional[int], default=None
            Seed for deterministic reproducibility.

        Raises
        ------
        ValueError
            If temperature <= 0.0.
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive (T > 0), got {temperature}")

        self.temperature: float = float(temperature)
        super().__init__(seed=seed)

    @property
    def strategy_name(self) -> str:
        return f"temperature(T={self.temperature})"

    def compute_scaled_probabilities(
        self, candidates: List[CandidateToken]
    ) -> List[float]:
        """Compute numerically stable temperature-scaled probabilities.

        Parameters
        ----------
        candidates : List[CandidateToken]
            Ranked candidate tokens.

        Returns
        -------
        List[float]
            Temperature-scaled probabilities matching order of input candidates.
        """
        if not candidates:
            return []

        # Filter strictly positive probabilities
        positive_indices = [i for i, c in enumerate(candidates) if c.probability > 0.0]

        if not positive_indices:
            # All zero: return uniform zeros
            return [0.0] * len(candidates)

        # For very small temperature, return pure one-hot on the argmax (candidates[0])
        if self.temperature < 1e-4:
            res = [0.0] * len(candidates)
            res[0] = 1.0
            return res

        # Numerically stable scaled logits: z_i = ln(P_i) / T
        logits = [math.log(candidates[i].probability) / self.temperature for i in positive_indices]
        max_logit = max(logits)

        # Subtract max logit before exponentiating
        exp_logits = [math.exp(l - max_logit) for l in logits]
        z_sum = sum(exp_logits)

        res = [0.0] * len(candidates)
        for idx, exp_val in zip(positive_indices, exp_logits):
            res[idx] = exp_val / z_sum

        return res

    def select_token(self, candidates: List[CandidateToken]) -> str:
        """Sample a candidate token according to temperature-scaled probabilities.

        Parameters
        ----------
        candidates : List[CandidateToken]
            Ranked candidate tokens.

        Returns
        -------
        str
            Sampled token.

        Raises
        ------
        ValueError
            If candidates list is empty.
        """
        if not candidates:
            raise ValueError("Cannot perform temperature sampling on empty candidates list.")

        # Very low temperature -> greedy fallback
        if self.temperature < 1e-4:
            return candidates[0].token

        scaled_probs = self.compute_scaled_probabilities(candidates)
        tokens = [c.token for c in candidates]

        total_scaled = sum(scaled_probs)
        if total_scaled <= 0.0:
            # Fallback to uniform over candidates
            return self.rng.choice(tokens)

        chosen = self.rng.choices(tokens, weights=scaled_probs, k=1)[0]
        return chosen
