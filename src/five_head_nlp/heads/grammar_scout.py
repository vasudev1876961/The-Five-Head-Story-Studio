"""Head 1 — Grammar Scout: Evaluates grammatical and statistical plausibility."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from five_head_nlp.heads.base_head import BaseHead
from five_head_nlp.heads.head_result import HeadScore
from five_head_nlp.prediction.generation_state import GenerationState
from five_head_nlp.prediction.prediction_result import CandidateToken


class GrammarScout(BaseHead):
    """Head 1 — Grammar Scout.

    Evaluates candidate tokens based on empirical N-gram statistical probabilities
    produced by the underlying probability and smoothing engines.

    Mathematical formulation:
        G(w | context) = P(w | context)
        Normalized in range [0.0, 1.0].
    """

    def __init__(self, weight: float = 0.40, enabled: bool = True) -> None:
        """Initialize Grammar Scout head."""
        super().__init__(name="Grammar Scout", weight=weight, enabled=enabled)

    def score_candidates(
        self,
        context: Tuple[str, ...],
        candidates: List[CandidateToken],
        state: Optional[GenerationState] = None,
    ) -> Dict[str, HeadScore]:
        """Score each candidate token using its conditional language model probability."""
        scores: Dict[str, HeadScore] = {}
        ctx_str = " ".join(context) if context else "<empty>"

        for c in candidates:
            prob = c.probability
            # Provide clear qualitative explanations based on probability magnitude
            if prob >= 0.50:
                explanation = f"Statistically dominant continuation ({prob:.4f}) given context [{ctx_str}]."
            elif prob >= 0.10:
                explanation = f"High-probability continuation ({prob:.4f}) supported by N-gram transitions."
            elif prob >= 0.01:
                explanation = f"Moderate continuation probability ({prob:.4f}) under N-gram model."
            else:
                explanation = f"Low probability ({prob:.4f}), infrequent transition or backoff floor."

            scores[c.token] = HeadScore(
                head_name=self.name,
                token=c.token,
                raw_score=prob,
                normalized_score=prob,
                explanation=explanation,
                evidence={
                    "rank": c.rank,
                    "conditional_probability": round(prob, 6),
                    "log_probability": round(c.log_probability, 4),
                    "context": list(context),
                },
            )

        return scores
