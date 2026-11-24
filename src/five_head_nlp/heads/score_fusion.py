"""Weighted score fusion module combining multi-head evaluations."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from five_head_nlp.heads.head_result import FusedCandidate, HeadScore
from five_head_nlp.prediction.prediction_result import CandidateToken


class ScoreFusion:
    """Combines individual scoring head outputs into a unified candidate score.

    Mathematical formulation:
        Score(w) = sum_{k in Heads} alpha_k * Score_k(w)
        subject to:
            alpha_k >= 0, sum_{k} alpha_k == 1.0

    Tie-breaking:
        If two candidates have identical fused scores (within 1e-9 tolerance),
        they are ordered alphabetically by token string.
    """

    def __init__(self, tolerance: float = 1e-9) -> None:
        """Initialize ScoreFusion."""
        self.tolerance: float = tolerance

    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize head weights to sum to 1.0.

        Parameters
        ----------
        weights : Dict[str, float]
            Raw head weights.

        Returns
        -------
        Dict[str, float]
            Normalized weights summing to 1.0.

        Raises
        ------
        ValueError
            If any weight is negative or total weight is non-positive.
        """
        for name, w in weights.items():
            if w < 0.0 or math.isnan(w):
                raise ValueError(f"Weight for head {name!r} must be non-negative finite float, got {w}")

        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError(f"Total weight sum must be strictly positive, got {total}")

        return {k: v / total for k, v in weights.items()}

    def fuse_scores(
        self,
        candidate_tokens: List[CandidateToken],
        head_scores: Dict[str, Dict[str, HeadScore]],
        weights: Dict[str, float],
    ) -> List[FusedCandidate]:
        """Fuse individual head scores for all candidate tokens and rank them.

        Parameters
        ----------
        candidate_tokens : List[CandidateToken]
            Original candidate tokens from language model / smoother.
        head_scores : Dict[str, Dict[str, HeadScore]]
            Map of {head_name: {token: HeadScore}}.
        weights : Dict[str, float]
            Configured weights for each head.

        Returns
        -------
        List[FusedCandidate]
            Ranked fused candidate tokens in descending score order.

        Raises
        ------
        ValueError
            If candidate_tokens is empty or weights are invalid.
        """
        if not candidate_tokens:
            return []

        norm_weights = self.normalize_weights(weights)

        fused_list: List[Tuple[float, str, FusedCandidate]] = []

        for c in candidate_tokens:
            tok = c.token
            fused_score = 0.0
            candidate_head_breakdown: Dict[str, HeadScore] = {}

            for head_name, alpha in norm_weights.items():
                if head_name in head_scores and tok in head_scores[head_name]:
                    hs = head_scores[head_name][tok]
                else:
                    # Default fallback if head did not score this token
                    hs = HeadScore(
                        head_name=head_name,
                        token=tok,
                        raw_score=0.0,
                        normalized_score=0.0,
                        explanation=f"Token '{tok}' not explicitly evaluated by {head_name}.",
                    )

                candidate_head_breakdown[head_name] = hs
                fused_score += alpha * hs.normalized_score

            # Preserve Candidate object
            cand_obj = FusedCandidate(
                token=tok,
                fused_score=fused_score,
                rank=0,  # will be assigned after sorting
                head_scores=candidate_head_breakdown,
                original_probability=c.probability,
            )
            fused_list.append((fused_score, tok, cand_obj))

        # Sort: descending by fused_score, ascending by token (alphabetical tie-breaking)
        def sort_key(item: Tuple[float, str, FusedCandidate]) -> Tuple[float, str]:
            score, tok, _ = item
            return (-score, tok)

        sorted_fused = sorted(fused_list, key=sort_key)

        ranked_results: List[FusedCandidate] = []
        for rank_idx, (_, _, cand) in enumerate(sorted_fused, start=1):
            cand.rank = rank_idx
            ranked_results.append(cand)

        return ranked_results
