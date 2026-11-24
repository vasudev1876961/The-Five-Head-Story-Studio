"""Hamilton integer apportionment method (largest remainder) for d10 point allocation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


class HamiltonApportionment:
    """Hamilton's largest-remainder method for integer budget apportionment.

    Translates continuous normalized head weights into an intuitive 10-point (d10)
    classroom game representation.

    Algorithm:
        1. Quota: q_i = K * (w_i / sum(w))
        2. Base Integer: a_i = floor(q_i)
        3. Fractional Remainder: r_i = q_i - a_i
        4. Surplus: S = K - sum(a_i)
        5. Distribute S points (+1 each) to heads with the largest remainders.
        6. Deterministic tie-breaking by predefined head priority order.
    """

    DEFAULT_PRIORITY: List[str] = [
        "Grammar Scout",
        "Character Keeper",
        "Setting Painter",
        "Action Booster",
        "Repeat Blocker",
    ]

    def __init__(
        self,
        total_points: int = 10,
        priority_order: Optional[List[str]] = None,
    ) -> None:
        """Initialize HamiltonApportionment.

        Parameters
        ----------
        total_points : int, default=10
            Total integer point budget (e.g., 10 for a d10 die).
        priority_order : Optional[List[str]], default=None
            Predefined deterministic tie-breaking priority.
        """
        if total_points <= 0:
            raise ValueError(f"total_points must be positive, got {total_points}")

        self.total_points: int = total_points
        self.priority_order: List[str] = priority_order or list(self.DEFAULT_PRIORITY)

    def apportion(self, weights: Dict[str, float]) -> Dict[str, int]:
        """Apportion total_points across heads based on their weights.

        Parameters
        ----------
        weights : Dict[str, float]
            Map of {head_name: weight}. Non-negative values.

        Returns
        -------
        Dict[str, int]
            Map of {head_name: integer_points} such that sum == total_points.

        Raises
        ------
        ValueError
            If any weight is negative or total weight sum is non-positive.
        """
        for name, w in weights.items():
            if w < 0.0 or math.isnan(w):
                raise ValueError(f"Head weight for {name!r} must be non-negative finite float, got {w}")

        total_weight = sum(weights.values())
        if total_weight <= 0.0:
            raise ValueError(f"Sum of weights must be strictly positive, got {total_weight}")

        # Normalized weights
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        quotas: Dict[str, float] = {}
        allocations: Dict[str, int] = {}
        remainders: Dict[str, float] = {}

        for name, w in norm_weights.items():
            q = self.total_points * w
            quotas[name] = q
            a = math.floor(q)
            allocations[name] = a
            remainders[name] = q - a

        surplus = self.total_points - sum(allocations.values())

        # Sort remainders descending with deterministic tie-breaking
        def tie_key(name: str) -> Tuple[float, int]:
            rem = remainders[name]
            prio = self.priority_order.index(name) if name in self.priority_order else 999
            return (-rem, prio)

        ranked_heads = sorted(remainders.keys(), key=tie_key)

        # Distribute surplus points to largest remainders
        for i in range(surplus):
            head_to_increment = ranked_heads[i % len(ranked_heads)]
            allocations[head_to_increment] += 1

        assert sum(allocations.values()) == self.total_points, "Hamilton allocation sum mismatch."
        return allocations
