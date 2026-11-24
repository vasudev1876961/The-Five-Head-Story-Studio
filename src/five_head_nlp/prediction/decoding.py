"""Common base decoder interface and decoding strategy abstractions."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional

from five_head_nlp.prediction.prediction_result import CandidateToken


class BaseDecoder(ABC):
    """Abstract Base Class for all next-token decoding strategies."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize decoder with an isolated random number generator.

        Parameters
        ----------
        seed : Optional[int], default=None
            Seed for deterministic reproducibility. Uses an isolated instance of
            random.Random, ensuring global random state is completely unaffected.
        """
        self.seed: Optional[int] = seed
        self.rng: random.Random = random.Random(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        """Reset or re-seed the isolated random number generator."""
        self.seed = seed
        self.rng = random.Random(seed)

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return canonical name of the decoding strategy."""
        pass

    @abstractmethod
    def select_token(self, candidates: List[CandidateToken]) -> str:
        """Select a next token from ranked candidate tokens.

        Parameters
        ----------
        candidates : List[CandidateToken]
            Ranked candidate tokens (ordered by descending probability with tie-breaking).

        Returns
        -------
        str
            The chosen token.

        Raises
        ------
        ValueError
            If candidates list is empty.
        """
        pass
