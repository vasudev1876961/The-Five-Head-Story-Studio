"""Unigram Language Model.

Implements a first-order (N=1) autoregressive language model under the
independent token assumption:
    P(w) = Count(w) / N_total
    P(w_1, ..., w_m) = prod_{i=1}^m P(w_i)
"""

from __future__ import annotations

import math
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

from five_head_nlp.models.base_ngram import BaseNGramModel


class UnigramModel(BaseNGramModel):
    """Unigram Language Model (N=1).

    Estimates individual word frequencies and empirical probabilities
    independent of context history.
    """

    def __init__(self) -> None:
        """Initialize the UnigramModel with order N=1."""
        super().__init__(n=1)

    def probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
    ) -> float:
        """Calculate unigram probability P(token) = Count(token) / N_total.

        In a unigram model, probability is strictly context-independent; any
        provided context is ignored.

        Parameters
        ----------
        token : str
            Candidate token string.
        context : Optional[Union[str, Sequence[str]]], default=None
            Ignored in UnigramModel (context-independent).

        Returns
        -------
        float
            Empirical unigram probability in range [0.0, 1.0].
            Returns 0.0 if token is unknown.
        """
        self._check_fitted()
        if not self.is_known_token(token):
            return 0.0
        return self.engine.p_unigram(token)

    def get_candidates(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> List[str]:
        """Return all vocabulary tokens in deterministic order.

        Ordered by descending unigram frequency, breaking ties alphabetically.
        Context is ignored in a unigram model.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]], default=None
            Ignored in UnigramModel.

        Returns
        -------
        List[str]
            Full vocabulary tokens in deterministic order.
        """
        self._check_fitted()
        assert self.vocabulary is not None
        return self.vocabulary.tokens

    def get_candidate_distribution(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, float]:
        """Return the unigram probability distribution across all vocabulary tokens.

        Sums to exactly 1.0 across all tokens in the vocabulary.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]], default=None
            Ignored in UnigramModel.

        Returns
        -------
        Dict[str, float]
            Map of {token: P(token)}.
        """
        self._check_fitted()
        assert self.vocabulary is not None
        return {token: self.probability(token) for token in self.vocabulary.tokens}

    def sequence_probability(self, tokens: Sequence[str]) -> float:
        """Calculate joint sequence probability under the unigram assumption:

        P(w_1, ..., w_m) = prod_{i=1}^m P(w_i)

        Parameters
        ----------
        tokens : Sequence[str]
            Sequence of tokens.

        Returns
        -------
        float
            Joint probability. Returns 1.0 for empty sequence, 0.0 if any token
            is unseen.
        """
        self._check_fitted()
        if not tokens:
            return 1.0

        prob = 1.0
        for token in tokens:
            p_w = self.probability(token)
            if p_w == 0.0:
                return 0.0
            prob *= p_w
        return prob
