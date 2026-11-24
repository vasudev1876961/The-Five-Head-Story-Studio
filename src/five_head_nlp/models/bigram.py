"""Bigram Language Model.

Implements a second-order (N=2) autoregressive language model conditioned on the
immediate preceding token:
    P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})
    P(w_1, ..., w_m) = P(w_1) * prod_{i=2}^m P(w_i | w_{i-1})
"""

from __future__ import annotations

from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

from five_head_nlp.models.base_ngram import BaseNGramModel


class BigramModel(BaseNGramModel):
    """Bigram Language Model (N=2).

    Conditioned on the immediate prior token w_{i-1}. Strictly enforces
    sentence-boundary isolation (the period '.' has zero outgoing transitions).
    """

    def __init__(self) -> None:
        """Initialize the BigramModel with order N=2."""
        super().__init__(n=2)

    def probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
    ) -> float:
        """Calculate conditional probability P(token | context).

        Parameters
        ----------
        token : str
            Candidate next token.
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding token w_{i-1}. If None or empty, returns unigram probability P(token).

        Returns
        -------
        float
            Conditional probability in range [0.0, 1.0].
            Returns 0.0 if token is unknown, context is unknown, or bigram was not observed.
        """
        self._check_fitted()
        if not self.is_known_token(token):
            return 0.0

        ctx_tuple = self._normalize_context(context)
        if len(ctx_tuple) == 0:
            return self.engine.p_unigram(token)

        # Use the most recent token if a longer history sequence was supplied
        prev_token = ctx_tuple[-1]
        return self.engine.p_bigram(token, prev_token)

    def get_candidates(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> List[str]:
        """Return candidate continuation tokens following context.

        Ordered deterministically by descending frequency, then alphabetical.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding token w_{i-1}. If None or empty, returns all vocabulary tokens.

        Returns
        -------
        List[str]
            Valid continuation tokens. Returns [] if context is unknown or terminal.
        """
        self._check_fitted()
        ctx_tuple = self._normalize_context(context)
        if len(ctx_tuple) == 0:
            assert self.vocabulary is not None
            return self.vocabulary.tokens

        prev_token = ctx_tuple[-1]
        return self.engine.get_candidates(prev_token, order=2)

    def get_candidate_distribution(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, float]:
        """Return normalized conditional probability distribution over candidates.

        P(w | context) = Count(context, w) / Count(context)

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding token w_{i-1}. If None or empty, returns unigram distribution.

        Returns
        -------
        Dict[str, float]
            Map of {token: P(token | context)}. Sums to 1.0 if context has continuations,
            or {} if context is unknown or dead-end (e.g. period '.').
        """
        self._check_fitted()
        ctx_tuple = self._normalize_context(context)
        if len(ctx_tuple) == 0:
            assert self.vocabulary is not None
            return {t: self.engine.p_unigram(t) for t in self.vocabulary.tokens}

        prev_token = ctx_tuple[-1]
        return self.engine.get_candidate_distribution(prev_token, order=2)

    def sequence_probability(self, tokens: Sequence[str]) -> float:
        """Calculate joint sequence probability under the bigram Markov assumption:

        P(w_1, ..., w_m) = P(w_1) * prod_{i=2}^m P(w_i | w_{i-1})

        Parameters
        ----------
        tokens : Sequence[str]
            Sequence of tokens.

        Returns
        -------
        float
            Joint probability in range [0.0, 1.0]. Returns 1.0 for empty sequence.
        """
        self._check_fitted()
        if not tokens:
            return 1.0

        return self.engine.sequence_probability(tokens, order=2)
