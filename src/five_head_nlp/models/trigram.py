"""Trigram Language Model.

Implements a third-order (N=3) autoregressive language model conditioned on the
two preceding tokens:
    P(w_i | w_{i-2}, w_{i-1}) = Count(w_{i-2}, w_{i-1}, w_i) / Count(w_{i-2}, w_{i-1})
    P(w_1, ..., w_m) = P(w_1) * P(w_2 | w_1) * prod_{i=3}^m P(w_i | w_{i-2}, w_{i-1})
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


class TrigramModel(BaseNGramModel):
    """Trigram Language Model (N=3).

    Conditioned on the two-token history (w_{i-2}, w_{i-1}). Strictly enforces
    sentence-boundary isolation.
    """

    def __init__(self) -> None:
        """Initialize the TrigramModel with order N=3."""
        super().__init__(n=3)

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
            Preceding tokens. If 2 or more tokens are provided, the last two (w_{i-2}, w_{i-1})
            form the trigram context. If 1 token is provided, evaluates bigram probability.
            If None or empty, evaluates unigram probability.

        Returns
        -------
        float
            Conditional probability in range [0.0, 1.0].
            Returns 0.0 if token is unknown, context is unknown, or trigram was not observed.
        """
        self._check_fitted()
        if not self.is_known_token(token):
            return 0.0

        ctx_tuple = self._normalize_context(context)
        if len(ctx_tuple) == 0:
            return self.engine.p_unigram(token)
        elif len(ctx_tuple) == 1:
            return self.engine.p_bigram(token, ctx_tuple[0])
        else:
            w_prev2, w_prev1 = ctx_tuple[-2], ctx_tuple[-1]
            return self.engine.p_trigram(token, w_prev2, w_prev1)

    def get_candidates(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> List[str]:
        """Return candidate continuation tokens following context in deterministic order.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding tokens. Uses the last two tokens if length >= 2.
            If length == 1, returns bigram candidates. If empty, returns unigram candidates.

        Returns
        -------
        List[str]
            Continuation tokens ordered by descending frequency, then alphabetical.
            Returns [] if context is unknown or dead-end.
        """
        self._check_fitted()
        ctx_tuple = self._normalize_context(context)
        if len(ctx_tuple) == 0:
            assert self.vocabulary is not None
            return self.vocabulary.tokens
        elif len(ctx_tuple) == 1:
            return self.engine.get_candidates(ctx_tuple[0], order=2)
        else:
            trigram_ctx = (ctx_tuple[-2], ctx_tuple[-1])
            return self.engine.get_candidates(trigram_ctx, order=3)

    def get_candidate_distribution(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, float]:
        """Return normalized conditional probability distribution over candidates.

        P(w | context) = Count(context, w) / Count(context)

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context.

        Returns
        -------
        Dict[str, float]
            Map of {token: P(token | context)}. Sums to 1.0 if context has continuations,
            or {} if context is unknown or terminal.
        """
        self._check_fitted()
        ctx_tuple = self._normalize_context(context)
        if len(ctx_tuple) == 0:
            assert self.vocabulary is not None
            return {t: self.engine.p_unigram(t) for t in self.vocabulary.tokens}
        elif len(ctx_tuple) == 1:
            return self.engine.get_candidate_distribution(ctx_tuple[0], order=2)
        else:
            trigram_ctx = (ctx_tuple[-2], ctx_tuple[-1])
            return self.engine.get_candidate_distribution(trigram_ctx, order=3)

    def sequence_probability(self, tokens: Sequence[str]) -> float:
        """Calculate joint sequence probability under the trigram Markov assumption:

        P(w_1, ..., w_m) = P(w_1) * P(w_2 | w_1) * prod_{i=3}^m P(w_i | w_{i-2}, w_{i-1})

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

        return self.engine.sequence_probability(tokens, order=3)
