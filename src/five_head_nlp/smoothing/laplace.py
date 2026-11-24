"""Laplace (Add-One) Probability Smoothing.

Implements Add-One smoothing across Unigram, Bigram, and Trigram language models:
    Unigram:     P_Laplace(w) = (Count(w) + 1) / (N_total + |V|)
    Conditional: P_Laplace(w | h) = (Count(h, w) + 1) / (Count(h) + |V|)
"""

from __future__ import annotations

from typing import (
    Dict,
    Optional,
    Sequence,
    Union,
)

from five_head_nlp.models.base_ngram import BaseNGramModel
from five_head_nlp.probability.mle import NGramProbabilityEngine
from five_head_nlp.smoothing.base_smoother import BaseSmoother
from five_head_nlp.tokenization.vocabulary import Vocabulary


class LaplaceSmoother(BaseSmoother):
    """Laplace (Add-One) Smoother.

    Adds 1 to every N-gram transition count and |V| to the context normalizer,
    guaranteeing that every token in the vocabulary receives non-zero probability.
    """

    def __init__(
        self,
        model: Optional[BaseNGramModel] = None,
        engine: Optional[NGramProbabilityEngine] = None,
        vocabulary: Optional[Vocabulary] = None,
        order: Optional[int] = None,
    ) -> None:
        """Initialize Laplace smoother."""
        super().__init__(model=model, engine=engine, vocabulary=vocabulary, order=order)

    def probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
    ) -> float:
        """Calculate Laplace-smoothed probability P_Laplace(token | context).

        Parameters
        ----------
        token : str
            Candidate continuation token.
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context tokens. If None or empty (or if order == 1),
            computes unigram Laplace probability.

        Returns
        -------
        float
            Probability in range (0.0, 1.0] for known vocabulary tokens; 0.0 for unknown.
        """
        if token not in self.vocabulary:
            return 0.0

        v_size = self.vocabulary.size
        ctx_tuple = self._normalize_context(context)

        # Unigram calculation
        if len(ctx_tuple) == 0 or self.order == 1:
            count_w = self.engine.get_unigram_count(token)
            n_total = self.engine.total_tokens
            return (count_w + 1) / (n_total + v_size)

        # Higher-order conditional calculation
        # Trim context to match self.order - 1 if a longer history was supplied
        effective_ctx = ctx_tuple[-(self.order - 1):]
        k = len(effective_ctx) + 1

        ngram = effective_ctx + (token,)
        c_ngram = self.engine.get_count(ngram)
        c_ctx = self.engine.get_context_count(effective_ctx, order=k)

        return (c_ngram + 1) / (c_ctx + v_size)

    def get_distribution(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, float]:
        """Return Laplace-smoothed probability distribution over all vocabulary tokens.

        Guaranteed: sum_{v in V} P_Laplace(v | context) == 1.0.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context tokens.

        Returns
        -------
        Dict[str, float]
            Map of {token: smoothed_prob} for all tokens in the vocabulary.
        """
        return {token: self.probability(token, context=context) for token in self.vocabulary.tokens}
