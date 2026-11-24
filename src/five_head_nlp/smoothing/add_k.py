"""Add-k Probability Smoothing.

Implements generalized Add-k smoothing for any k > 0 across Unigram, Bigram,
and Trigram language models:
    Unigram:     P_Add-k(w) = (Count(w) + k) / (N_total + k|V|)
    Conditional: P_Add-k(w | h) = (Count(h, w) + k) / (Count(h) + k|V|)
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Union,
)

from five_head_nlp.models.base_ngram import BaseNGramModel
from five_head_nlp.probability.mle import NGramProbabilityEngine
from five_head_nlp.smoothing.base_smoother import BaseSmoother
from five_head_nlp.tokenization.vocabulary import Vocabulary


class AddKSmoother(BaseSmoother):
    """Add-k Smoother.

    Extends Laplace smoothing by parameterizing the pseudo-count k (k > 0).
    Smaller values of k (e.g. k=0.1, 0.5) mitigate the over-allocation of
    probability mass to unseen events that typically plagues add-one smoothing.
    """

    def __init__(
        self,
        k: float = 0.5,
        model: Optional[BaseNGramModel] = None,
        engine: Optional[NGramProbabilityEngine] = None,
        vocabulary: Optional[Vocabulary] = None,
        order: Optional[int] = None,
    ) -> None:
        """Initialize Add-k smoother.

        Parameters
        ----------
        k : float, default=0.5
            Smoothing pseudo-count. Must be strictly positive (k > 0).
        model : Optional[BaseNGramModel]
            Trained language model instance.
        engine : Optional[NGramProbabilityEngine]
            Underlying probability engine.
        vocabulary : Optional[Vocabulary]
            Vocabulary manager.
        order : Optional[int]
            N-gram order.

        Raises
        ------
        ValueError
            If k <= 0.
        """
        if k <= 0:
            raise ValueError(f"Smoothing parameter k must be strictly positive (k > 0), got {k}")

        self.k: float = float(k)
        super().__init__(model=model, engine=engine, vocabulary=vocabulary, order=order)

    def probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
    ) -> float:
        """Calculate Add-k smoothed probability P_Add-k(token | context).

        Parameters
        ----------
        token : str
            Candidate continuation token.
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context tokens.

        Returns
        -------
        float
            Probability in range (0.0, 1.0] for vocabulary tokens; 0.0 for unknown.
        """
        if token not in self.vocabulary:
            return 0.0

        v_size = self.vocabulary.size
        ctx_tuple = self._normalize_context(context)

        # Unigram calculation
        if len(ctx_tuple) == 0 or self.order == 1:
            count_w = self.engine.get_unigram_count(token)
            n_total = self.engine.total_tokens
            return (count_w + self.k) / (n_total + self.k * v_size)

        # Higher-order conditional calculation
        effective_ctx = ctx_tuple[-(self.order - 1):]
        k_order = len(effective_ctx) + 1

        ngram = effective_ctx + (token,)
        c_ngram = self.engine.get_count(ngram)
        c_ctx = self.engine.get_context_count(effective_ctx, order=k_order)

        return (c_ngram + self.k) / (c_ctx + self.k * v_size)

    def get_distribution(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, float]:
        """Return Add-k probability distribution over all vocabulary tokens.

        Guaranteed: sum_{v in V} P_Add-k(v | context) == 1.0.
        """
        return {token: self.probability(token, context=context) for token in self.vocabulary.tokens}

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of smoother hyperparameters."""
        params = super().get_parameters()
        params["k"] = self.k
        return params
