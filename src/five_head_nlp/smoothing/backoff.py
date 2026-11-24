"""Backoff Probability Smoothing.

Implements absolute discounting backoff across Trigram, Bigram, and Unigram tiers:
    Trigram (w_1, w_2) -> Bigram (w_2) -> Unigram ()

Mathematical Formulation:
--------------------------
For context h with discount d in (0, 1):
    If C(h, w) > 0:
        P(w | h) = (C(h, w) - d) / C(h)
    If C(h, w) == 0:
        P(w | h) = alpha(h) * P_lower(w | lower(h))

where reserved probability mass beta(h) and normalization factor alpha(h) are:
    beta(h) = (d * |V_cand(h)|) / C(h)
    alpha(h) = beta(h) / sum_{v: C(h, v) == 0} P_lower(v | lower(h))
"""

from __future__ import annotations

import math
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from five_head_nlp.models.base_ngram import BaseNGramModel
from five_head_nlp.probability.mle import NGramProbabilityEngine
from five_head_nlp.smoothing.base_smoother import BaseSmoother
from five_head_nlp.tokenization.vocabulary import Vocabulary


class BackoffSmoother(BaseSmoother):
    """Absolute Discounting Backoff Smoother.

    Gracefully cascades through Trigram -> Bigram -> Unigram layers while
    strictly conserving probability mass via normalization factors alpha(h).
    """

    def __init__(
        self,
        discount: float = 0.5,
        model: Optional[BaseNGramModel] = None,
        engine: Optional[NGramProbabilityEngine] = None,
        vocabulary: Optional[Vocabulary] = None,
        order: Optional[int] = None,
    ) -> None:
        """Initialize the Backoff smoother.

        Parameters
        ----------
        discount : float, default=0.5
            Absolute discount parameter d in range (0.0, 1.0).
        model : Optional[BaseNGramModel]
            Trained language model instance.
        engine : Optional[NGramProbabilityEngine]
            Underlying probability engine.
        vocabulary : Optional[Vocabulary]
            Vocabulary manager.
        order : Optional[int]
            N-gram order (defaults to 3 for trigram backoff).

        Raises
        ------
        ValueError
            If discount is not in range (0.0, 1.0).
        """
        if not (0.0 < discount < 1.0):
            raise ValueError(f"Discount parameter d must be in range (0.0, 1.0), got {discount}")

        self.discount: float = float(discount)
        self._cache: Dict[Tuple[str, ...], Dict[str, float]] = {}
        self._bi_alpha_cache: Dict[str, float] = {}
        self._tri_alpha_cache: Dict[Tuple[str, str], float] = {}
        super().__init__(model=model, engine=engine, vocabulary=vocabulary, order=order)

    def clear_cache(self) -> None:
        """Clear context distribution cache."""
        self._cache.clear()
        self._bi_alpha_cache.clear()
        self._tri_alpha_cache.clear()

    def get_fallback_level(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
    ) -> str:
        """Identify which model tier provides the primary evidence for the token.

        Returns
        -------
        str
            One of 'trigram', 'bigram', or 'unigram'.
        """
        if token not in self.vocabulary:
            return "unigram"

        ctx_tuple = self._normalize_context(context)

        # Trigram check
        if len(ctx_tuple) >= 2 and self.order >= 3:
            tri_ctx = (ctx_tuple[-2], ctx_tuple[-1])
            tri_ngram = tri_ctx + (token,)
            if self.engine.get_count(tri_ngram) > 0:
                return "trigram"

        # Bigram check
        if len(ctx_tuple) >= 1 and self.order >= 2:
            bi_ctx = (ctx_tuple[-1],)
            bi_ngram = bi_ctx + (token,)
            if self.engine.get_count(bi_ngram) > 0:
                return "bigram"

        return "unigram"

    def _p_unigram(self, token: str) -> float:
        """Compute base unigram probability with add-one floor ensuring full vocabulary coverage."""
        if token not in self.vocabulary:
            return 0.0
        c_w = self.engine.get_unigram_count(token)
        v_size = self.vocabulary.size
        total = self.engine.total_tokens
        return (c_w + 1) / (total + v_size)

    def _p_bigram_backoff(self, token: str, prev_token: str) -> float:
        """Compute bigram backoff probability with absolute discounting."""
        bi_ctx = (prev_token,)
        c_ctx = self.engine.get_context_count(bi_ctx, order=2)

        # If context was never observed, fall back directly to unigram
        if c_ctx == 0:
            return self._p_unigram(token)

        c_ngram = self.engine.get_count(bi_ctx + (token,))

        if c_ngram > 0:
            return (c_ngram - self.discount) / c_ctx
        else:
            if prev_token not in self._bi_alpha_cache:
                cand_tokens = self.engine.get_candidates(bi_ctx, order=2)
                beta = (self.discount * len(cand_tokens)) / c_ctx
                unseen_tokens = [t for t in self.vocabulary.tokens if t not in cand_tokens]
                unseen_prob_sum = sum(self._p_unigram(t) for t in unseen_tokens)

                if unseen_prob_sum > 0:
                    self._bi_alpha_cache[prev_token] = beta / unseen_prob_sum
                else:
                    self._bi_alpha_cache[prev_token] = -1.0

            alpha = self._bi_alpha_cache[prev_token]
            if alpha >= 0:
                return alpha * self._p_unigram(token)
            else:
                cand_tokens = self.engine.get_candidates(bi_ctx, order=2)
                n_unseen = len(self.vocabulary.tokens) - len(cand_tokens)
                beta = (self.discount * len(cand_tokens)) / c_ctx
                if n_unseen > 0:
                    return beta / n_unseen
                return 0.0

    def _p_trigram_backoff(self, token: str, prev2: str, prev1: str) -> float:
        """Compute trigram backoff probability with absolute discounting."""
        tri_ctx = (prev2, prev1)
        c_ctx = self.engine.get_context_count(tri_ctx, order=3)

        # If trigram context was never observed, back off directly to bigram
        if c_ctx == 0:
            return self._p_bigram_backoff(token, prev1)

        c_ngram = self.engine.get_count(tri_ctx + (token,))

        if c_ngram > 0:
            return (c_ngram - self.discount) / c_ctx
        else:
            if tri_ctx not in self._tri_alpha_cache:
                cand_tokens = self.engine.get_candidates(tri_ctx, order=3)
                beta = (self.discount * len(cand_tokens)) / c_ctx
                unseen_tokens = [t for t in self.vocabulary.tokens if t not in cand_tokens]
                unseen_prob_sum = sum(self._p_bigram_backoff(t, prev1) for t in unseen_tokens)

                if unseen_prob_sum > 0:
                    self._tri_alpha_cache[tri_ctx] = beta / unseen_prob_sum
                else:
                    self._tri_alpha_cache[tri_ctx] = -1.0

            alpha = self._tri_alpha_cache[tri_ctx]
            if alpha >= 0:
                return alpha * self._p_bigram_backoff(token, prev1)
            else:
                cand_tokens = self.engine.get_candidates(tri_ctx, order=3)
                n_unseen = len(self.vocabulary.tokens) - len(cand_tokens)
                beta = (self.discount * len(cand_tokens)) / c_ctx
                if n_unseen > 0:
                    return beta / n_unseen
                return 0.0

    def probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
    ) -> float:
        """Calculate backoff smoothed probability P_backoff(token | context).

        Parameters
        ----------
        token : str
            Candidate continuation token.
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context tokens.

        Returns
        -------
        float
            Smoothed backoff probability in range [0.0, 1.0].
        """
        if token not in self.vocabulary:
            return 0.0

        ctx_tuple = self._normalize_context(context)

        if len(ctx_tuple) == 0 or self.order == 1:
            return self._p_unigram(token)
        elif len(ctx_tuple) == 1 or self.order == 2:
            return self._p_bigram_backoff(token, ctx_tuple[-1])
        else:
            return self._p_trigram_backoff(token, ctx_tuple[-2], ctx_tuple[-1])

    def get_distribution(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, float]:
        """Return backoff probability distribution over all vocabulary tokens.

        Guaranteed: sum_{v in V} P_backoff(v | context) == 1.0.
        """
        ctx_tuple = self._normalize_context(context)
        if ctx_tuple in self._cache:
            return self._cache[ctx_tuple]
        dist = {token: self.probability(token, context=ctx_tuple) for token in self.vocabulary.tokens}
        self._cache[ctx_tuple] = dist
        return dist

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of backoff hyperparameters."""
        params = super().get_parameters()
        params["discount"] = self.discount
        return params
