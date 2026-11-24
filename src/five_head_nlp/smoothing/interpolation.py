"""Linear Interpolation Probability Smoothing.

Implements unigram-bigram-trigram linear interpolation:
    P_interp(w | w1, w2) = lambda_1 * P_uni(w)
                         + lambda_2 * P_bi(w | w2)
                         + lambda_3 * P_tri(w | w1, w2)

subject to:
    lambda_1, lambda_2, lambda_3 >= 0
    lambda_1 + lambda_2 + lambda_3 = 1.0
"""

from __future__ import annotations

import math
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from five_head_nlp.models.base_ngram import BaseNGramModel
from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.models.unigram import UnigramModel
from five_head_nlp.probability.mle import NGramProbabilityEngine
from five_head_nlp.smoothing.base_smoother import BaseSmoother
from five_head_nlp.tokenization.vocabulary import Vocabulary


class InterpolationSmoother(BaseSmoother):
    """Linear Interpolation Smoother.

    Combines Unigram, Bigram, and Trigram probabilities via convex combination.
    Guarantees non-zero probability mass whenever the unigram probability is non-zero,
    even for novel bigram or trigram contexts.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float] = (0.15, 0.35, 0.50),
        model: Optional[BaseNGramModel] = None,
        engine: Optional[NGramProbabilityEngine] = None,
        vocabulary: Optional[Vocabulary] = None,
        order: Optional[int] = 3,
        unigram_model: Optional[UnigramModel] = None,
        bigram_model: Optional[BigramModel] = None,
        trigram_model: Optional[TrigramModel] = None,
    ) -> None:
        """Initialize the Linear Interpolation smoother.

        Parameters
        ----------
        weights : Tuple[float, float, float], default=(0.15, 0.35, 0.50)
            Interpolation weights (lambda_1, lambda_2, lambda_3) for unigram,
            bigram, and trigram tiers respectively.
        model : Optional[BaseNGramModel]
            Trained language model instance.
        engine : Optional[NGramProbabilityEngine]
            Underlying probability engine.
        vocabulary : Optional[Vocabulary]
            Vocabulary manager.
        order : Optional[int], default=3
            N-gram order.
        unigram_model : Optional[UnigramModel]
            Pre-configured unigram model.
        bigram_model : Optional[BigramModel]
            Pre-configured bigram model.
        trigram_model : Optional[TrigramModel]
            Pre-configured trigram model.

        Raises
        ------
        ValueError
            If weights are negative or do not sum to 1.0 (within float tolerance).
        """
        self._validate_weights(weights)
        self.weights: Tuple[float, float, float] = weights

        super().__init__(model=model, engine=engine, vocabulary=vocabulary, order=order if order is not None else 3)

        # Initialize component models
        if unigram_model is not None and bigram_model is not None and trigram_model is not None:
            self.uni_model = unigram_model
            self.bi_model = bigram_model
            self.tri_model = trigram_model
        else:
            # Wire component models to shared engine and vocabulary
            self.uni_model = UnigramModel()
            self.uni_model.engine = self.engine
            self.uni_model.vocabulary = self.vocabulary
            self.uni_model._is_fitted = True

            self.bi_model = BigramModel()
            self.bi_model.engine = self.engine
            self.bi_model.vocabulary = self.vocabulary
            self.bi_model._is_fitted = True

            self.tri_model = TrigramModel()
            self.tri_model.engine = self.engine
            self.tri_model.vocabulary = self.vocabulary
            self.tri_model._is_fitted = True

    @staticmethod
    def _validate_weights(weights: Tuple[float, float, float]) -> None:
        """Validate that weights are non-negative and sum to 1.0."""
        if len(weights) != 3:
            raise ValueError(f"Interpolation weights must contain exactly 3 values, got {len(weights)}")
        for i, w in enumerate(weights):
            if w < 0.0:
                raise ValueError(f"Weight lambda_{i+1} must be non-negative, got {w}")
        total = sum(weights)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Interpolation weights must sum to 1.0, got {total}")

    def set_weights(self, weights: Tuple[float, float, float]) -> None:
        """Update interpolation weights with validation."""
        self._validate_weights(weights)
        self.weights = weights

    def probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
    ) -> float:
        """Calculate linear interpolated probability.

        P_interp(w | w1, w2) = l1 * P_uni(w) + l2 * P_bi(w | w2) + l3 * P_tri(w | w1, w2)

        Parameters
        ----------
        token : str
            Candidate continuation token.
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context tokens.

        Returns
        -------
        float
            Smoothed interpolated probability in range [0.0, 1.0].
        """
        if token not in self.vocabulary:
            return 0.0

        ctx_tuple = self._normalize_context(context)
        l1, l2, l3 = self.weights

        # Base unigram probability with add-one floor to guarantee positive support across vocabulary
        c_w = self.engine.get_unigram_count(token)
        v_size = self.vocabulary.size
        n_total = self.engine.total_tokens
        p_uni = (c_w + 1) / (n_total + v_size)

        if len(ctx_tuple) == 0:
            return p_uni
        elif len(ctx_tuple) == 1:
            # 1-token context: normalize weights between unigram and bigram
            sub_total = l1 + l2
            if sub_total > 0:
                w1_norm = l1 / sub_total
                w2_norm = l2 / sub_total
            else:
                w1_norm, w2_norm = 1.0, 0.0
            p_bi = self.bi_model.probability(token, context=ctx_tuple[0])
            return (w1_norm * p_uni) + (w2_norm * p_bi)
        else:
            w1, w2 = ctx_tuple[-2], ctx_tuple[-1]
            c_tri_ctx = self.engine.get_context_count((w1, w2), order=3)
            c_bi_ctx = self.engine.get_context_count((w2,), order=2)

            if c_tri_ctx > 0 and c_bi_ctx > 0:
                p_bi = self.bi_model.probability(token, context=w2)
                p_tri = self.tri_model.probability(token, context=(w1, w2))
                return (l1 * p_uni) + (l2 * p_bi) + (l3 * p_tri)
            elif c_bi_ctx > 0:
                sub_total = l1 + l2
                w1_norm = l1 / sub_total if sub_total > 0 else 1.0
                w2_norm = l2 / sub_total if sub_total > 0 else 0.0
                p_bi = self.bi_model.probability(token, context=w2)
                return (w1_norm * p_uni) + (w2_norm * p_bi)
            else:
                return p_uni

    def get_distribution(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, float]:
        """Return interpolated probability distribution over all vocabulary tokens.

        Guaranteed: sum_{v in V} P_interp(v | context) == 1.0.
        """
        return {token: self.probability(token, context=context) for token in self.vocabulary.tokens}

    def optimize_weights(
        self,
        validation_sentences: Sequence[Sequence[str]],
        step: float = 0.1,
    ) -> Tuple[float, float, float]:
        """Search the 3-simplex grid to find weights minimizing validation perplexity.

        Parameters
        ----------
        validation_sentences : Sequence[Sequence[str]]
            Held-out validation sentences.
        step : float, default=0.1
            Grid search step size.

        Returns
        -------
        Tuple[float, float, float]
            Best weights (lambda_1, lambda_2, lambda_3).
        """
        if not validation_sentences:
            return self.weights

        best_weights = self.weights
        best_pp = float("inf")

        # Generate simplex grid points
        steps = int(round(1.0 / step))
        for i in range(steps + 1):
            l1 = i * step
            for j in range(steps - i + 1):
                l2 = j * step
                l3 = 1.0 - l1 - l2
                if l3 < -1e-6:
                    continue
                l3 = max(0.0, l3)

                candidate_weights = (round(l1, 4), round(l2, 4), round(l3, 4))
                # Skip candidates where sum is zero or unigram weight is zero (which gives inf on unseen)
                if candidate_weights[0] <= 0.0:
                    continue

                self.set_weights(candidate_weights)
                pp = self.perplexity(validation_sentences)
                if pp < best_pp:
                    best_pp = pp
                    best_weights = candidate_weights

        self.set_weights(best_weights)
        return best_weights

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of interpolation hyperparameters."""
        params = super().get_parameters()
        params["weights"] = list(self.weights)
        return params
