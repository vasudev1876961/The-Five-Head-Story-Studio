"""Good-Turing Probability Estimation.

Implements Good-Turing discounting and unseen mass estimation using
frequency-of-frequency statistics:
    N_r = Number of N-gram types occurring exactly r times
    r* = (r + 1) * N_{r+1} / N_r
    P_0 = N_1 / N  (total probability mass allocated to unseen events)
"""

from __future__ import annotations

import math
from collections import Counter
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


class GoodTuringSmoother(BaseSmoother):
    """Good-Turing Frequency Estimator and Smoother.

    Re-estimates N-gram counts using the frequency-of-frequencies distribution:
        N_r = Number of distinct N-gram event types occurring exactly r times.
        Total singleton occurrences = 1 * N_1 = N_1.
        Total observed token events N = sum_{r >= 1} r * N_r.
        Adjusted count r* = (r + 1) * N_{r+1} / N_r.
        Unseen probability mass P_0 = N_1 / N.

    Small-Corpus Considerations and Fallback Approximations:
    -------------------------------------------------------
    In small datasets (e.g. 34 sentences), frequency-of-frequency bins become
    sparse, leading to missing bins where N_{r+1} == 0. When N_{r+1} == 0 or
    r exceeds k_threshold (default 5), counts are un-discounted (r* = r) to
    prevent assigning zero probability to frequently observed events.
    Because these fallback approximations are used, this is a Simple Good-Turing
    implementation rather than an asymptotic linear-regression estimator.
    """

    def __init__(
        self,
        k_threshold: int = 5,
        model: Optional[BaseNGramModel] = None,
        engine: Optional[NGramProbabilityEngine] = None,
        vocabulary: Optional[Vocabulary] = None,
        order: Optional[int] = None,
    ) -> None:
        """Initialize the Good-Turing smoother.

        Parameters
        ----------
        k_threshold : int, default=5
            Threshold above which raw counts r are treated as reliable and
            un-discounted (r* = r).
        model : Optional[BaseNGramModel]
            Trained language model instance.
        engine : Optional[NGramProbabilityEngine]
            Underlying probability engine.
        vocabulary : Optional[Vocabulary]
            Vocabulary manager.
        order : Optional[int]
            N-gram order.
        """
        super().__init__(model=model, engine=engine, vocabulary=vocabulary, order=order)
        self.k_threshold: int = k_threshold

        # Compute frequency-of-frequencies and adjusted count tables per order
        self._fof_tables: Dict[int, Dict[int, int]] = {}
        self._adjusted_r_tables: Dict[int, Dict[int, float]] = {}
        self._unseen_mass_tables: Dict[int, float] = {}

        self._compute_good_turing_tables()

    def _compute_good_turing_tables(self) -> None:
        """Calculate frequency-of-frequencies N_r and adjusted counts r* for each order."""
        for k in range(1, self.order + 1):
            if k == 1:
                # Unigrams
                counts = [
                    self.engine.get_unigram_count(t)
                    for t in self.vocabulary.tokens
                ]
            else:
                ngram_counts = self.engine._ngram_counts.get(k, {})
                counts = list(ngram_counts.values())

            # Frequency-of-frequency distribution N_r
            fof = Counter(counts)
            # Remove zero counts if any
            fof.pop(0, None)
            self._fof_tables[k] = dict(sorted(fof.items()))

            total_n = sum(r * count for r, count in fof.items())
            n_1 = fof.get(1, 0)
            self._unseen_mass_tables[k] = (n_1 / total_n) if total_n > 0 else 0.0

            # Adjusted counts r*
            adj_r: Dict[int, float] = {}
            for r in sorted(fof.keys()):
                n_r = fof[r]
                n_r1 = fof.get(r + 1, 0)

                if r <= self.k_threshold and n_r1 > 0 and n_r > 0:
                    adj_r[r] = (r + 1) * n_r1 / n_r
                else:
                    # Fallback when N_{r+1} is missing or count is above threshold
                    adj_r[r] = float(r)

            self._adjusted_r_tables[k] = adj_r

    def get_frequency_of_frequencies(self, order: Optional[int] = None) -> Dict[int, int]:
        """Return the frequency-of-frequencies table N_r for the given order."""
        k = order if order is not None else self.order
        return dict(self._fof_tables.get(k, {}))

    def get_adjusted_count(self, r: int, order: Optional[int] = None) -> float:
        """Return the Good-Turing adjusted count r* for frequency r."""
        if r <= 0:
            return 0.0
        k = order if order is not None else self.order
        table = self._adjusted_r_tables.get(k, {})
        return table.get(r, float(r))

    def get_unseen_mass(self, order: Optional[int] = None) -> float:
        """Return total probability mass P_0 = N_1 / N allocated to unobserved events."""
        k = order if order is not None else self.order
        return self._unseen_mass_tables.get(k, 0.0)

    def probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
    ) -> float:
        """Calculate Good-Turing smoothed probability P_GT(token | context).

        Parameters
        ----------
        token : str
            Candidate continuation token.
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context tokens.

        Returns
        -------
        float
            Smoothed probability in range [0.0, 1.0].
        """
        if token not in self.vocabulary:
            return 0.0

        ctx_tuple = self._normalize_context(context)

        # Unigram calculation
        if len(ctx_tuple) == 0 or self.order == 1:
            raw_r = self.engine.get_unigram_count(token)
            unseen_unigrams = [t for t in self.vocabulary.tokens if self.engine.get_unigram_count(t) == 0]
            n_0 = len(unseen_unigrams)
            p_0 = self.get_unseen_mass(order=1)

            if raw_r == 0:
                if n_0 > 0 and p_0 > 0:
                    return p_0 / n_0
                return 1.0 / (self.vocabulary.size * (self.engine.total_tokens + 1))
            else:
                adj_r = self.get_adjusted_count(raw_r, order=1)
                seen_tokens = [t for t in self.vocabulary.tokens if self.engine.get_unigram_count(t) > 0]
                total_adj = sum(
                    self.get_adjusted_count(self.engine.get_unigram_count(t), order=1)
                    for t in seen_tokens
                )
                if n_0 > 0 and p_0 > 0:
                    return (1.0 - p_0) * (adj_r / total_adj) if total_adj > 0 else 1.0 / self.vocabulary.size
                return adj_r / total_adj if total_adj > 0 else 1.0 / self.vocabulary.size

        # Higher-order conditional calculation
        effective_ctx = ctx_tuple[-(self.order - 1):]
        k_order = len(effective_ctx) + 1

        ngram = effective_ctx + (token,)
        raw_r = self.engine.get_count(ngram)
        c_ctx = self.engine.get_context_count(effective_ctx, order=k_order)

        # If context was never observed, assign uniform distribution
        if c_ctx == 0:
            return 1.0 / self.vocabulary.size

        # Active continuation candidates for this context
        candidates = self.engine.get_candidates(effective_ctx, order=k_order)
        unseen_vocab_count = self.vocabulary.size - len(candidates)

        # Global unseen mass for order k
        p_0 = self.get_unseen_mass(order=k_order)

        if raw_r > 0:
            # Observed continuation: reweight by r*
            adj_r = self.get_adjusted_count(raw_r, order=k_order)
            cand_adj_sum = sum(
                self.get_adjusted_count(
                    self.engine.get_count(effective_ctx + (w,)), order=k_order
                )
                for w in candidates
            )
            if cand_adj_sum == 0:
                return 1.0 / self.vocabulary.size

            if unseen_vocab_count > 0:
                # Distribute (1 - P_0) across observed continuations
                return (1.0 - p_0) * (adj_r / cand_adj_sum)
            else:
                # All vocabulary tokens have been observed after this context
                return adj_r / cand_adj_sum
        else:
            # Unobserved continuation for this context
            if unseen_vocab_count > 0 and p_0 > 0:
                return p_0 / unseen_vocab_count
            return 1.0 / (self.vocabulary.size * (c_ctx + 1))

    def get_distribution(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, float]:
        """Return Good-Turing smoothed distribution across all vocabulary tokens.

        Guaranteed: sum_{v in V} P_GT(v | context) == 1.0.
        """
        return {token: self.probability(token, context=context) for token in self.vocabulary.tokens}

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of Good-Turing hyperparameters and statistics."""
        params = super().get_parameters()
        params["k_threshold"] = self.k_threshold
        params["fof_tables"] = {
            str(k): {str(r): n_r for r, n_r in table.items()}
            for k, table in self._fof_tables.items()
        }
        params["unseen_mass"] = {str(k): mass for k, mass in self._unseen_mass_tables.items()}
        return params
