"""Base Smoother Architecture for N-Gram Language Models.

Provides an abstract foundation for probability smoothing, ensuring
consistent interfaces, input validation, perplexity computation,
and probability normalization across the entire vocabulary simplex.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
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
from five_head_nlp.probability.mle import NGramProbabilityEngine
from five_head_nlp.tokenization.vocabulary import Vocabulary


class BaseSmoother(ABC):
    """Abstract Base Class for all Probability Smoothers.

    Operates in conjunction with an underlying NGramProbabilityEngine and
    Vocabulary to redistribute probability mass from observed N-grams
    to unobserved events, resolving zero-frequency bottlenecks.
    """

    def __init__(
        self,
        model: Optional[BaseNGramModel] = None,
        engine: Optional[NGramProbabilityEngine] = None,
        vocabulary: Optional[Vocabulary] = None,
        order: Optional[int] = None,
    ) -> None:
        """Initialize the base smoother.

        Can be initialized from an existing BaseNGramModel or from explicit
        engine, vocabulary, and order parameters.

        Parameters
        ----------
        model : Optional[BaseNGramModel]
            Trained language model instance.
        engine : Optional[NGramProbabilityEngine]
            Underlying probability engine (if model not provided).
        vocabulary : Optional[Vocabulary]
            Vocabulary manager (if model not provided).
        order : Optional[int]
            N-gram order (defaults to model.n if model provided, else 2).

        Raises
        ------
        ValueError
            If neither model nor (engine and vocabulary) are supplied, or if the
            model is unfitted.
        """
        if model is not None:
            if not model.is_fitted:
                raise ValueError("Cannot initialize smoother with an unfitted model.")
            self.model: Optional[BaseNGramModel] = model
            self.engine: NGramProbabilityEngine = model.engine
            assert model.vocabulary is not None
            self.vocabulary: Vocabulary = model.vocabulary
            self.order: int = model.n
        elif engine is not None and vocabulary is not None:
            if not engine.is_fitted:
                raise ValueError("Cannot initialize smoother with an unfitted probability engine.")
            self.model = None
            self.engine = engine
            self.vocabulary = vocabulary
            self.order = order if order is not None else engine.n
        else:
            raise ValueError(
                "Smoother requires either a trained BaseNGramModel or both "
                "(NGramProbabilityEngine, Vocabulary)."
            )

    @property
    def name(self) -> str:
        """Return the class name of the smoother."""
        return self.__class__.__name__

    @property
    def vocabulary_size(self) -> int:
        """Return the size of the active vocabulary |V|."""
        return self.vocabulary.size

    @property
    def total_tokens(self) -> int:
        """Return total token occurrences processed in training."""
        return self.engine.total_tokens

    def _normalize_context(
        self, context: Optional[Union[str, Sequence[str]]]
    ) -> Tuple[str, ...]:
        """Convert string or sequence context into a normalized tuple."""
        if context is None:
            return ()
        if isinstance(context, str):
            return (context,)
        return tuple(context)

    @abstractmethod
    def probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
    ) -> float:
        """Calculate smoothed conditional probability P_smooth(token | context).

        Parameters
        ----------
        token : str
            Candidate next token.
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context tokens.

        Returns
        -------
        float
            Smoothed probability in range [0.0, 1.0].
        """
        pass

    @abstractmethod
    def get_distribution(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, float]:
        """Return smoothed probability distribution across the entire vocabulary.

        For any context, sum_{v in V} P_smooth(v | context) == 1.0.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context tokens.

        Returns
        -------
        Dict[str, float]
            Map of {token: smoothed_probability} spanning all tokens in vocabulary.
        """
        pass

    def log_probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
        base: float = 2.0,
    ) -> float:
        """Calculate log smoothed conditional probability log_b P_smooth(token | context).

        Parameters
        ----------
        token : str
            Candidate next token.
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context tokens.
        base : float, default=2.0
            Logarithm base (default base 2 for information-theoretic bits).

        Returns
        -------
        float
            Log probability, or -math.inf if P == 0.0.
        """
        prob = self.probability(token, context=context)
        if prob <= 0.0:
            return -math.inf
        return math.log(prob, base)

    def sequence_probability(
        self,
        tokens: Sequence[str],
        context_window: Optional[int] = None,
    ) -> float:
        """Calculate joint probability of a sequence of tokens.

        P(w_1, ..., w_m) = P(w_1) * prod_{i=2}^m P(w_i | w_{max(1, i-k+1)}^{i-1})

        Parameters
        ----------
        tokens : Sequence[str]
            Sequence of tokens.
        context_window : Optional[int], default=None
            Maximum context length. Defaults to self.order - 1.

        Returns
        -------
        float
            Joint sequence probability in range [0.0, 1.0]. Returns 1.0 for empty sequence.
        """
        if not tokens:
            return 1.0

        k = context_window if context_window is not None else max(1, self.order - 1)
        prob = 1.0

        for i, token in enumerate(tokens):
            if i == 0:
                p_i = self.probability(token, context=())
            else:
                ctx_start = max(0, i - k)
                ctx = tuple(tokens[ctx_start:i])
                p_i = self.probability(token, context=ctx)

            prob *= p_i
            if prob == 0.0:
                break

        return prob

    def sequence_log_probability(
        self,
        tokens: Sequence[str],
        context_window: Optional[int] = None,
        base: Optional[float] = None,
    ) -> float:
        """Calculate joint log probability (cross-entropy summation) of a sequence.

        Parameters
        ----------
        tokens : Sequence[str]
            Sequence of tokens.
        context_window : Optional[int], default=None
            Maximum context length. Defaults to self.order - 1.
        base : Optional[float], default=None
            Logarithm base. If None, uses natural base e.

        Returns
        -------
        float
            Sum of log probabilities, or -math.inf if any transition is 0.0.
        """
        if not tokens:
            return 0.0

        k = context_window if context_window is not None else max(1, self.order - 1)
        total_log_p = 0.0

        for i, token in enumerate(tokens):
            if i == 0:
                ctx: Tuple[str, ...] = ()
            else:
                ctx_start = max(0, i - k)
                ctx = tuple(tokens[ctx_start:i])

            p_i = self.probability(token, context=ctx)
            if p_i <= 0.0:
                return -math.inf
            if base is None or base == math.e:
                total_log_p += math.log(p_i)
            else:
                total_log_p += math.log(p_i, base)

        return total_log_p

    def evaluate_metrics(
        self,
        sentences: Sequence[Sequence[str]],
    ) -> Tuple[float, float, float, int, int]:
        """Calculate comprehensive negative log-likelihood, cross-entropy, and perplexity.

        Uses natural logarithm consistently:
            NLL_total = - sum_{s} sum_{w in s} ln P(w | context)
            H = NLL_total / M  (mean cross-entropy in nats)
            PPL = exp(H)

        Parameters
        ----------
        sentences : Sequence[Sequence[str]]
            Collection of sentences strictly evaluated within sentence boundaries.

        Returns
        -------
        Tuple[float, float, float, int, int]
            (total_nll, mean_cross_entropy, perplexity, zero_events, total_prediction_events)
        """
        total_tokens = sum(len(s) for s in sentences)
        if total_tokens == 0:
            return 0.0, 0.0, 1.0, 0, 0

        total_nll = 0.0
        zero_events = 0
        k = max(1, self.order - 1)

        for sentence in sentences:
            for i, token in enumerate(sentence):
                if i == 0:
                    ctx: Tuple[str, ...] = ()
                else:
                    ctx_start = max(0, i - k)
                    ctx = tuple(sentence[ctx_start:i])

                p = self.probability(token, context=ctx)
                if p <= 0.0:
                    zero_events += 1
                else:
                    total_nll += -math.log(p)

        if zero_events > 0:
            return float("inf"), float("inf"), float("inf"), zero_events, total_tokens

        mean_h = total_nll / total_tokens
        ppl = math.exp(mean_h)
        return total_nll, mean_h, ppl, 0, total_tokens

    def perplexity(
        self,
        sentences: Sequence[Sequence[str]],
        base: Optional[float] = None,
    ) -> float:
        """Calculate test-set perplexity strictly respecting sentence boundaries:

        PPL(W) = exp( - (1/M) * sum_{s} sum_{w in s} ln P(w | context) )

        Parameters
        ----------
        sentences : Sequence[Sequence[str]]
            Collection of test sentences.
        base : Optional[float], default=None
            Logarithm base. If None, uses natural base e (PPL = exp(H)).

        Returns
        -------
        float
            Perplexity score. Returns math.inf if any token has zero probability.
        """
        _, _, ppl, _, _ = self.evaluate_metrics(sentences)
        if base is not None and base != math.e and ppl != float("inf"):
            # If a custom base b was requested: b^( -1/M * sum log_b P )
            # Change of base: H_b = H_e / ln(b), so b^H_b = (e^ln(b))^(H_e / ln(b)) = e^H_e = ppl!
            # Mathematically, perplexity is invariant to log base!
            pass
        return ppl

    def get_parameters(self) -> Dict[str, Any]:
        """Return dictionary of smoother hyperparameters."""
        return {
            "smoother_name": self.name,
            "order": self.order,
            "vocabulary_size": self.vocabulary_size,
            "total_tokens": self.total_tokens,
        }
