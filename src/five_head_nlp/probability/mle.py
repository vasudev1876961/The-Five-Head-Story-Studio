"""Maximum Likelihood Estimation (MLE) for N-Gram Language Models.

This module provides mathematically rigorous N-gram sequence extraction and
MLE probability calculation strictly respecting sentence boundaries.

Mathematical Formulations:
--------------------------
1. Unigram MLE Probability:
   P_MLE(w) = Count(w) / N_total

2. Bigram MLE Probability:
   P_MLE(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})

3. Trigram MLE Probability:
   P_MLE(w_i | w_{i-2}, w_{i-1}) = Count(w_{i-2}, w_{i-1}, w_i) / Count(w_{i-2}, w_{i-1})

4. General N-Gram MLE Probability:
   P_MLE(w_i | w_{i-n+1}^{i-1}) = Count(w_{i-n+1}^i) / Count(w_{i-n+1}^{i-1})
   where context count is the sum of outgoing transitions:
   Count(c) = sum_{v in V} Count(c, v)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


def extract_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Extract N-grams of order n from a single sequence of tokens.

    Parameters
    ----------
    tokens : Sequence[str]
        A sequence of word/punctuation tokens.
    n : int
        The N-gram order (must be an integer >= 1).

    Returns
    -------
    List[Tuple[str, ...]]
        A list of n-tuples representing the sliding window over tokens.
        Returns an empty list if len(tokens) < n.

    Raises
    ------
    ValueError
        If n < 1.
    """
    if n < 1:
        raise ValueError(f"N-gram order must be an integer >= 1, got {n}")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def extract_sentence_ngrams(
    sentences: Sequence[Sequence[str]], n: int
) -> List[Tuple[str, ...]]:
    """Extract N-grams of order n across sentences, strictly respecting boundaries.

    No N-gram is ever constructed spanning across sentence boundaries.
    For example, the sentence-ending period '.' of sentence k will never
    form a bigram or trigram with the first token of sentence k+1.

    Parameters
    ----------
    sentences : Sequence[Sequence[str]]
        A collection of tokenized sentences.
    n : int
        The N-gram order (must be an integer >= 1).

    Returns
    -------
    List[Tuple[str, ...]]
        A flat list of all intra-sentence N-grams across all sentences.

    Raises
    ------
    ValueError
        If n < 1.
    """
    if n < 1:
        raise ValueError(f"N-gram order must be an integer >= 1, got {n}")

    all_ngrams: List[Tuple[str, ...]] = []
    for sentence in sentences:
        if len(sentence) >= n:
            for i in range(len(sentence) - n + 1):
                all_ngrams.append(tuple(sentence[i : i + n]))
    return all_ngrams


def extract_unigrams(sentences: Sequence[Sequence[str]]) -> List[Tuple[str]]:
    """Convenience helper extracting intra-sentence 1-tuples (unigrams)."""
    return [tuple([tok]) for s in sentences for tok in s]


def extract_bigrams(sentences: Sequence[Sequence[str]]) -> List[Tuple[str, str]]:
    """Convenience helper extracting intra-sentence 2-tuples (bigrams)."""
    return [(s[i], s[i + 1]) for s in sentences for i in range(len(s) - 1)]


def extract_trigrams(
    sentences: Sequence[Sequence[str]],
) -> List[Tuple[str, str, str]]:
    """Convenience helper extracting intra-sentence 3-tuples (trigrams)."""
    return [
        (s[i], s[i + 1], s[i + 2])
        for s in sentences
        for i in range(len(s) - 2)
    ]


class NGramProbabilityEngine:
    """Reusable N-Gram Probability Engine using Maximum Likelihood Estimation.

    Calculates empirical frequency distributions, conditional transition
    probabilities, and joint sequence probabilities up to order n.

    Guarantees strict intra-sentence isolation without artificial boundary crossing.
    """

    def __init__(self, n: int = 2, include_lower_orders: bool = True) -> None:
        """Initialize the N-Gram probability engine.

        Parameters
        ----------
        n : int, default=2
            Maximum order of N-grams to calculate (1=unigram, 2=bigram, 3=trigram).
        include_lower_orders : bool, default=True
            Whether to also extract and store counts for orders 1 through n-1.
            Required for backoff, interpolation, and conditional MLE lookups.

        Raises
        ------
        ValueError
            If n < 1.
        """
        if n < 1:
            raise ValueError(f"N-gram order must be >= 1, got {n}")

        self.n: int = n
        self.include_lower_orders: bool = include_lower_orders

        # Internal state
        self._is_fitted: bool = False
        self._total_tokens: int = 0
        self._vocabulary: Set[str] = set()

        # N-gram counters by order: order -> Counter(Tuple[str, ...])
        self._ngram_counts: Dict[int, Counter[Tuple[str, ...]]] = defaultdict(Counter)

        # Context counts by order: order -> Counter(Tuple[str, ...])
        # For order k, context has length k-1.
        # Context count C(c) = sum_{w} C(c + (w,))
        self._context_counts: Dict[int, Counter[Tuple[str, ...]]] = defaultdict(Counter)

        # Transition dictionaries: order -> Dict[ContextTuple, Counter[NextToken]]
        self._transitions: Dict[int, Dict[Tuple[str, ...], Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )

        # Total number of n-grams extracted per order
        self._total_ngrams: Dict[int, int] = defaultdict(int)

    @property
    def is_fitted(self) -> bool:
        """Return True if the engine has been trained on a corpus."""
        return self._is_fitted

    @property
    def total_tokens(self) -> int:
        """Return total number of tokens observed during training."""
        return self._total_tokens

    @property
    def vocabulary(self) -> Set[str]:
        """Return the set of unique tokens observed during training."""
        return set(self._vocabulary)

    @property
    def vocabulary_size(self) -> int:
        """Return the size of the active vocabulary |V|."""
        return len(self._vocabulary)

    def fit(self, sentences: Sequence[Sequence[str]]) -> "NGramProbabilityEngine":
        """Fit the probability engine on tokenized sentences.

        Sentence boundaries are strictly preserved. Tokens at the end of a sentence
        never transition to tokens at the beginning of another sentence.

        Parameters
        ----------
        sentences : Sequence[Sequence[str]]
            A collection of sentences, where each sentence is a sequence of tokens.

        Returns
        -------
        NGramProbabilityEngine
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If sentences is empty.
        """
        if not sentences:
            raise ValueError("Cannot fit probability engine on empty sentence list.")

        # Reset state
        self._total_tokens = 0
        self._vocabulary.clear()
        self._ngram_counts.clear()
        self._context_counts.clear()
        self._transitions.clear()
        self._total_ngrams.clear()

        # Count total tokens and populate unigram vocabulary
        for sentence in sentences:
            self._total_tokens += len(sentence)
            for token in sentence:
                self._vocabulary.add(token)

        # Determine orders to extract
        orders_to_fit = range(1, self.n + 1) if self.include_lower_orders else [self.n]

        for k in orders_to_fit:
            ngrams_k = extract_sentence_ngrams(sentences, k)
            self._total_ngrams[k] = len(ngrams_k)
            k_counts = Counter(ngrams_k)
            self._ngram_counts[k] = k_counts

            if k == 1:
                # Unigram context is empty tuple ()
                self._context_counts[1][()] = self._total_tokens
                for (token,), count in k_counts.items():
                    self._transitions[1][()][token] = count
            else:
                for ngram, count in k_counts.items():
                    context = ngram[:-1]
                    next_token = ngram[-1]
                    self._context_counts[k][context] += count
                    self._transitions[k][context][next_token] += count

        self._is_fitted = True
        return self

    def fit_corpus(self, corpus_loader: object) -> "NGramProbabilityEngine":
        """Fit using an existing CorpusLoader instance.

        Parameters
        ----------
        corpus_loader : object
            Instance of CorpusLoader possessing a get_sentences() method.

        Returns
        -------
        NGramProbabilityEngine
            Returns self for method chaining.
        """
        if not hasattr(corpus_loader, "get_sentences"):
            raise TypeError("corpus_loader must provide a get_sentences() method.")
        return self.fit(corpus_loader.get_sentences())

    def _normalize_context(
        self, context: Optional[Union[str, Sequence[str]]]
    ) -> Tuple[str, ...]:
        """Normalize context argument into a consistent tuple of strings."""
        if context is None:
            return ()
        if isinstance(context, str):
            return (context,)
        return tuple(context)

    def get_count(self, ngram: Tuple[str, ...]) -> int:
        """Get occurrence frequency of a specific N-gram tuple.

        Parameters
        ----------
        ngram : Tuple[str, ...]
            The N-gram tuple to look up.

        Returns
        -------
        int
            Occurrence count (0 if unseen or order not tracked).
        """
        k = len(ngram)
        if k in self._ngram_counts:
            return self._ngram_counts[k].get(ngram, 0)
        return 0

    def get_unigram_count(self, token: str) -> int:
        """Get the occurrence count of a single unigram token."""
        return self.get_count((token,))

    def get_context_count(
        self, context: Optional[Union[str, Sequence[str]]], order: Optional[int] = None
    ) -> int:
        """Get total number of times a context was followed by any token.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]]
            Context tokens preceding the target word.
        order : Optional[int], default=None
            The N-gram order. If None, inferred as len(context) + 1.

        Returns
        -------
        int
            Sum of outgoing transition counts for this context.
        """
        ctx_tuple = self._normalize_context(context)
        k = order if order is not None else len(ctx_tuple) + 1

        if k in self._context_counts:
            return self._context_counts[k].get(ctx_tuple, 0)
        return 0

    def get_total_ngrams(self, order: Optional[int] = None) -> int:
        """Get total number of N-grams extracted for the given order.

        Parameters
        ----------
        order : Optional[int], default=None
            The order to check. Defaults to primary order self.n.

        Returns
        -------
        int
            Total N-gram count.
        """
        k = order if order is not None else self.n
        return self._total_ngrams.get(k, 0)

    def get_candidates(
        self, context: Optional[Union[str, Sequence[str]]], order: Optional[int] = None
    ) -> List[str]:
        """Return candidate next tokens following context, deterministically ordered.

        Ordered by descending transition frequency, breaking ties alphabetically.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]]
            Context tokens preceding the target word.
        order : Optional[int], default=None
            The N-gram order. If None, inferred as len(context) + 1.

        Returns
        -------
        List[str]
            List of candidate continuation tokens.
        """
        ctx_tuple = self._normalize_context(context)
        k = order if order is not None else len(ctx_tuple) + 1

        if k in self._transitions and ctx_tuple in self._transitions[k]:
            next_counts = self._transitions[k][ctx_tuple]
            return sorted(next_counts.keys(), key=lambda w: (-next_counts[w], w))
        return []

    def get_candidate_counts(
        self, context: Optional[Union[str, Sequence[str]]], order: Optional[int] = None
    ) -> Dict[str, int]:
        """Return dictionary mapping candidate next tokens to their transition frequencies.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]]
            Context tokens preceding the target word.
        order : Optional[int], default=None
            The N-gram order. If None, inferred as len(context) + 1.

        Returns
        -------
        Dict[str, int]
            Map of {token: count}.
        """
        ctx_tuple = self._normalize_context(context)
        k = order if order is not None else len(ctx_tuple) + 1

        if k in self._transitions and ctx_tuple in self._transitions[k]:
            return dict(self._transitions[k][ctx_tuple])
        return {}

    def get_candidate_distribution(
        self, context: Optional[Union[str, Sequence[str]]], order: Optional[int] = None
    ) -> Dict[str, float]:
        """Calculate empirical MLE probability distribution over candidate next tokens.

        P(w | context) = Count(context, w) / Count(context)

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]]
            Context tokens preceding the target word.
        order : Optional[int], default=None
            The N-gram order. If None, inferred as len(context) + 1.

        Returns
        -------
        Dict[str, float]
            Map of {token: probability}. Probabilities sum to 1.0 (unless context is unseen,
            which returns empty dict).
        """
        counts = self.get_candidate_counts(context, order=order)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {w: c / total for w, c in counts.items()}

    def probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
        order: Optional[int] = None,
    ) -> float:
        """Calculate MLE conditional probability P_MLE(token | context).

        Parameters
        ----------
        token : str
            The candidate continuation word/punctuation.
        context : Optional[Union[str, Sequence[str]]], default=None
            The preceding context tokens. If None or empty, computes unigram MLE P(token).
        order : Optional[int], default=None
            The order of the N-gram model. If None, inferred as len(context) + 1.

        Returns
        -------
        float
            Empirical probability in range [0.0, 1.0].
        """
        if not self._is_fitted:
            raise RuntimeError("NGramProbabilityEngine must be fitted before computing probabilities.")

        ctx_tuple = self._normalize_context(context)
        k = order if order is not None else len(ctx_tuple) + 1

        if k == 1 or len(ctx_tuple) == 0:
            # Unigram MLE: Count(token) / N_total
            if self._total_tokens == 0:
                return 0.0
            return self.get_unigram_count(token) / self._total_tokens

        # Higher-order conditional MLE: Count(context, token) / Count(context)
        context_count = self.get_context_count(ctx_tuple, order=k)
        if context_count == 0:
            return 0.0

        ngram = ctx_tuple + (token,)
        ngram_count = self.get_count(ngram)
        return ngram_count / context_count

    def log_probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
        order: Optional[int] = None,
        base: float = 2.0,
    ) -> float:
        """Calculate log MLE conditional probability log_b P_MLE(token | context).

        Returns -infinity for zero probability.

        Parameters
        ----------
        token : str
            The candidate continuation token.
        context : Optional[Union[str, Sequence[str]]], default=None
            The preceding context tokens.
        order : Optional[int], default=None
            The N-gram order.
        base : float, default=2.0
            The logarithm base (default base 2 for information theoretic bits).

        Returns
        -------
        float
            Log probability, or -math.inf if P == 0.0.
        """
        prob = self.probability(token, context, order=order)
        if prob <= 0.0:
            return -math.inf
        return math.log(prob, base)

    def p_unigram(self, token: str) -> float:
        """Calculate unigram MLE probability P_MLE(w) = Count(w) / N_total."""
        return self.probability(token, context=(), order=1)

    def p_bigram(self, token: str, prev_token: str) -> float:
        """Calculate bigram MLE probability P_MLE(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})."""
        return self.probability(token, context=(prev_token,), order=2)

    def p_trigram(self, token: str, prev_token2: str, prev_token1: str) -> float:
        """Calculate trigram MLE probability P_MLE(w_i | w_{i-2}, w_{i-1}) = Count(w_{i-2}, w_{i-1}, w_i) / Count(w_{i-2}, w_{i-1})."""
        return self.probability(token, context=(prev_token2, prev_token1), order=3)

    def sequence_probability(
        self, sequence: Sequence[str], order: Optional[int] = None
    ) -> float:
        """Calculate the joint autoregressive probability of a token sequence.

        Uses the chain rule of probability:
        P(w_1, ..., w_m) = P(w_1) * prod_{i=2}^m P(w_i | w_{max(1, i-k+1)}^{i-1})

        Parameters
        ----------
        sequence : Sequence[str]
            A sequence of tokens representing a sentence or phrase.
        order : Optional[int], default=None
            Maximum N-gram order used for context. Defaults to self.n.

        Returns
        -------
        float
            Joint probability in range [0.0, 1.0]. Returns 1.0 for empty sequence.
        """
        if not sequence:
            return 1.0

        k = order if order is not None else self.n
        prob = 1.0

        for i, token in enumerate(sequence):
            if i == 0:
                p_i = self.p_unigram(token)
            else:
                context_start = max(0, i - (k - 1))
                context = tuple(sequence[context_start:i])
                p_i = self.probability(token, context=context)

            prob *= p_i
            if prob == 0.0:
                break

        return prob

    def branching_factor(
        self, context: Optional[Union[str, Sequence[str]]], order: Optional[int] = None
    ) -> int:
        """Calculate the branching factor (number of unique next tokens) for a context.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]]
            The context tokens.
        order : Optional[int], default=None
            The N-gram order.

        Returns
        -------
        int
            Number of distinct continuation tokens observed following this context.
        """
        return len(self.get_candidates(context, order=order))

    def average_branching_factor(self, order: Optional[int] = None) -> float:
        """Calculate the average branching factor across all active contexts.

        Parameters
        ----------
        order : Optional[int], default=None
            The N-gram order. Defaults to self.n.

        Returns
        -------
        float
            Average branching factor. Returns 0.0 if no contexts exist.
        """
        k = order if order is not None else self.n
        if k not in self._transitions or not self._transitions[k]:
            return 0.0

        total_branches = sum(
            len(followers) for followers in self._transitions[k].values()
        )
        total_contexts = len(self._transitions[k])
        return total_branches / total_contexts

    def to_dict(self) -> Dict[str, Any]:
        """Export engine state to a JSON-serializable dictionary.

        Returns
        -------
        Dict[str, Any]
            State dictionary containing configuration, vocabulary, counts, and metadata.
        """
        return {
            "n": self.n,
            "include_lower_orders": self.include_lower_orders,
            "is_fitted": self._is_fitted,
            "total_tokens": self._total_tokens,
            "vocabulary_size": len(self._vocabulary),
            "vocabulary": sorted(list(self._vocabulary)),
            "total_ngrams": {str(k): v for k, v in self._total_ngrams.items()},
            "ngram_counts": {
                str(k): [
                    {"tokens": list(ngram), "count": count}
                    for ngram, count in counts.items()
                ]
                for k, counts in self._ngram_counts.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NGramProbabilityEngine":
        """Reconstruct engine state from a serialized dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            State dictionary produced by to_dict().

        Returns
        -------
        NGramProbabilityEngine
            Fully reconstituted, operational probability engine.
        """
        engine = cls(
            n=data["n"],
            include_lower_orders=data.get("include_lower_orders", True),
        )
        engine._is_fitted = data.get("is_fitted", False)
        engine._total_tokens = data.get("total_tokens", 0)
        engine._vocabulary = set(data.get("vocabulary", []))

        total_ngrams_data = data.get("total_ngrams", {})
        for str_k, count in total_ngrams_data.items():
            engine._total_ngrams[int(str_k)] = count

        ngram_counts_data = data.get("ngram_counts", {})
        for str_k, item_list in ngram_counts_data.items():
            k = int(str_k)
            for item in item_list:
                tokens = tuple(item["tokens"])
                count = item["count"]
                engine._ngram_counts[k][tokens] = count
                if k == 1:
                    token = tokens[0]
                    engine._context_counts[1][()] = engine._total_tokens
                    engine._transitions[1][()][token] = count
                else:
                    context = tokens[:-1]
                    next_token = tokens[-1]
                    engine._context_counts[k][context] += count
                    engine._transitions[k][context][next_token] += count

        return engine
