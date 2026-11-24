"""Base N-Gram Language Model Architecture.

Provides an extensible, object-oriented foundation for Unigram, Bigram,
and Trigram language models using Maximum Likelihood Estimation.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from five_head_nlp.probability.mle import NGramProbabilityEngine
from five_head_nlp.tokenization.vocabulary import Vocabulary


class BaseNGramModel(ABC):
    """Abstract Base Class for N-Gram Language Models.

    Orchestrates high-level vocabulary management, sentence-boundary-preserving
    N-gram probability estimation, model inspection, and safe JSON serialization.
    """

    def __init__(self, n: int = 1) -> None:
        """Initialize the base N-gram model.

        Parameters
        ----------
        n : int
            The N-gram order (1=unigram, 2=bigram, 3=trigram).

        Raises
        ------
        ValueError
            If n < 1.
        """
        if n < 1:
            raise ValueError(f"N-gram order must be an integer >= 1, got {n}")

        self.n: int = n
        self.engine: NGramProbabilityEngine = NGramProbabilityEngine(
            n=n, include_lower_orders=True
        )
        self.vocabulary: Optional[Vocabulary] = None
        self._is_fitted: bool = False
        self.metadata: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Return the class name of the model."""
        return self.__class__.__name__

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been trained."""
        return self._is_fitted

    @property
    def vocabulary_size(self) -> int:
        """Return size of full vocabulary (|V|)."""
        if self.vocabulary is None:
            return 0
        return self.vocabulary.size

    @property
    def total_tokens(self) -> int:
        """Return total tokens processed during training."""
        if not self._is_fitted:
            return 0
        return self.engine.total_tokens

    def fit(self, sentences: Sequence[Sequence[str]]) -> "BaseNGramModel":
        """Train the model on tokenized sentences.

        Sentence boundaries are strictly preserved.

        Parameters
        ----------
        sentences : Sequence[Sequence[str]]
            Collection of sentences where each sentence is a sequence of tokens.

        Returns
        -------
        BaseNGramModel
            Self for method chaining.

        Raises
        ------
        ValueError
            If sentences is empty or contains no tokens.
        """
        if not sentences:
            raise ValueError("Cannot train model on empty sentences collection.")

        # Train underlying probability engine
        self.engine.fit(sentences)

        # Build vocabulary
        self.vocabulary = Vocabulary(sentences)

        self._is_fitted = True
        self.metadata = {
            "model_type": self.name,
            "order": self.n,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "sentence_count": len(sentences),
            "total_tokens": self.engine.total_tokens,
            "vocabulary_size": self.vocabulary.size,
            "lexical_vocabulary_size": self.vocabulary.lexical_size,
        }
        return self

    def fit_corpus(self, corpus_loader: object) -> "BaseNGramModel":
        """Train the model using a CorpusLoader instance.

        Parameters
        ----------
        corpus_loader : object
            CorpusLoader possessing a get_sentences() method.

        Returns
        -------
        BaseNGramModel
            Self for method chaining.
        """
        if not hasattr(corpus_loader, "get_sentences"):
            raise TypeError("corpus_loader must provide a get_sentences() method.")
        return self.fit(corpus_loader.get_sentences())

    def _check_fitted(self) -> None:
        """Raise RuntimeError if model has not been trained."""
        if not self._is_fitted or self.vocabulary is None:
            raise RuntimeError(f"{self.name} must be trained before querying probabilities.")

    def _normalize_context(
        self, context: Optional[Union[str, Sequence[str]]]
    ) -> Tuple[str, ...]:
        """Convert string or sequence context into a normalized tuple."""
        if context is None:
            return ()
        if isinstance(context, str):
            return (context,)
        return tuple(context)

    def is_known_token(self, token: str) -> bool:
        """Return True if the token exists in the model's vocabulary.

        Parameters
        ----------
        token : str
            Token string to test.

        Returns
        -------
        bool
            True if token is present in training vocabulary, False otherwise.
        """
        if self.vocabulary is None:
            return False
        return token in self.vocabulary

    def is_known_context(
        self, context: Optional[Union[str, Sequence[str]]]
    ) -> bool:
        """Return True if context was observed preceding at least one continuation.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]]
            Preceding context tokens.

        Returns
        -------
        bool
            True if context count > 0 in training data, False otherwise.
        """
        if not self._is_fitted:
            return False
        ctx_tuple = self._normalize_context(context)
        return self.engine.get_context_count(ctx_tuple) > 0

    @abstractmethod
    def probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
    ) -> float:
        """Calculate conditional probability P(token | context)."""
        pass

    def conditional_probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
    ) -> float:
        """Explicit alias for probability(token, context)."""
        return self.probability(token, context=context)

    def log_probability(
        self,
        token: str,
        context: Optional[Union[str, Sequence[str]]] = None,
        base: float = 2.0,
    ) -> float:
        """Calculate base-b log probability log_b P(token | context).

        Returns -math.inf if probability is 0.0.
        """
        prob = self.probability(token, context=context)
        if prob <= 0.0:
            return -math.inf
        return math.log(prob, base)

    @abstractmethod
    def get_candidates(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> List[str]:
        """Return candidate next tokens for context in deterministic order."""
        pass

    @abstractmethod
    def get_candidate_distribution(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, float]:
        """Return normalized probability distribution over candidate next tokens."""
        pass

    @abstractmethod
    def sequence_probability(self, tokens: Sequence[str]) -> float:
        """Calculate joint probability of a sequence of tokens."""
        pass

    def sequence_log_probability(
        self, tokens: Sequence[str], base: float = 2.0
    ) -> float:
        """Calculate joint log probability of a sequence of tokens.

        Returns -math.inf if any transition has zero probability.
        """
        prob = self.sequence_probability(tokens)
        if prob <= 0.0:
            return -math.inf
        return math.log(prob, base)

    def get_model_info(self) -> Dict[str, Any]:
        """Return summary dictionary describing model parameters and training state."""
        return {
            "model_type": self.name,
            "order": self.n,
            "is_fitted": self._is_fitted,
            "vocabulary_size": self.vocabulary_size,
            "total_tokens": self.total_tokens,
            "metadata": dict(self.metadata),
        }

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Export complete model state to a JSON-serializable dictionary."""
        self._check_fitted()
        return {
            "model_type": self.name,
            "order": self.n,
            "metadata": self.metadata,
            "vocabulary": self.vocabulary.to_dict() if self.vocabulary else None,
            "engine": self.engine.to_dict(),
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Save model to a JSON file safely.

        Parameters
        ----------
        path : Union[str, Path]
            Destination file path.

        Returns
        -------
        Path
            Absolute path to the saved model file.
        """
        target_path = Path(path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        state = self.to_dict()
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        return target_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseNGramModel":
        """Load and reconstitute a model from a saved JSON file.

        Validates schema, instantiates the appropriate model class,
        and restores exact engine and vocabulary states.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the saved JSON model file.

        Returns
        -------
        BaseNGramModel
            Reconstructed operational model instance.

        Raises
        ------
        FileNotFoundError
            If path does not exist.
        ValueError
            If file contains invalid or corrupted model data.
        """
        source_path = Path(path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Model file not found at {source_path}")

        with open(source_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt JSON model file: {exc}") from exc

        # Schema validation
        required_keys = {"model_type", "order", "engine", "vocabulary"}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            raise ValueError(f"Invalid model file: missing required keys {missing}")

        model_type_str = data["model_type"]
        order = data["order"]

        # Circular import resolution for polymorphic loading
        from five_head_nlp.models.unigram import UnigramModel
        from five_head_nlp.models.bigram import BigramModel
        from five_head_nlp.models.trigram import TrigramModel

        model_class_map: Dict[str, Type[BaseNGramModel]] = {
            "UnigramModel": UnigramModel,
            "BigramModel": BigramModel,
            "TrigramModel": TrigramModel,
        }

        target_cls = model_class_map.get(model_type_str)
        if target_cls is None:
            raise ValueError(f"Unknown model_type in file: {model_type_str}")

        # If load was called directly on a specific subclass, verify compatibility
        if cls is not BaseNGramModel and cls is not target_cls:
            raise ValueError(
                f"Cannot load {model_type_str} into requested class {cls.__name__}"
            )

        # Instantiate instance
        instance = target_cls()
        instance.engine = NGramProbabilityEngine.from_dict(data["engine"])
        instance.vocabulary = Vocabulary.from_dict(data["vocabulary"])
        instance.metadata = data.get("metadata", {})
        instance._is_fitted = True
        return instance
