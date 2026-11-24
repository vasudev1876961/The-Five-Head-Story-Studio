"""Vocabulary management and frequency analysis module.

Builds deterministic token-to-ID mappings, frequency distributions,
and dual full/lexical vocabulary representations for the Five-Head Story-Generation System.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from five_head_nlp.tokenization.corpus_loader import is_punctuation_token


class Vocabulary:
    """Manages vocabulary mappings, unigram distributions, and lexical subsets."""

    def __init__(
        self,
        tokens_or_sentences: Iterable[str] | Iterable[Iterable[str]] | None = None,
    ) -> None:
        """Initialize and construct the vocabulary.

        Args:
            tokens_or_sentences: Either a flat sequence of token strings, or a sequence
                                 of token lists (e.g., from CorpusLoader.get_sentences()).
                                 If None, initializes an empty vocabulary.
        """
        self._frequencies: Counter[str] = Counter()
        self._total_token_occurrences: int = 0

        self._tokens: list[str] = []
        self._lexical_tokens: list[str] = []
        self._punctuation_tokens: list[str] = []

        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

        if tokens_or_sentences is not None:
            self.build(tokens_or_sentences)

    def build(
        self,
        tokens_or_sentences: Iterable[str] | Iterable[Iterable[str]],
    ) -> Vocabulary:
        """Construct vocabulary and frequency tables from tokens or sentence lists.

        Vocabulary ordering is strictly deterministic:
        1. Descending token frequency.
        2. Alphabetical (lexicographical) order for equal frequencies.

        Args:
            tokens_or_sentences: Flat sequence of tokens or list of sentence token lists.

        Returns:
            The Vocabulary instance (for chaining).
        """
        # Flatten input if nested
        flat_tokens: list[str] = []
        for item in tokens_or_sentences:
            if isinstance(item, str):
                flat_tokens.append(item)
            elif isinstance(item, Iterable):
                for sub_item in item:
                    if isinstance(sub_item, str):
                        flat_tokens.append(sub_item)
                    else:
                        raise TypeError(
                            f"Expected string token, got {type(sub_item).__name__}"
                        )
            else:
                raise TypeError(f"Expected token string or list of tokens, got {type(item).__name__}")

        self._total_token_occurrences = len(flat_tokens)
        self._frequencies = Counter(flat_tokens)

        # Sort deterministically: descending frequency (-count), then ascending token string (token)
        sorted_tokens = sorted(
            self._frequencies.keys(),
            key=lambda token: (-self._frequencies[token], token),
        )

        self._tokens = sorted_tokens
        self._lexical_tokens = [t for t in sorted_tokens if not is_punctuation_token(t)]
        self._punctuation_tokens = [t for t in sorted_tokens if is_punctuation_token(t)]

        self._token_to_id = {token: idx for idx, token in enumerate(sorted_tokens)}
        self._id_to_token = {idx: token for idx, token in enumerate(sorted_tokens)}
        return self

    # --- Sizing and Properties ---

    def __len__(self) -> int:
        """Return the total number of unique tokens in the full vocabulary."""
        return len(self._tokens)

    def __contains__(self, token: str) -> bool:
        """Check if a token exists in the vocabulary."""
        return token in self._token_to_id

    @property
    def size(self) -> int:
        """Total unique tokens in the full vocabulary (words + punctuation)."""
        return len(self._tokens)

    @property
    def lexical_size(self) -> int:
        """Total unique lexical word tokens (excluding punctuation)."""
        return len(self._lexical_tokens)

    @property
    def punctuation_size(self) -> int:
        """Total unique punctuation tokens."""
        return len(self._punctuation_tokens)

    @property
    def total_tokens(self) -> int:
        """Total token occurrences processed during vocabulary construction (N)."""
        return self._total_token_occurrences

    @property
    def tokens(self) -> list[str]:
        """Full vocabulary tokens in deterministic order."""
        return list(self._tokens)

    @property
    def lexical_tokens(self) -> list[str]:
        """Lexical word tokens in deterministic order."""
        return list(self._lexical_tokens)

    @property
    def punctuation_tokens(self) -> list[str]:
        """Punctuation tokens in deterministic order."""
        return list(self._punctuation_tokens)

    # --- Lookups ---

    def token_to_id(self, token: str) -> int:
        """Get the integer ID for a token.

        Args:
            token: Token string.

        Returns:
            The integer ID assigned to the token.

        Raises:
            KeyError: If the token is not present in the vocabulary.
        """
        if token not in self._token_to_id:
            raise KeyError(f"Token '{token}' not found in vocabulary.")
        return self._token_to_id[token]

    def get_id(self, token: str, default: int | None = None) -> int | None:
        """Safely get the integer ID for a token with a fallback default.

        Args:
            token: Token string.
            default: Value to return if token is not found.

        Returns:
            Integer ID or default.
        """
        return self._token_to_id.get(token, default)

    def id_to_token(self, token_id: int) -> str:
        """Get the token string associated with an integer ID.

        Args:
            token_id: Integer ID.

        Returns:
            Token string.

        Raises:
            KeyError: If token_id is not in the vocabulary.
        """
        if token_id not in self._id_to_token:
            raise KeyError(f"ID {token_id} not found in vocabulary.")
        return self._id_to_token[token_id]

    def get_token(self, token_id: int, default: str | None = None) -> str | None:
        """Safely get the token string for an ID with a fallback default.

        Args:
            token_id: Integer ID.
            default: Value to return if token_id is not found.

        Returns:
            Token string or default.
        """
        return self._id_to_token.get(token_id, default)

    def get_frequency(self, token: str) -> int:
        """Return the unigram frequency count of a token f(w).

        Args:
            token: Token string.

        Returns:
            Integer count of occurrences (0 if not present).
        """
        return self._frequencies.get(token, 0)

    def get_probability(self, token: str) -> float:
        """Return the empirical unigram probability P(w) = f(w) / N.

        Args:
            token: Token string.

        Returns:
            Unigram probability float (0.0 if not present or vocabulary empty).
        """
        if self._total_token_occurrences == 0:
            return 0.0
        return self._frequencies.get(token, 0) / self._total_token_occurrences

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Export vocabulary mappings and frequency statistics to a JSON-compatible dict."""
        return {
            "vocabulary_size": self.size,
            "lexical_vocabulary_size": self.lexical_size,
            "punctuation_vocabulary_size": self.punctuation_size,
            "total_token_occurrences": self.total_tokens,
            "full_vocabulary": self.tokens,
            "lexical_vocabulary": self.lexical_tokens,
            "punctuation_vocabulary": self.punctuation_tokens,
            "token_to_id": dict(self._token_to_id),
            "id_to_token": {str(k): v for k, v in self._id_to_token.items()},
            "token_frequencies": dict(self._frequencies),
            "token_probabilities": {
                token: round(self.get_probability(token), 6) for token in self._tokens
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        """Reconstruct a Vocabulary instance from a dictionary exported via to_dict().

        Args:
            data: Dictionary containing vocabulary state.

        Returns:
            Reconstituted Vocabulary instance.
        """
        vocab = cls()
        tokens = list(data.get("full_vocabulary", []))
        vocab._tokens = tokens
        vocab._lexical_tokens = list(
            data.get(
                "lexical_vocabulary",
                [t for t in tokens if not is_punctuation_token(t)],
            )
        )
        vocab._punctuation_tokens = list(
            data.get(
                "punctuation_vocabulary",
                [t for t in tokens if is_punctuation_token(t)],
            )
        )
        raw_freqs = data.get("token_frequencies", {})
        vocab._frequencies = Counter(raw_freqs)
        vocab._total_token_occurrences = data.get(
            "total_token_occurrences", sum(vocab._frequencies.values())
        )
        vocab._token_to_id = {token: idx for idx, token in enumerate(tokens)}
        vocab._id_to_token = {idx: token for idx, token in enumerate(tokens)}
        return vocab
