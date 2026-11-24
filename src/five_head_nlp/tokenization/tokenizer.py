"""Word-level tokenizer for the Five-Head Story-Generation System.

Provides deterministic tokenization, punctuation isolation, case normalization,
and contraction preservation.

Tokenization Policies:
1. Words: Contiguous sequences of alphanumeric characters.
2. Contractions: Preserved as single tokens (e.g., "it's", "don't", "zara's").
   Both ASCII single quote (') and Unicode right quotation mark (’) are normalized to (').
3. Punctuation: Individual punctuation marks (.,!?;:-\"'()[] etc.) are separated as
   independent tokens.
4. Normalization: Alphabetic characters are lowercased by default.
5. Whitespace: Multiple spaces, tabs, and newlines are treated as token delimiters and
   do not generate empty tokens.
6. Unsupported/Special Symbols: Symbols (e.g., @, #, $, %, ★) are preserved as individual
   tokens rather than silently discarded, ensuring lossless token extraction.
"""

from __future__ import annotations

import re
from typing import Final

# Regex pattern:
# 1. \w+(?:['’]\w+)?: Word or contraction (e.g., "milo", "it's", "zara’s")
# 2. [^\w\s]: Any non-word, non-whitespace character (punctuation or symbol)
TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]")


def normalize_token(token: str) -> str:
    """Normalize an individual token string.

    Applies:
    - Stripping surrounding whitespace.
    - Normalizing right single quotation mark ('’') to standard ASCII apostrophe ('\'').
    - Converting alphabetic characters to lowercase.

    Args:
        token: The raw token string.

    Returns:
        The normalized token string.

    Raises:
        TypeError: If token is not an instance of str.
    """
    if not isinstance(token, str):
        raise TypeError(f"Token must be a string, got {type(token).__name__}")
    return token.strip().replace("’", "'").lower()


class StoryTokenizer:
    """Configurable word-level tokenizer for children's stories."""

    def __init__(self, lowercase: bool = True, preserve_contractions: bool = True) -> None:
        """Initialize the tokenizer.

        Args:
            lowercase: Whether to convert tokens to lowercase (default: True).
            preserve_contractions: Whether to keep contractions like "don't" intact
                                   (default: True).
        """
        self.lowercase = lowercase
        self.preserve_contractions = preserve_contractions
        if preserve_contractions:
            self._pattern = TOKEN_PATTERN
        else:
            self._pattern = re.compile(r"\w+|[^\w\s]")

    def tokenize(self, text: str) -> list[str]:
        """Convert input text into a list of normalized tokens.

        Args:
            text: Raw input text.

        Returns:
            List of discrete token strings.

        Raises:
            TypeError: If text is not an instance of str.
        """
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text).__name__}")

        if not text.strip():
            return []

        raw_matches = self._pattern.findall(text)
        tokens: list[str] = []

        for raw in raw_matches:
            cleaned = raw.strip()
            if not cleaned:
                continue

            # Normalize apostrophe
            normalized = cleaned.replace("’", "'")
            if self.lowercase:
                normalized = normalized.lower()

            if normalized:
                tokens.append(normalized)

        return tokens


# Module-level default instance and convenience function
_DEFAULT_TOKENIZER = StoryTokenizer(lowercase=True, preserve_contractions=True)


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Convenience function to tokenize text using the default StoryTokenizer.

    Args:
        text: Input text string.
        lowercase: Whether to lowercase tokens (default: True).

    Returns:
        List of extracted tokens.
    """
    if lowercase:
        return _DEFAULT_TOKENIZER.tokenize(text)
    tokenizer = StoryTokenizer(lowercase=False, preserve_contractions=True)
    return tokenizer.tokenize(text)
