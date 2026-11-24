"""Corpus loading and sentence boundary preservation module.

Provides the CorpusLoader class for reading, tokenizing, and extracting
sentence-structured and flattened token sequences from children's story corpora.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from five_head_nlp.tokenization.tokenizer import StoryTokenizer


def is_punctuation_token(token: str) -> bool:
    """Check if a token consists entirely of punctuation or non-alphanumeric symbols.

    Words and contractions (e.g., 'milo', 'it\'s', 'café') contain alphanumeric
    characters. Punctuation marks (e.g., '.', ',', '!', '?') do not.

    Args:
        token: The token string to check.

    Returns:
        True if the token contains no alphanumeric characters, False otherwise.
    """
    return not any(c.isalnum() for c in token)


class CorpusLoader:
    """Loads, validates, and tokenizes children's story corpora."""

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        tokenizer: StoryTokenizer | None = None,
    ) -> None:
        """Initialize the corpus loader.

        Args:
            corpus_path: Path to the raw story corpus text file. If None,
                         defaults to 'data/raw/story_corpus.txt' relative
                         to the project root.
            tokenizer: Optional StoryTokenizer instance. If None, a default
                       StoryTokenizer(lowercase=True) is instantiated.
        """
        if corpus_path is None:
            # Default to project root data/raw/story_corpus.txt
            corpus_path = (
                Path(__file__).resolve().parents[3] / "data" / "raw" / "story_corpus.txt"
            )
        self.corpus_path = Path(corpus_path)
        self.tokenizer = tokenizer if tokenizer is not None else StoryTokenizer(lowercase=True)

        self._sentences: list[list[str]] = []
        self._flat_tokens: list[str] = []
        self._is_loaded: bool = False

    def load(self) -> CorpusLoader:
        """Read and tokenize the corpus from disk.

        Returns:
            The loaded CorpusLoader instance (for method chaining).

        Raises:
            FileNotFoundError: If corpus_path does not exist or is not a file.
            ValueError: If the corpus file contains no non-empty lines or valid sentences.
        """
        if not self.corpus_path.is_file():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        try:
            content = self.corpus_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise UnicodeDecodeError(
                exc.encoding,
                exc.object,
                exc.start,
                exc.end,
                f"Corpus file '{self.corpus_path}' must be valid UTF-8 text.",
            ) from exc

        raw_lines = [line.strip() for line in content.splitlines()]
        non_empty_lines = [line for line in raw_lines if line]

        if not non_empty_lines:
            raise ValueError(f"Corpus file '{self.corpus_path}' contains no valid sentences.")

        tokenized_sentences: list[list[str]] = []
        flat_tokens: list[str] = []

        for line in non_empty_lines:
            sent_tokens = self.tokenizer.tokenize(line)
            if sent_tokens:
                tokenized_sentences.append(sent_tokens)
                flat_tokens.extend(sent_tokens)

        if not tokenized_sentences or not flat_tokens:
            raise ValueError(
                f"Corpus file '{self.corpus_path}' produced no valid tokens after tokenization."
            )

        self._sentences = tokenized_sentences
        self._flat_tokens = flat_tokens
        self._is_loaded = True
        return self

    def _ensure_loaded(self) -> None:
        """Ensure the corpus has been loaded before accessing data."""
        if not self._is_loaded:
            self.load()

    def get_sentences(self) -> list[list[str]]:
        """Return the list of tokenized sentences, preserving sentence boundaries.

        Returns:
            A list where each element is a list of token strings for one sentence.
        """
        self._ensure_loaded()
        return [list(sentence) for sentence in self._sentences]

    def get_flat_tokens(self) -> list[str]:
        """Return all tokens in the corpus as a single flattened sequence.

        Returns:
            A continuous list of tokens across all sentences.
        """
        self._ensure_loaded()
        return list(self._flat_tokens)

    def get_sentence_count(self) -> int:
        """Return the number of sentences in the loaded corpus."""
        self._ensure_loaded()
        return len(self._sentences)

    def get_token_count(self) -> int:
        """Return the total number of tokens (words + punctuation)."""
        self._ensure_loaded()
        return len(self._flat_tokens)

    def get_word_count(self) -> int:
        """Return the number of lexical word tokens (excluding punctuation)."""
        self._ensure_loaded()
        return sum(1 for token in self._flat_tokens if not is_punctuation_token(token))

    def get_punctuation_count(self) -> int:
        """Return the number of punctuation tokens in the corpus."""
        self._ensure_loaded()
        return sum(1 for token in self._flat_tokens if is_punctuation_token(token))

    def get_metadata(self) -> dict[str, Any]:
        """Return high-level corpus descriptive statistics.

        Returns:
            Dictionary containing sentence count, token counts, and averages.
        """
        self._ensure_loaded()
        sent_count = self.get_sentence_count()
        word_count = self.get_word_count()
        total_tokens = self.get_token_count()
        punct_count = self.get_punctuation_count()

        return {
            "corpus_path": str(self.corpus_path),
            "sentence_count": sent_count,
            "total_token_count": total_tokens,
            "word_token_count": word_count,
            "punctuation_token_count": punct_count,
            "avg_words_per_sentence": round(word_count / sent_count, 2) if sent_count else 0.0,
            "avg_tokens_per_sentence": round(total_tokens / sent_count, 2) if sent_count else 0.0,
        }
