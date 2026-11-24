"""Pytest test suite for word-level StoryTokenizer (Phase 2.2)."""

from __future__ import annotations

from pathlib import Path
import pytest

from five_head_nlp.tokenization import StoryTokenizer, normalize_token, tokenize


class TestBasicTokenization:
    """Tests covering fundamental word and punctuation splitting."""

    def test_basic_sentence(self):
        text = "Milo found the silver key."
        expected = ["milo", "found", "the", "silver", "key", "."]
        assert tokenize(text) == expected

    def test_comma_and_exclamation(self):
        text = "Pip flew to the bright forest, and Zara smiled!"
        expected = [
            "pip",
            "flew",
            "to",
            "the",
            "bright",
            "forest",
            ",",
            "and",
            "zara",
            "smiled",
            "!",
        ]
        assert tokenize(text) == expected

    def test_question_mark_and_colon(self):
        text = "Where is the key? Look here: inside the box."
        expected = [
            "where",
            "is",
            "the",
            "key",
            "?",
            "look",
            "here",
            ":",
            "inside",
            "the",
            "box",
            ".",
        ]
        assert tokenize(text) == expected

    def test_semicolon_and_parentheses(self):
        text = "The cat slept; the dragon (Pip) watched."
        expected = [
            "the",
            "cat",
            "slept",
            ";",
            "the",
            "dragon",
            "(",
            "pip",
            ")",
            "watched",
            ".",
        ]
        assert tokenize(text) == expected


class TestNormalizationAndWhitespace:
    """Tests covering casing, whitespace, and empty inputs."""

    def test_lowercase_normalization(self):
        text = "ZARA Liked The CRYSTAL Castle."
        tokens = tokenize(text)
        assert tokens == ["zara", "liked", "the", "crystal", "castle", "."]
        assert all(t == t.lower() for t in tokens)

    def test_multiple_spaces(self):
        text = "milo    found    the     key   ."
        expected = ["milo", "found", "the", "key", "."]
        assert tokenize(text) == expected

    def test_newlines_and_tabs(self):
        text = "milo lived near\n\nthe crystal castle\t.\npip flew."
        expected = [
            "milo",
            "lived",
            "near",
            "the",
            "crystal",
            "castle",
            ".",
            "pip",
            "flew",
            ".",
        ]
        assert tokenize(text) == expected

    def test_empty_string(self):
        assert tokenize("") == []

    def test_whitespace_only(self):
        assert tokenize("     \n\t   ") == []

    def test_no_empty_tokens_generated(self):
        text = "  the   silver   key   .  "
        tokens = tokenize(text)
        assert all(len(t) > 0 for t in tokens)
        assert "" not in tokens


class TestContractionsAndSpecialCharacters:
    """Tests covering contractions, apostrophes, and unsupported symbols."""

    def test_ascii_contractions(self):
        text = "It's Zara's wand, and Milo couldn't jump."
        expected = ["it's", "zara's", "wand", ",", "and", "milo", "couldn't", "jump", "."]
        assert tokenize(text) == expected

    def test_unicode_curly_apostrophe(self):
        text = "They didn’t see Zara’s wand."
        expected = ["they", "didn't", "see", "zara's", "wand", "."]
        assert tokenize(text) == expected

    def test_symbols_preserved_as_individual_tokens(self):
        # Policy: special symbols are not silently dropped
        text = "Milo found #1 key for $5 @ noon!"
        expected = ["milo", "found", "#", "1", "key", "for", "$", "5", "@", "noon", "!"]
        assert tokenize(text) == expected

    def test_unicode_accented_words(self):
        text = "The café was in the forest."
        expected = ["the", "café", "was", "in", "the", "forest", "."]
        assert tokenize(text) == expected


class TestErrorHandlingAndAPI:
    """Tests for type checking and component interfaces."""

    def test_non_string_input_raises_type_error(self):
        with pytest.raises(TypeError, match="Input text must be a string"):
            tokenize(None)  # type: ignore

        with pytest.raises(TypeError, match="Input text must be a string"):
            tokenize(12345)  # type: ignore

    def test_normalize_token_standalone(self):
        assert normalize_token("  MILO  ") == "milo"
        assert normalize_token("Zara’s") == "zara's"
        assert normalize_token(".") == "."

        with pytest.raises(TypeError, match="Token must be a string"):
            normalize_token(42)  # type: ignore

    def test_class_configuration_preserve_case(self):
        tok = StoryTokenizer(lowercase=False)
        assert tok.tokenize("Milo Found Key.") == ["Milo", "Found", "Key", "."]


class TestCorpusIntegration:
    """Integration test validating tokenizer behavior against data/raw/story_corpus.txt."""

    @pytest.fixture
    def corpus_text(self) -> str:
        corpus_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "story_corpus.txt"
        assert corpus_path.is_file(), f"Missing corpus file at {corpus_path}"
        return corpus_path.read_text(encoding="utf-8")

    def test_tokenize_entire_corpus(self, corpus_text: str):
        tokens = tokenize(corpus_text)

        # 1. Total tokens must dynamically match corpus calculation (415)
        assert len(tokens) == 415, f"Expected 415 tokens, got {len(tokens)}"

        # 2. Exactly 34 sentence-ending period tokens
        period_tokens = [t for t in tokens if t == "."]
        assert len(period_tokens) == 34, f"Expected 34 period tokens, got {len(period_tokens)}"

        # 3. Word tokens must be 381
        word_tokens = [t for t in tokens if t != "."]
        assert len(word_tokens) == 381, f"Expected 381 word tokens, got {len(word_tokens)}"

        # 4. No empty tokens
        assert all(len(t) > 0 for t in tokens)

        # 5. All tokens lowercased
        assert all(t == t.lower() for t in tokens)

        # 6. Verify first sentence token sequence
        first_sentence = tokens[:10]
        expected_first = [
            "the",
            "clever",
            "cat",
            "milo",
            "lived",
            "near",
            "the",
            "crystal",
            "castle",
            ".",
        ]
        assert first_sentence == expected_first

        # 7. Verify last sentence token sequence
        last_sentence = tokens[-11:]
        expected_last = [
            "there",
            "the",
            "brave",
            "friends",
            "lived",
            "happily",
            "in",
            "a",
            "peaceful",
            "kingdom",
            ".",
        ]
        assert last_sentence == expected_last
