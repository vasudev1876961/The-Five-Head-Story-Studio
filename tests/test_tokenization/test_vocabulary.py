"""Pytest test suite for Vocabulary and Frequency Management (Phase 2.3)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from five_head_nlp.tokenization import (
    CorpusLoader,
    Vocabulary,
    export_processed_artifacts,
    is_punctuation_token,
)


class TestVocabularyOrderingAndConstruction:
    """Tests for deterministic ordering and vocabulary construction."""

    def test_deterministic_ordering_with_frequency_and_alphabetical_tie_break(self):
        # Tokens: "cat" x3, "apple" x2, "banana" x2, "zebra" x1, "." x3
        tokens = ["cat", "apple", "banana", "cat", "apple", "banana", "cat", ".", ".", ".", "zebra"]
        vocab = Vocabulary(tokens)

        # Frequencies:
        # cat: 3, .: 3 -> tie broken alphabetically: '.' comes before 'cat'
        # apple: 2, banana: 2 -> tie broken alphabetically: 'apple' comes before 'banana'
        # zebra: 1
        expected_order = [".", "cat", "apple", "banana", "zebra"]
        assert vocab.tokens == expected_order

    def test_build_from_nested_sentence_lists(self):
        sentences = [
            ["milo", "found", "the", "key", "."],
            ["pip", "found", "the", "castle", "."],
        ]
        vocab = Vocabulary(sentences)
        assert len(vocab) == 7
        assert "milo" in vocab
        assert "castle" in vocab
        assert "." in vocab
        assert vocab.get_frequency("found") == 2
        assert vocab.get_frequency("the") == 2
        assert vocab.get_frequency(".") == 2

    def test_empty_vocabulary(self):
        vocab = Vocabulary([])
        assert len(vocab) == 0
        assert vocab.size == 0
        assert vocab.lexical_size == 0
        assert vocab.tokens == []
        assert vocab.get_frequency("anything") == 0
        assert vocab.get_probability("anything") == 0.0


class TestLookupsAndBidirectionalMappings:
    """Tests for token-to-ID and ID-to-token mappings."""

    @pytest.fixture
    def sample_vocab(self) -> Vocabulary:
        return Vocabulary(["the", "silver", "key", ".", "the", "crystal", "key", "."])

    def test_token_to_id_and_reverse(self, sample_vocab: Vocabulary):
        for token_id, token in enumerate(sample_vocab.tokens):
            assert sample_vocab.token_to_id(token) == token_id
            assert sample_vocab.id_to_token(token_id) == token

    def test_unknown_token_lookup_raises_key_error(self, sample_vocab: Vocabulary):
        with pytest.raises(KeyError, match="Token 'dragon' not found"):
            sample_vocab.token_to_id("dragon")

    def test_get_id_with_default(self, sample_vocab: Vocabulary):
        assert sample_vocab.get_id("the") is not None
        assert sample_vocab.get_id("unknown", default=-1) == -1
        assert sample_vocab.get_id("unknown") is None

    def test_invalid_id_lookup_raises_key_error(self, sample_vocab: Vocabulary):
        with pytest.raises(KeyError, match="ID 999 not found"):
            sample_vocab.id_to_token(999)

    def test_get_token_with_default(self, sample_vocab: Vocabulary):
        assert sample_vocab.get_token(0) is not None
        assert sample_vocab.get_token(999, default="<UNK>") == "<UNK>"
        assert sample_vocab.get_token(999) is None

    def test_unigram_probability_math(self):
        # 10 tokens total
        tokens = ["the", "the", "the", "the", "cat", "cat", "key", "key", ".", "."]
        vocab = Vocabulary(tokens)
        assert vocab.get_probability("the") == pytest.approx(0.4)
        assert vocab.get_probability("cat") == pytest.approx(0.2)
        assert vocab.get_probability("key") == pytest.approx(0.2)
        assert vocab.get_probability(".") == pytest.approx(0.2)
        assert vocab.get_probability("unseen") == 0.0


class TestCorpusVocabularyIntegration:
    """Integration tests on data/raw/story_corpus.txt."""

    @pytest.fixture
    def corpus_vocab(self) -> Vocabulary:
        loader = CorpusLoader().load()
        return Vocabulary(loader.get_flat_tokens())

    def test_corpus_vocabulary_size_is_120(self, corpus_vocab: Vocabulary):
        assert len(corpus_vocab) == 120
        assert corpus_vocab.size == 120

    def test_lexical_vocabulary_size_is_119(self, corpus_vocab: Vocabulary):
        assert corpus_vocab.lexical_size == 119
        assert "." not in corpus_vocab.lexical_tokens

    def test_period_included_in_full_vocabulary(self, corpus_vocab: Vocabulary):
        assert "." in corpus_vocab.tokens
        assert corpus_vocab.get_frequency(".") == 34

    def test_most_frequent_token_is_the(self, corpus_vocab: Vocabulary):
        top_token = corpus_vocab.tokens[0]
        assert top_token == "the"
        assert corpus_vocab.get_frequency("the") == 63
        assert corpus_vocab.token_to_id("the") == 0

    def test_all_vocabulary_tokens_are_unique(self, corpus_vocab: Vocabulary):
        assert len(set(corpus_vocab.tokens)) == 120
        assert len(set(corpus_vocab.lexical_tokens)) == 119

    def test_punctuation_classification(self):
        assert is_punctuation_token(".") is True
        assert is_punctuation_token(",") is True
        assert is_punctuation_token("!") is True
        assert is_punctuation_token("?") is True
        assert is_punctuation_token("milo") is False
        assert is_punctuation_token("it's") is False


class TestProcessedArtifactsExport:
    """Tests verifying generated processed artifacts."""

    def test_export_and_inspect_artifacts(self, tmp_path: Path):
        raw_corpus = Path(__file__).resolve().parents[2] / "data" / "raw" / "story_corpus.txt"
        exported = export_processed_artifacts(corpus_path=raw_corpus, output_dir=tmp_path)

        # 1. tokenized_corpus.json
        tok_file = exported["tokenized_corpus"]
        assert tok_file.is_file()
        tok_data = json.loads(tok_file.read_text(encoding="utf-8"))
        assert tok_data["sentence_count"] == 34
        assert tok_data["total_token_count"] == 415
        assert tok_data["word_token_count"] == 381
        assert tok_data["punctuation_token_count"] == 34
        assert len(tok_data["sentences"]) == 34
        assert len(tok_data["flattened_tokens"]) == 415

        # 2. vocabulary.json
        voc_file = exported["vocabulary"]
        assert voc_file.is_file()
        voc_data = json.loads(voc_file.read_text(encoding="utf-8"))
        assert voc_data["vocabulary_size"] == 120
        assert voc_data["lexical_vocabulary_size"] == 119
        assert voc_data["full_vocabulary"][0] == "the"
        assert voc_data["token_to_id"]["the"] == 0

        # 3. word_frequencies.csv
        csv_file = exported["word_frequencies"]
        assert csv_file.is_file()
        with csv_file.open("r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        header = reader[0]
        rows = reader[1:]
        assert header == ["token", "frequency", "token_type"]
        assert len(rows) == 120
        assert rows[0] == ["the", "63", "word"]
        assert rows[1] == [".", "34", "punctuation"]
