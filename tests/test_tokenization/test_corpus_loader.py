"""Pytest test suite for CorpusLoader (Phase 2.3)."""

from __future__ import annotations

from pathlib import Path
import pytest

from five_head_nlp.tokenization import CorpusLoader


@pytest.fixture
def corpus_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "story_corpus.txt"


@pytest.fixture
def loaded_corpus(corpus_path: Path) -> CorpusLoader:
    loader = CorpusLoader(corpus_path)
    return loader.load()


class TestCorpusLoaderCore:
    """Core functionality tests for loading and tokenizing the corpus."""

    def test_loads_existing_corpus(self, loaded_corpus: CorpusLoader):
        assert loaded_corpus.get_sentence_count() > 0
        assert loaded_corpus.get_token_count() > 0

    def test_reads_exactly_34_sentences(self, loaded_corpus: CorpusLoader):
        assert loaded_corpus.get_sentence_count() == 34
        sentences = loaded_corpus.get_sentences()
        assert len(sentences) == 34
        assert all(len(s) > 0 for s in sentences)

    def test_preserves_sentence_boundaries(self, loaded_corpus: CorpusLoader):
        sentences = loaded_corpus.get_sentences()
        # Verify every sentence ends with a period token '.'
        for idx, sentence in enumerate(sentences, 1):
            assert sentence[-1] == ".", f"Sentence {idx} does not end with period: {sentence}"

    def test_total_token_count_is_415(self, loaded_corpus: CorpusLoader):
        assert loaded_corpus.get_token_count() == 415

    def test_word_count_is_381(self, loaded_corpus: CorpusLoader):
        assert loaded_corpus.get_word_count() == 381

    def test_preserves_34_periods(self, loaded_corpus: CorpusLoader):
        assert loaded_corpus.get_punctuation_count() == 34
        periods = [t for t in loaded_corpus.get_flat_tokens() if t == "."]
        assert len(periods) == 34

    def test_flattened_sequence_matches_concatenation(self, loaded_corpus: CorpusLoader):
        flat = loaded_corpus.get_flat_tokens()
        concatenated: list[str] = []
        for s in loaded_corpus.get_sentences():
            concatenated.extend(s)
        assert flat == concatenated
        assert len(flat) == 415

    def test_metadata_summary(self, loaded_corpus: CorpusLoader):
        meta = loaded_corpus.get_metadata()
        assert meta["sentence_count"] == 34
        assert meta["total_token_count"] == 415
        assert meta["word_token_count"] == 381
        assert meta["punctuation_token_count"] == 34
        assert meta["avg_words_per_sentence"] == 11.21
        assert meta["avg_tokens_per_sentence"] == 12.21


class TestCorpusLoaderErrorHandling:
    """Error handling and boundary condition tests."""

    def test_missing_file_raises_filenotfound(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.txt"
        loader = CorpusLoader(missing)
        with pytest.raises(FileNotFoundError, match="Corpus file not found"):
            loader.load()

    def test_empty_file_raises_value_error(self, tmp_path: Path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        loader = CorpusLoader(empty_file)
        with pytest.raises(ValueError, match="contains no valid sentences"):
            loader.load()

    def test_whitespace_only_file_raises_value_error(self, tmp_path: Path):
        ws_file = tmp_path / "whitespace.txt"
        ws_file.write_text("   \n\n\t   \n", encoding="utf-8")
        loader = CorpusLoader(ws_file)
        with pytest.raises(ValueError, match="contains no valid sentences"):
            loader.load()
