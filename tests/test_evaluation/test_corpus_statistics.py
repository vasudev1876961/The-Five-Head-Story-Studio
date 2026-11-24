"""Pytest test suite for CorpusStatistics (Phase 2.4)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from five_head_nlp.tokenization import CorpusLoader, Vocabulary
from five_head_nlp.evaluation import CorpusStatistics


class TestCorpusStatisticsSynthetic:
    """Mathematical and logical correctness tests on small controlled synthetic sentences."""

    @pytest.fixture
    def synthetic_sentences(self) -> list[list[str]]:
        # 3 sentences
        # S1: "the", "cat", "saw", "milo", "." (5 tokens)
        # S2: "milo", "liked", "the", "cat", "." (5 tokens)
        # S3: "the", "cat", "jumped", "." (4 tokens)
        return [
            ["the", "cat", "saw", "milo", "."],
            ["milo", "liked", "the", "cat", "."],
            ["the", "cat", "jumped", "."],
        ]

    @pytest.fixture
    def synthetic_stats(self, synthetic_sentences: list[list[str]]) -> CorpusStatistics:
        return CorpusStatistics(synthetic_sentences)

    def test_basic_dimensions(self, synthetic_stats: CorpusStatistics):
        basic = synthetic_stats.get_basic_statistics()
        assert basic["num_sentences"] == 3
        # Total tokens = 5 + 5 + 4 = 14
        assert basic["total_tokens"] == 14
        # Punctuation: 3 periods '.'
        assert basic["punctuation_tokens"] == 3
        # Words: 14 - 3 = 11
        assert basic["word_tokens"] == 11
        # Sentence lengths: [5, 5, 4] -> min=4, max=5, avg=14/3 = 4.67, median=5.0
        assert basic["min_sentence_length"] == 4
        assert basic["max_sentence_length"] == 5
        assert basic["avg_sentence_length"] == pytest.approx(4.67, rel=1e-2)
        assert basic["median_sentence_length"] == 5.0

    def test_vocabulary_and_hapax(self, synthetic_stats: CorpusStatistics):
        # Tokens: "the" (3), "cat" (3), "." (3), "milo" (2), "saw" (1), "liked" (1), "jumped" (1)
        # Unique tokens = 7 (6 words + 1 period)
        assert synthetic_stats.vocabulary.size == 7
        assert synthetic_stats.vocabulary.lexical_size == 6

        wf = synthetic_stats.get_word_frequency_analysis()
        # Hapax legomena: "saw", "liked", "jumped" -> count = 3
        assert wf["hapax_legomena_count"] == 3
        assert sorted(wf["hapax_legomena"]) == ["jumped", "liked", "saw"]
        # Hapax ratio = 3 / 6 = 0.5
        assert wf["hapax_legomena_ratio"] == 0.5

    def test_within_sentence_bigrams_strictly_enforced(self, synthetic_stats: CorpusStatistics):
        bg_stats = synthetic_stats.get_bigram_statistics()
        # Transitions within sentences:
        # S1 (len 5): 4 transitions: ("the", "cat"), ("cat", "saw"), ("saw", "milo"), ("milo", ".")
        # S2 (len 5): 4 transitions: ("milo", "liked"), ("liked", "the"), ("the", "cat"), ("cat", ".")
        # S3 (len 4): 3 transitions: ("the", "cat"), ("cat", "jumped"), ("jumped", ".")
        # Total transitions = 4 + 4 + 3 = 11
        assert bg_stats["total_bigram_transitions"] == 11

        # Check that '.' NEVER has an outgoing transition
        conts = bg_stats["continuations_per_token"]
        assert conts["."] == 0

        # Unique bigrams:
        # ("the", "cat") x3
        # ("cat", "saw") x1
        # ("saw", "milo") x1
        # ("milo", ".") x1
        # ("milo", "liked") x1
        # ("liked", "the") x1
        # ("cat", ".") x1
        # ("cat", "jumped") x1
        # ("jumped", ".") x1
        # Total unique = 9
        assert bg_stats["unique_bigrams_count"] == 9

        # Branching factor:
        # Sum of out-degrees = 9. Full vocab size = 7. Active tokens = 6 (excluding '.')
        # B_full = 9 / 7 = 1.2857
        # B_active = 9 / 6 = 1.5
        assert bg_stats["average_branching_factor_full_vocab"] == pytest.approx(9 / 7, abs=1e-3)
        assert bg_stats["average_branching_factor_active_vocab"] == pytest.approx(1.5, abs=1e-3)


class TestCorpusStatisticsRealCorpus:
    """Statistical verification on the reconstructed 34-sentence story corpus."""

    @pytest.fixture
    def corpus_stats(self) -> CorpusStatistics:
        loader = CorpusLoader().load()
        return CorpusStatistics(loader)

    def test_real_corpus_basic_statistics(self, corpus_stats: CorpusStatistics):
        basic = corpus_stats.get_basic_statistics()
        assert basic["num_sentences"] == 34
        assert basic["total_tokens"] == 415
        assert basic["word_tokens"] == 381
        assert basic["punctuation_tokens"] == 34
        assert basic["unique_full_vocabulary"] == 120
        assert basic["unique_lexical_vocabulary"] == 119
        assert basic["min_sentence_length"] == 10
        assert basic["max_sentence_length"] == 14
        assert basic["median_sentence_length"] == 12.0
        assert basic["avg_sentence_length"] == 12.21

    def test_real_corpus_bigram_counts(self, corpus_stats: CorpusStatistics):
        bgs = corpus_stats.get_bigram_statistics()
        # Sum of (L_i - 1) for 34 sentences = 415 - 34 = 381 transitions
        assert bgs["total_bigram_transitions"] == 381
        # Within-sentence unique bigrams = 259
        assert bgs["unique_bigrams_count"] == 259
        # Period has 0 outgoing transitions within sentences
        assert bgs["continuations_per_token"]["."] == 0
        # Active tokens count = 119
        assert bgs["active_tokens_count"] == 119
        assert bgs["average_branching_factor_full_vocab"] == 2.1583
        assert bgs["average_branching_factor_active_vocab"] == 2.1765

    def test_real_corpus_hapax_legomena(self, corpus_stats: CorpusStatistics):
        wf = corpus_stats.get_word_frequency_analysis()
        assert wf["hapax_legomena_count"] == 57
        assert wf["hapax_legomena_ratio"] == round(57 / 119, 4)
        assert len(wf["top_20_lexical_words"]) == 20
        assert wf["top_20_lexical_words"][0]["word"] == "the"
        assert wf["top_20_lexical_words"][0]["frequency"] == 63

    def test_real_corpus_pivot_statistics(self, corpus_stats: CorpusStatistics):
        pivots = {p["token"]: p for p in corpus_stats.get_pivot_token_analysis()}

        # Verify key tokens
        assert pivots["."]["unigram_frequency"] == 34
        assert pivots["."]["outgoing_transitions"] == 0
        assert pivots["."]["unique_next_tokens"] == 0

        assert pivots["the"]["unigram_frequency"] == 63
        assert pivots["the"]["outgoing_transitions"] == 63
        assert pivots["the"]["unique_next_tokens"] == 24

        assert pivots["near"]["unigram_frequency"] == 7
        assert pivots["near"]["outgoing_transitions"] == 7
        assert pivots["near"]["unique_next_tokens"] == 1

        assert pivots["castle"]["unigram_frequency"] == 9
        assert pivots["castle"]["outgoing_transitions"] == 9
        assert pivots["castle"]["unique_next_tokens"] == 6

        assert pivots["cave"]["unigram_frequency"] == 7
        assert pivots["cave"]["outgoing_transitions"] == 7
        assert pivots["cave"]["unique_next_tokens"] == 5

        assert pivots["milo"]["unigram_frequency"] == 7
        assert pivots["milo"]["outgoing_transitions"] == 7
        assert pivots["milo"]["unique_next_tokens"] == 7

        assert pivots["pip"]["unigram_frequency"] == 7
        assert pivots["pip"]["outgoing_transitions"] == 7
        assert pivots["pip"]["unique_next_tokens"] == 6

        assert pivots["zara"]["unigram_frequency"] == 7
        assert pivots["zara"]["outgoing_transitions"] == 7
        assert pivots["zara"]["unique_next_tokens"] == 6

    def test_real_corpus_characters_and_settings(self, corpus_stats: CorpusStatistics):
        entities = corpus_stats.get_entity_analysis()
        chars = entities["characters"]
        assert chars["Milo"] == 7
        assert chars["Pip"] == 7
        assert chars["Zara"] == 7
        assert chars["Sleepy Owl"] == 5
        assert chars["Playful Fox"] == 4

        phrases = entities["setting_phrases"]
        assert phrases["Crystal Castle"] == 6
        assert phrases["Dark Cave"] == 6
        assert phrases["Secret Garden"] == 5
        assert phrases["Bright Forest"] == 3

    def test_artifact_export_roundtrip(self, corpus_stats: CorpusStatistics, tmp_path: Path):
        saved = corpus_stats.save_artifacts(tmp_path)
        json_file = saved["statistics_json"]
        assert json_file.is_file()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert data["basic_statistics"]["num_sentences"] == 34
        assert data["basic_statistics"]["total_tokens"] == 415

        bigram_csv = saved["bigram_csv"]
        assert bigram_csv.is_file()
        pivot_csv = saved["pivot_csv"]
        assert pivot_csv.is_file()
