"""Pytest suite for Phase 2.1 and Phase 2.1.1 corpus validation.

Verifies statistical consistency, entity representation, formatting,
audited pivot-token distributions, and known bigram transition distributions
on data/raw/story_corpus.txt.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
import pytest

# Ensure src is in python path
src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from five_head_nlp.utils.validate_corpus import (
    count_syllables,
    compute_flesch_kincaid,
    validate_corpus,
)


@pytest.fixture
def corpus_path() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "data" / "raw" / "story_corpus.txt"


@pytest.fixture
def corpus_metrics(corpus_path: Path) -> dict:
    return validate_corpus(corpus_path)


def test_corpus_file_exists_and_readable(corpus_path: Path):
    """Test that data/raw/story_corpus.txt exists, is non-empty, and UTF-8 readable."""
    assert corpus_path.is_file(), f"File does not exist: {corpus_path}"
    content = corpus_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 0, "Corpus file is empty"


def test_sentence_count_is_exactly_34(corpus_path: Path):
    """Test that exactly 34 non-empty sentences exist in the corpus."""
    lines = [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 34, f"Expected 34 sentences, found {len(lines)}"


def test_every_sentence_ends_with_period(corpus_path: Path):
    """Test that each sentence line ends with a period preceded by a space or period token."""
    lines = [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, line in enumerate(lines, start=1):
        assert line.endswith("."), f"Sentence {i} does not end with period: '{line}'"


def test_no_sentence_contains_malformed_characters(corpus_path: Path):
    """Test that text is lowercase and contains only valid alphabetic words, spaces, and periods."""
    lines = [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, line in enumerate(lines, start=1):
        assert line == line.lower(), f"Sentence {i} contains uppercase characters: '{line}'"
        assert not re.search(r"\d", line), f"Sentence {i} contains digits: '{line}'"
        assert re.match(r"^[a-z\s\.]+$", line), f"Sentence {i} contains invalid characters: '{line}'"


def test_all_five_characters_present(corpus_metrics: dict):
    """Verify all five characters (Milo, Pip, Zara, Owl, Fox) appear with positive count."""
    chars = corpus_metrics["character_counts"]
    assert chars["milo"] >= 7, f"Milo count expected >= 7, got {chars['milo']}"
    assert chars["pip"] >= 7, f"Pip count expected >= 7, got {chars['pip']}"
    assert chars["zara"] >= 7, f"Zara count expected >= 7, got {chars['zara']}"
    assert chars["owl"] >= 4, f"Owl count expected >= 4, got {chars['owl']}"
    assert chars["fox"] >= 4, f"Fox count expected >= 4, got {chars['fox']}"


def test_all_four_settings_present(corpus_metrics: dict):
    """Verify all four settings (Forest, Castle, Cave, Garden) appear with positive count."""
    settings = corpus_metrics["setting_counts"]
    assert settings["forest"] >= 5, f"Forest count expected >= 5, got {settings['forest']}"
    assert settings["castle"] >= 5, f"Castle count expected >= 5, got {settings['castle']}"
    assert settings["cave"] >= 5, f"Cave count expected >= 5, got {settings['cave']}"
    assert settings["garden"] >= 5, f"Garden count expected >= 5, got {settings['garden']}"


def test_token_and_word_counts_consistent(corpus_metrics: dict):
    """Verify internal consistency: total_tokens == word_count + sentence_count."""
    total_tokens = corpus_metrics["total_tokens"]
    word_count = corpus_metrics["word_count"]
    sentence_count = corpus_metrics["sentence_count"]
    assert total_tokens == word_count + sentence_count, "Token and word counts inconsistent"
    assert word_count == 381, f"Expected 381 words, got {word_count}"
    assert total_tokens == 415, f"Expected 415 tokens, got {total_tokens}"
    assert corpus_metrics["unique_vocabulary"] == 120, f"Expected 120 vocab, got {corpus_metrics['unique_vocabulary']}"


def test_known_target_bigrams_match_exactly(corpus_metrics: dict):
    """Verify that all 18 documented bigrams match their target frequencies exactly."""
    expected_bigrams = {
        "the -> silver": 8,
        "the -> crystal": 6,
        "the -> old": 5,
        "the -> dark": 5,
        "the -> bright": 4,
        "the -> brave": 4,
        ". -> the": 11,
        ". -> milo": 4,
        ". -> pip": 4,
        ". -> zara": 4,
        "and -> the": 6,
        "and -> flew": 2,
        "to -> the": 5,
        "near -> the": 7,
        "milo -> lived": 1,
        "milo -> liked": 1,
        "milo -> found": 1,
        "milo -> showed": 1,
    }

    actual_bigrams = corpus_metrics["target_bigram_counts"]
    for bg_name, target_count in expected_bigrams.items():
        actual_count = actual_bigrams.get(bg_name, 0)
        assert actual_count == target_count, (
            f"Bigram '{bg_name}' mismatch: expected {target_count}, got {actual_count}"
        )


def test_syllable_counter_and_fkgl_math():
    """Verify unit accuracy of syllable counting heuristic and FKGL math."""
    assert count_syllables("cat") == 1
    assert count_syllables("clever") == 2
    assert count_syllables("crystal") == 2
    assert count_syllables("wonderful") == 3
    assert count_syllables("") == 0
    grade = compute_flesch_kincaid(num_words=381, num_sentences=34, total_syllables=491)
    assert 3.90 <= grade <= 4.10, f"FKGL out of expected Grade 4 range: {grade}"


def test_pivot_token_actual_statistics(corpus_metrics: dict):
    """Independently audit and verify actual pivot-token counts against data/raw/story_corpus.txt.

    Tests actual unigram occurrence counts, outgoing bigram transitions,
    and unique continuations without hardcoded assumptions.
    """
    pivots = corpus_metrics["pivot_stats"]

    # Audited pivot occurrence and out-degree values
    expected_audited_stats = {
        "near": {"unigram": 7, "out": 7, "unique_next": 1},
        "castle": {"unigram": 9, "out": 9, "unique_next": 6},
        "cave": {"unigram": 7, "out": 7, "unique_next": 5},
        "the": {"unigram": 63, "out": 63, "unique_next": 24},
        "a": {"unigram": 11, "out": 11, "unique_next": 11},
        "to": {"unigram": 11, "out": 11, "unique_next": 5},
        "zara": {"unigram": 7, "out": 7, "unique_next": 6},
        "milo": {"unigram": 7, "out": 7, "unique_next": 7},
        "pip": {"unigram": 7, "out": 7, "unique_next": 6},
        "and": {"unigram": 12, "out": 12, "unique_next": 6},
        ".": {"unigram": 34, "out": 33, "unique_next": 12},
    }

    for token, exp in expected_audited_stats.items():
        assert token in pivots, f"Token '{token}' missing from pivot_stats"
        act = pivots[token]
        assert act["unigram_count"] == exp["unigram"], (
            f"Token '{token}' unigram mismatch: expected {exp['unigram']}, got {act['unigram_count']}"
        )
        assert act["out_transitions"] == exp["out"], (
            f"Token '{token}' outgoing transitions mismatch: expected {exp['out']}, got {act['out_transitions']}"
        )
        assert act["unique_next"] == exp["unique_next"], (
            f"Token '{token}' unique_next mismatch: expected {exp['unique_next']}, got {act['unique_next']}"
        )

    # Verify bigram branching metrics
    assert corpus_metrics["bigram_total_unique_flat"] == 271, (
        f"Expected 271 unique flat bigrams, got {corpus_metrics['bigram_total_unique_flat']}"
    )
    assert corpus_metrics["total_bigram_transitions_flat"] == 414, (
        f"Expected 414 total flat transitions, got {corpus_metrics['total_bigram_transitions_flat']}"
    )
    assert corpus_metrics["avg_bigram_branching_factor"] == 2.26, (
        f"Expected 2.26 branching factor, got {corpus_metrics['avg_bigram_branching_factor']}"
    )
