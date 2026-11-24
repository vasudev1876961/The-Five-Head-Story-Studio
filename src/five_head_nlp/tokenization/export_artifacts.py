"""Export processed corpus artifacts to data/processed/.

Generates:
1. data/processed/tokenized_corpus.json
2. data/processed/vocabulary.json
3. data/processed/word_frequencies.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

# Ensure src is on sys.path if executed directly
src_dir = Path(__file__).resolve().parents[2]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from five_head_nlp.tokenization.corpus_loader import CorpusLoader, is_punctuation_token
from five_head_nlp.tokenization.vocabulary import Vocabulary


def export_processed_artifacts(
    corpus_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Load raw corpus, compute vocabulary, and save processed data artifacts.

    Args:
        corpus_path: Optional path to raw corpus file (defaults to data/raw/story_corpus.txt).
        output_dir: Optional path to output directory (defaults to data/processed/).

    Returns:
        Dict mapping artifact keys to their resolved Path destinations.
    """
    project_root = Path(__file__).resolve().parents[3]
    if corpus_path is None:
        corpus_path = project_root / "data" / "raw" / "story_corpus.txt"
    if output_dir is None:
        output_dir = project_root / "data" / "processed"

    corpus_file = Path(corpus_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load corpus
    loader = CorpusLoader(corpus_file).load()
    sentences = loader.get_sentences()
    flat_tokens = loader.get_flat_tokens()

    # 2. Build vocabulary
    vocab = Vocabulary(flat_tokens)

    # 3. Create data/processed/tokenized_corpus.json
    tokenized_corpus_data = {
        "sentence_count": loader.get_sentence_count(),
        "total_token_count": loader.get_token_count(),
        "word_token_count": loader.get_word_count(),
        "punctuation_token_count": loader.get_punctuation_count(),
        "sentences": sentences,
        "flattened_tokens": flat_tokens,
    }
    tokenized_corpus_path = out_path / "tokenized_corpus.json"
    tokenized_corpus_path.write_text(
        json.dumps(tokenized_corpus_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 4. Create data/processed/vocabulary.json
    vocab_data = vocab.to_dict()
    vocab_path = out_path / "vocabulary.json"
    vocab_path.write_text(
        json.dumps(vocab_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 5. Create data/processed/word_frequencies.csv
    csv_path = out_path / "word_frequencies.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "frequency", "token_type"])
        for token in vocab.tokens:
            freq = vocab.get_frequency(token)
            token_type = "punctuation" if is_punctuation_token(token) else "word"
            writer.writerow([token, freq, token_type])

    return {
        "tokenized_corpus": tokenized_corpus_path,
        "vocabulary": vocab_path,
        "word_frequencies": csv_path,
    }


if __name__ == "__main__":
    paths = export_processed_artifacts()
    print("Exported processed artifacts:")
    for k, p in paths.items():
        print(f"  - {k}: {p}")
