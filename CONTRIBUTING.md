# Contributing to The Five-Head Story-Generation System

Thank you for your interest in contributing to the Five-Head Story-Generation System project.

## Development Setup

1. Clone or download the repository.
2. Create and activate a Python 3.10+ virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
   ```
3. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Guidelines

- **Classical NLP & Transparent Architecture:** The project relies on classical probabilistic and statistical NLP methods (N-grams, smoothing, linear score fusion, Hamilton apportionment). Deep neural networks (PyTorch, Transformers, LLMs) are strictly out of scope.
- **Corpus Immutability:** The canonical 34-sentence children's story corpus in `data/raw/story_corpus.txt` is an immutable evaluation baseline.
- **Test-Driven Changes:** All bug fixes and enhancements must be accompanied by automated unit or integration tests in `tests/`.
- **Code Style:** Type annotations (`from __future__ import annotations`), descriptive naming, and clean docstrings are mandatory across all modules.

## Running Tests

Run the complete test suite:
```bash
pytest tests -v
```

Run specific test sub-suites:
```bash
pytest tests/test_heads -v
pytest tests/test_smoothing -v
pytest tests/test_decoding -v
pytest tests/test_ui -v
```
