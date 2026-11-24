"""Unit and integration tests for model serialization and safe persistence."""

import json
import pytest
from pathlib import Path

from five_head_nlp.models.base_ngram import BaseNGramModel
from five_head_nlp.models.unigram import UnigramModel
from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


@pytest.fixture
def corpus_sentences():
    loader = CorpusLoader()
    return loader.get_sentences()


class TestModelSerialization:
    """Test model persistence, schema validation, and prediction fidelity."""

    def test_unigram_serialization_roundtrip(self, corpus_sentences, tmp_path):
        model = UnigramModel()
        model.fit(corpus_sentences)

        save_path = tmp_path / "unigram_model.json"
        returned_path = model.save(save_path)
        assert returned_path.is_file()

        # Load polymorphically
        loaded_model = BaseNGramModel.load(save_path)
        assert isinstance(loaded_model, UnigramModel)
        assert loaded_model.n == 1
        assert loaded_model.is_fitted
        assert loaded_model.vocabulary_size == model.vocabulary_size
        assert loaded_model.total_tokens == model.total_tokens

        # Check identical predictions
        for token in ["the", "milo", "pip", "castle", ".", "unseen"]:
            assert loaded_model.probability(token) == model.probability(token)

        assert loaded_model.get_candidates() == model.get_candidates()
        assert loaded_model.get_candidate_distribution() == model.get_candidate_distribution()

    def test_bigram_serialization_roundtrip(self, corpus_sentences, tmp_path):
        model = BigramModel()
        model.fit(corpus_sentences)

        save_path = tmp_path / "bigram_model.json"
        model.save(save_path)

        loaded_model = BigramModel.load(save_path)
        assert isinstance(loaded_model, BigramModel)
        assert loaded_model.n == 2

        # Check predictions on key transitions
        test_cases = [
            ("castle", "crystal"),
            ("cave", "dark"),
            ("garden", "secret"),
            ("the", "near"),
            ("milo", "."),
        ]
        for target, ctx in test_cases:
            assert loaded_model.probability(target, context=ctx) == model.probability(target, context=ctx)
            assert loaded_model.get_candidates(context=ctx) == model.get_candidates(context=ctx)

        # Check sequence probability
        seq = ["pip", "flew", "to", "the", "bright", "forest", "."]
        assert loaded_model.sequence_probability(seq) == model.sequence_probability(seq)

    def test_trigram_serialization_roundtrip(self, corpus_sentences, tmp_path):
        model = TrigramModel()
        model.fit(corpus_sentences)

        save_path = tmp_path / "trigram_model.json"
        model.save(save_path)

        loaded_model = BaseNGramModel.load(save_path)
        assert isinstance(loaded_model, TrigramModel)
        assert loaded_model.n == 3

        # Check trigram prediction
        ctx = ("the", "crystal")
        assert loaded_model.probability("castle", context=ctx) == model.probability("castle", context=ctx)
        assert loaded_model.get_candidates(context=ctx) == model.get_candidates(context=ctx)

    def test_subclass_type_check_enforced(self, corpus_sentences, tmp_path):
        model = UnigramModel()
        model.fit(corpus_sentences)
        save_path = tmp_path / "unigram.json"
        model.save(save_path)

        # Attempting to load UnigramModel JSON via BigramModel.load must raise ValueError
        with pytest.raises(ValueError, match="Cannot load UnigramModel into requested class BigramModel"):
            BigramModel.load(save_path)

    def test_save_creates_parent_directories(self, corpus_sentences, tmp_path):
        nested_path = tmp_path / "nested" / "subfolder" / "model.json"
        model = BigramModel()
        model.fit(corpus_sentences)
        model.save(nested_path)
        assert nested_path.is_file()

    def test_missing_file_raises_filenotfound(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            BaseNGramModel.load(missing)

    def test_corrupt_json_raises_value_error(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ corrupt json ...", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupt JSON model file"):
            BaseNGramModel.load(bad_json)

    def test_missing_keys_raises_value_error(self, tmp_path):
        invalid_schema = tmp_path / "invalid.json"
        invalid_schema.write_text(json.dumps({"model_type": "UnigramModel"}), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required keys"):
            BaseNGramModel.load(invalid_schema)

    def test_unknown_model_type_raises_value_error(self, tmp_path):
        unknown_type = tmp_path / "unknown.json"
        unknown_type.write_text(
            json.dumps({
                "model_type": "SuperNeuralTransformerModel",
                "order": 1,
                "engine": {},
                "vocabulary": {},
            }),
            encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Unknown model_type"):
            BaseNGramModel.load(unknown_type)

    def test_serialization_to_experiments_models_directory(self, corpus_sentences):
        """Verify saving directly to experiments/models/ as required by specification."""
        dest_dir = Path("experiments/models")
        dest_dir.mkdir(parents=True, exist_ok=True)

        for ModelCls in [UnigramModel, BigramModel, TrigramModel]:
            m = ModelCls()
            m.fit(corpus_sentences)
            path = dest_dir / f"{m.name.lower()}_corpus.json"
            saved_path = m.save(path)
            assert saved_path.is_file()

            # Verify reload
            loaded = BaseNGramModel.load(saved_path)
            assert loaded.total_tokens == 415
