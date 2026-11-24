"""Unit tests for experiment configuration and registry."""

import pytest

from five_head_nlp.experiments.experiment_config import ExperimentConfig
from five_head_nlp.experiments.experiment_registry import (
    EXPERIMENT_REGISTRY,
    get_experiment_config,
    list_all_experiments,
)


class TestExperimentConfig:
    """Test experiment configuration dataclasses and registry validation."""

    def test_registry_contains_all_eight_conditions(self):
        expected_ids = {"E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"}
        assert set(EXPERIMENT_REGISTRY.keys()) == expected_ids

    def test_get_experiment_config_case_insensitive(self):
        cfg_upper = get_experiment_config("E1")
        cfg_lower = get_experiment_config("e1")
        assert cfg_upper.config_id == "E1"
        assert cfg_lower.config_id == "E1"
        assert cfg_upper.name == "Baseline MLE"

    def test_get_experiment_config_invalid_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown experiment ID"):
            get_experiment_config("E99")

    def test_list_all_experiments_returns_sorted_eight(self):
        all_cfgs = list_all_experiments()
        assert len(all_cfgs) == 8
        assert [c.config_id for c in all_cfgs] == [f"E{i}" for i in range(1, 9)]

    def test_e4_grammar_only_weights(self):
        cfg = get_experiment_config("E4")
        assert cfg.mode == "five_head"
        assert cfg.scoring_config is not None
        assert cfg.scoring_config.grammar_weight == 1.0
        assert cfg.scoring_config.character_weight == 0.0
        assert cfg.scoring_config.repeat_weight == 0.0

    def test_e5_thematic_heads_weights(self):
        cfg = get_experiment_config("E5")
        assert cfg.mode == "five_head"
        assert cfg.scoring_config is not None
        assert cfg.scoring_config.grammar_weight == 0.50
        assert cfg.scoring_config.character_weight == 0.20
        assert cfg.scoring_config.repeat_weight == 0.00
        assert not cfg.scoring_config.enable_repeat

    def test_e6_e7_e8_decoders_and_weights(self):
        e6 = get_experiment_config("E6")
        e7 = get_experiment_config("E7")
        e8 = get_experiment_config("E8")

        assert e6.decoder_type == "top_k"
        assert e6.decoder_params["k"] == 5
        assert e7.decoder_type == "greedy"
        assert e8.decoder_type == "temperature"
        assert e8.decoder_params["temperature"] == 0.8

        for cfg in (e6, e7, e8):
            assert cfg.scoring_config.repeat_weight == 0.20
            assert cfg.scoring_config.enable_repeat is True

    def test_to_dict_serialization(self):
        cfg = get_experiment_config("E6")
        d = cfg.to_dict()
        assert d["config_id"] == "E6"
        assert d["decoder_type"] == "top_k"
        assert isinstance(d["scoring_config"], dict)
