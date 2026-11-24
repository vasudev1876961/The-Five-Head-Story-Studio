"""Registry defining the 8 canonical experimental conditions (E1 to E8)."""

from __future__ import annotations

from typing import Dict, List

from five_head_nlp.experiments.experiment_config import ExperimentConfig
from five_head_nlp.heads.scoring_config import ScoringConfig


def create_e1_baseline_mle() -> ExperimentConfig:
    """E1: Baseline Unsmoothed Trigram MLE with greedy decoding."""
    return ExperimentConfig(
        config_id="E1",
        name="Baseline MLE",
        description="Unsmoothed Trigram Model using Maximum Likelihood Estimation with Greedy Decoding.",
        model_type="trigram",
        smoother_type="none",
        mode="baseline",
        decoder_type="greedy",
    )


def create_e2_laplace() -> ExperimentConfig:
    """E2: Trigram Model with Laplace Smoothing (+1 add-one) and greedy decoding."""
    return ExperimentConfig(
        config_id="E2",
        name="Laplace Smoothing",
        description="Trigram Model with Add-1 Laplace Smoothing with Greedy Decoding.",
        model_type="trigram",
        smoother_type="laplace",
        smoother_params={"alpha": 1.0},
        mode="baseline",
        decoder_type="greedy",
    )


def create_e3_trigram_backoff() -> ExperimentConfig:
    """E3: Trigram Model with Absolute Discounting Backoff and greedy decoding."""
    return ExperimentConfig(
        config_id="E3",
        name="Trigram Backoff",
        description="Trigram Model with Absolute Discounting Backoff Smoothing with Greedy Decoding.",
        model_type="trigram",
        smoother_type="backoff",
        smoother_params={"discount": 0.5},
        mode="baseline",
        decoder_type="greedy",
    )


def create_e4_grammar_only() -> ExperimentConfig:
    """E4: Five-Head Architecture with Grammar Scout only (weight=1.0, others=0.0)."""
    cfg = ScoringConfig(
        grammar_weight=1.0,
        character_weight=0.0,
        setting_weight=0.0,
        action_weight=0.0,
        repeat_weight=0.0,
        enable_grammar=True,
        enable_character=False,
        enable_setting=False,
        enable_action=False,
        enable_repeat=False,
    )
    return ExperimentConfig(
        config_id="E4",
        name="Grammar Only",
        description="Multi-head candidate scoring with Grammar Scout only (w1=1.0), Greedy Decoding.",
        model_type="trigram",
        smoother_type="backoff",
        smoother_params={"discount": 0.5},
        mode="five_head",
        scoring_config=cfg,
        decoder_type="greedy",
    )


def create_e5_thematic_heads() -> ExperimentConfig:
    """E5: Thematic Heads (Grammar + Character + Setting + Action, No Repeat Blocker)."""
    cfg = ScoringConfig(
        grammar_weight=0.50,
        character_weight=0.20,
        setting_weight=0.20,
        action_weight=0.10,
        repeat_weight=0.0,
        enable_grammar=True,
        enable_character=True,
        enable_setting=True,
        enable_action=True,
        enable_repeat=False,
    )
    return ExperimentConfig(
        config_id="E5",
        name="Thematic Heads",
        description="Grammar + Character + Setting + Action (no Repeat Blocker), Greedy Decoding.",
        model_type="trigram",
        smoother_type="backoff",
        smoother_params={"discount": 0.5},
        mode="five_head",
        scoring_config=cfg,
        decoder_type="greedy",
    )


def create_e6_full_five_head() -> ExperimentConfig:
    """E6: Full Five-Head Architecture with standard weights and Top-K (k=5) decoding."""
    cfg = ScoringConfig(
        grammar_weight=0.40,
        character_weight=0.15,
        setting_weight=0.15,
        action_weight=0.10,
        repeat_weight=0.20,
        enable_grammar=True,
        enable_character=True,
        enable_setting=True,
        enable_action=True,
        enable_repeat=True,
    )
    return ExperimentConfig(
        config_id="E6",
        name="Full Five-Head (Top-K)",
        description="Full Five-Head System with balanced weights and Top-K (k=5) stochastic decoding.",
        model_type="trigram",
        smoother_type="backoff",
        smoother_params={"discount": 0.5},
        mode="five_head",
        scoring_config=cfg,
        decoder_type="top_k",
        decoder_params={"k": 5},
    )


def create_e7_full_five_head_greedy() -> ExperimentConfig:
    """E7: Full Five-Head Architecture with standard weights and Greedy decoding."""
    cfg = ScoringConfig(
        grammar_weight=0.40,
        character_weight=0.15,
        setting_weight=0.15,
        action_weight=0.10,
        repeat_weight=0.20,
        enable_grammar=True,
        enable_character=True,
        enable_setting=True,
        enable_action=True,
        enable_repeat=True,
    )
    return ExperimentConfig(
        config_id="E7",
        name="Full Five-Head (Greedy)",
        description="Full Five-Head System with balanced weights and deterministic Greedy Decoding.",
        model_type="trigram",
        smoother_type="backoff",
        smoother_params={"discount": 0.5},
        mode="five_head",
        scoring_config=cfg,
        decoder_type="greedy",
    )


def create_e8_full_five_head_temperature() -> ExperimentConfig:
    """E8: Full Five-Head Architecture with standard weights and Temperature (T=0.8) decoding."""
    cfg = ScoringConfig(
        grammar_weight=0.40,
        character_weight=0.15,
        setting_weight=0.15,
        action_weight=0.10,
        repeat_weight=0.20,
        enable_grammar=True,
        enable_character=True,
        enable_setting=True,
        enable_action=True,
        enable_repeat=True,
    )
    return ExperimentConfig(
        config_id="E8",
        name="Full Five-Head (Temperature)",
        description="Full Five-Head System with balanced weights and Temperature Sampling (T=0.8).",
        model_type="trigram",
        smoother_type="backoff",
        smoother_params={"discount": 0.5},
        mode="five_head",
        scoring_config=cfg,
        decoder_type="temperature",
        decoder_params={"temperature": 0.8},
    )


EXPERIMENT_REGISTRY: Dict[str, ExperimentConfig] = {
    "E1": create_e1_baseline_mle(),
    "E2": create_e2_laplace(),
    "E3": create_e3_trigram_backoff(),
    "E4": create_e4_grammar_only(),
    "E5": create_e5_thematic_heads(),
    "E6": create_e6_full_five_head(),
    "E7": create_e7_full_five_head_greedy(),
    "E8": create_e8_full_five_head_temperature(),
}


def get_experiment_config(config_id: str) -> ExperimentConfig:
    """Retrieve experiment config by ID ('E1' - 'E8')."""
    cid = config_id.upper()
    if cid not in EXPERIMENT_REGISTRY:
        raise KeyError(
            f"Unknown experiment ID: '{config_id}'. Available: {list(EXPERIMENT_REGISTRY.keys())}"
        )
    return EXPERIMENT_REGISTRY[cid]


def list_all_experiments() -> List[ExperimentConfig]:
    """Return all registered experiment configurations in order (E1 to E8)."""
    return [EXPERIMENT_REGISTRY[k] for k in sorted(EXPERIMENT_REGISTRY.keys())]
