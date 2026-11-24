"""Configuration module for multi-head candidate scoring and lexicons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class CharacterConfig:
    """Configuration for Head 2 — Character Keeper."""

    # Primary named characters from corpus
    named_characters: Set[str] = field(
        default_factory=lambda: {"milo", "pip", "zara", "owl", "fox"}
    )
    # Character nouns appearing in corpus
    character_nouns: Set[str] = field(
        default_factory=lambda: {"cat", "dragon", "wizard", "friends", "creature", "birds"}
    )
    continuity_bonus: float = 0.85
    introduction_bonus: float = 0.50
    non_character_score: float = 0.20
    immediate_character_repeat_penalty: float = 0.10


@dataclass
class SettingConfig:
    """Configuration for Head 3 — Setting Painter."""

    # Core settings established in corpus
    core_settings: Set[str] = field(
        default_factory=lambda: {"castle", "cave", "forest", "garden"}
    )
    # Secondary environmental and architectural setting nouns from corpus
    setting_nouns: Set[str] = field(
        default_factory=lambda: {
            "tree",
            "trees",
            "lake",
            "kingdom",
            "tower",
            "path",
            "sky",
            "stars",
            "floor",
            "walls",
            "door",
        }
    )
    consistency_bonus: float = 0.85
    shift_bonus: float = 0.40
    non_setting_score: float = 0.20


@dataclass
class ActionConfig:
    """Configuration for Head 4 — Action Booster."""

    # Dynamic action verbs from corpus
    dynamic_actions: Set[str] = field(
        default_factory=lambda: {
            "flew",
            "ran",
            "walked",
            "running",
            "blew",
            "found",
            "showed",
            "opened",
            "made",
            "help",
            "helped",
            "held",
            "shared",
            "play",
            "smiled",
        }
    )
    # Stative, perceptual, and descriptive verbs from corpus
    state_verbs: Set[str] = field(
        default_factory=lambda: {
            "saw",
            "watched",
            "read",
            "sang",
            "rested",
            "woke",
            "liked",
            "lived",
            "smelled",
        }
    )
    dynamic_action_bonus: float = 0.90
    state_action_bonus: float = 0.50
    non_action_score: float = 0.20


@dataclass
class RepeatConfig:
    """Configuration for Head 5 — Repeat Blocker."""

    history_window: int = 8
    decay_gamma: float = 0.80
    immediate_repeat_score: float = 0.05
    bigram_repeat_score: float = 0.20
    trigram_repeat_score: float = 0.10
    # Function words and punctuation exempt from repetition penalty
    exempt_tokens: Set[str] = field(
        default_factory=lambda: {
            "the",
            "a",
            "an",
            "and",
            "in",
            "to",
            "near",
            "under",
            "with",
            "for",
            "from",
            "into",
            "across",
            "around",
            "above",
            "was",
            "had",
            "is",
            ".",
            ",",
        }
    )


@dataclass
class ScoringConfig:
    """Global configuration orchestrating all five heads, weights, and toggles."""

    # Relative head weights (normalized during fusion)
    grammar_weight: float = 0.40
    character_weight: float = 0.15
    setting_weight: float = 0.15
    action_weight: float = 0.10
    repeat_weight: float = 0.20

    # Individual head enabling toggles
    enable_grammar: bool = True
    enable_character: bool = True
    enable_setting: bool = True
    enable_action: bool = True
    enable_repeat: bool = True

    # Component configurations
    character: CharacterConfig = field(default_factory=CharacterConfig)
    setting: SettingConfig = field(default_factory=SettingConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    repeat: RepeatConfig = field(default_factory=RepeatConfig)

    def validate(self) -> None:
        """Validate configuration weights and ranges."""
        weights = [
            self.grammar_weight,
            self.character_weight,
            self.setting_weight,
            self.action_weight,
            self.repeat_weight,
        ]
        for w in weights:
            if w < 0.0:
                raise ValueError(f"Head weights must be non-negative, got {w}")
        if sum(weights) <= 0.0:
            raise ValueError(f"Total head weight sum must be positive, got {sum(weights)}")

    def get_weight_dict(self) -> Dict[str, float]:
        """Return dictionary of configured head weights."""
        return {
            "Grammar Scout": self.grammar_weight if self.enable_grammar else 0.0,
            "Character Keeper": self.character_weight if self.enable_character else 0.0,
            "Setting Painter": self.setting_weight if self.enable_setting else 0.0,
            "Action Booster": self.action_weight if self.enable_action else 0.0,
            "Repeat Blocker": self.repeat_weight if self.enable_repeat else 0.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "weights": {
                "grammar": self.grammar_weight,
                "character": self.character_weight,
                "setting": self.setting_weight,
                "action": self.action_weight,
                "repeat": self.repeat_weight,
            },
            "enabled": {
                "grammar": self.enable_grammar,
                "character": self.enable_character,
                "setting": self.enable_setting,
                "action": self.enable_action,
                "repeat": self.enable_repeat,
            },
            "character_lexicon": list(self.character.named_characters.union(self.character.character_nouns)),
            "setting_lexicon": list(self.setting.core_settings.union(self.setting.setting_nouns)),
            "action_lexicon": list(self.action.dynamic_actions.union(self.action.state_verbs)),
            "repeat_window": self.repeat.history_window,
            "repeat_gamma": self.repeat.decay_gamma,
        }
