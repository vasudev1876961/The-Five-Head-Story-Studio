"""Head 3 — Setting Painter: Maintains setting consistency and environment tracking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from five_head_nlp.heads.base_head import BaseHead
from five_head_nlp.heads.head_result import HeadScore
from five_head_nlp.heads.scoring_config import SettingConfig
from five_head_nlp.prediction.generation_state import GenerationState
from five_head_nlp.prediction.prediction_result import CandidateToken


class SettingPainter(BaseHead):
    """Head 3 — Setting Painter.

    Maintains environmental and setting consistency throughout the narrative.
    Rewards tokens aligning with the active setting (e.g., castle, cave, garden, forest)
    while supporting deliberate setting transitions.

    State updates occur ONLY after token selection via update_state().
    """

    def __init__(
        self,
        config: Optional[SettingConfig] = None,
        weight: float = 0.15,
        enabled: bool = True,
    ) -> None:
        """Initialize Setting Painter head."""
        super().__init__(name="Setting Painter", weight=weight, enabled=enabled)
        self.config: SettingConfig = config or SettingConfig()
        self.current_setting: Optional[str] = None
        self.setting_history: List[str] = []

    @property
    def all_setting_tokens(self) -> Set[str]:
        """Union of core settings and environmental setting nouns."""
        return self.config.core_settings.union(self.config.setting_nouns)

    def is_setting(self, token: str) -> bool:
        """Return True if token is in setting lexicon."""
        return token.lower() in self.all_setting_tokens

    def score_candidates(
        self,
        context: Tuple[str, ...],
        candidates: List[CandidateToken],
        state: Optional[GenerationState] = None,
    ) -> Dict[str, HeadScore]:
        """Score each candidate token for setting consistency and relevance."""
        scores: Dict[str, HeadScore] = {}

        for c in candidates:
            tok = c.token.lower()

            if self.is_setting(tok):
                if self.current_setting is None:
                    # Initial setting establishment
                    raw = self.config.consistency_bonus
                    explanation = f"Establishes initial setting environment '{tok}'."
                elif tok == self.current_setting:
                    raw = self.config.consistency_bonus
                    explanation = f"Reinforces current setting environment '{tok}'."
                elif tok in self.config.core_settings:
                    # Core setting transition (e.g. from forest to castle)
                    raw = self.config.shift_bonus
                    explanation = f"Narrative transition to new core setting '{tok}'."
                else:
                    raw = (self.config.consistency_bonus + self.config.shift_bonus) / 2.0
                    explanation = f"Associated setting/environmental descriptor '{tok}'."
            else:
                raw = self.config.non_setting_score
                explanation = "Non-setting lexical token (neutral score)."

            norm = max(0.0, min(1.0, raw))
            scores[c.token] = HeadScore(
                head_name=self.name,
                token=c.token,
                raw_score=raw,
                normalized_score=norm,
                explanation=explanation,
                evidence={
                    "is_setting": self.is_setting(tok),
                    "current_setting": self.current_setting,
                    "setting_history": list(self.setting_history),
                },
            )

        return scores

    def update_state(self, selected_token: str) -> None:
        """Update active setting after token is chosen."""
        tok = selected_token.lower()
        if self.is_setting(tok):
            self.current_setting = tok
            self.setting_history.append(tok)

    def reset_state(self) -> None:
        """Reset setting tracking state."""
        self.current_setting = None
        self.setting_history.clear()
