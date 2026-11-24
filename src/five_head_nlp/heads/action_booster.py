"""Head 4 — Action Booster: Encourages dynamic verbs and narrative progression."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from five_head_nlp.heads.base_head import BaseHead
from five_head_nlp.heads.head_result import HeadScore
from five_head_nlp.heads.scoring_config import ActionConfig
from five_head_nlp.prediction.generation_state import GenerationState
from five_head_nlp.prediction.prediction_result import CandidateToken


class ActionBooster(BaseHead):
    """Head 4 — Action Booster.

    Encourages meaningful verbs and dynamic events (movement, interaction, discovery)
    to prevent passive or static descriptions and propel plot forward.
    """

    def __init__(
        self,
        config: Optional[ActionConfig] = None,
        weight: float = 0.10,
        enabled: bool = True,
    ) -> None:
        """Initialize Action Booster head."""
        super().__init__(name="Action Booster", weight=weight, enabled=enabled)
        self.config: ActionConfig = config or ActionConfig()
        self.action_history: List[str] = []

    @property
    def all_actions(self) -> Set[str]:
        """Union of dynamic actions and stative/perceptual verbs."""
        return self.config.dynamic_actions.union(self.config.state_verbs)

    def is_action(self, token: str) -> bool:
        """Return True if token is an action verb."""
        return token.lower() in self.all_actions

    def score_candidates(
        self,
        context: Tuple[str, ...],
        candidates: List[CandidateToken],
        state: Optional[GenerationState] = None,
    ) -> Dict[str, HeadScore]:
        """Score each candidate token for action intensity."""
        scores: Dict[str, HeadScore] = {}

        for c in candidates:
            tok = c.token.lower()

            if tok in self.config.dynamic_actions:
                raw = self.config.dynamic_action_bonus
                explanation = f"Dynamic action verb '{tok}' advances plot momentum."
                category = "dynamic"
            elif tok in self.config.state_verbs:
                raw = self.config.state_action_bonus
                explanation = f"Perceptual or state verb '{tok}' describes character experience."
                category = "stative"
            else:
                raw = self.config.non_action_score
                explanation = "Non-action token (neutral score)."
                category = "non_action"

            norm = max(0.0, min(1.0, raw))
            scores[c.token] = HeadScore(
                head_name=self.name,
                token=c.token,
                raw_score=raw,
                normalized_score=norm,
                explanation=explanation,
                evidence={
                    "is_action": self.is_action(tok),
                    "action_category": category,
                },
            )

        return scores

    def update_state(self, selected_token: str) -> None:
        """Record selected action verb into action history."""
        tok = selected_token.lower()
        if self.is_action(tok):
            self.action_history.append(tok)

    def reset_state(self) -> None:
        """Reset action history."""
        self.action_history.clear()
