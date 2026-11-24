"""Head 2 — Character Keeper: Maintains narrative character continuity and focus."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from five_head_nlp.heads.base_head import BaseHead
from five_head_nlp.heads.head_result import HeadScore
from five_head_nlp.heads.scoring_config import CharacterConfig
from five_head_nlp.prediction.generation_state import GenerationState
from five_head_nlp.prediction.prediction_result import CandidateToken


class CharacterKeeper(BaseHead):
    """Head 2 — Character Keeper.

    Maintains character continuity throughout the story by tracking mentioned characters,
    identifying character names, and rewarding character continuity and introductions.

    State updates occur ONLY after token selection via update_state().
    """

    def __init__(
        self,
        config: Optional[CharacterConfig] = None,
        weight: float = 0.15,
        enabled: bool = True,
    ) -> None:
        """Initialize Character Keeper head."""
        super().__init__(name="Character Keeper", weight=weight, enabled=enabled)
        self.config: CharacterConfig = config or CharacterConfig()
        self.active_character: Optional[str] = None
        self.mentioned_characters: List[str] = []
        self.character_counts: Dict[str, int] = {}

    @property
    def all_character_tokens(self) -> Set[str]:
        """Union of named characters and character nouns."""
        return self.config.named_characters.union(self.config.character_nouns)

    def is_character(self, token: str) -> bool:
        """Return True if token is in character lexicon."""
        return token.lower() in self.all_character_tokens

    def score_candidates(
        self,
        context: Tuple[str, ...],
        candidates: List[CandidateToken],
        state: Optional[GenerationState] = None,
    ) -> Dict[str, HeadScore]:
        """Score each candidate token for character continuity and presence."""
        scores: Dict[str, HeadScore] = {}
        prev_token = context[-1] if context else (state.all_tokens[-1] if state and state.all_tokens else None)

        for c in candidates:
            tok = c.token.lower()

            if self.is_character(tok):
                # Check for immediate repeat of character name (e.g., 'milo milo')
                if prev_token and prev_token.lower() == tok:
                    raw = self.config.immediate_character_repeat_penalty
                    explanation = f"Penalized immediate duplicate character mention '{tok}'."
                elif tok == self.active_character:
                    raw = self.config.continuity_bonus
                    explanation = f"Maintains active character focus on '{tok}'."
                elif tok in self.character_counts:
                    raw = (self.config.continuity_bonus + self.config.introduction_bonus) / 2.0
                    explanation = f"Re-introduces previously mentioned character '{tok}'."
                else:
                    raw = self.config.introduction_bonus
                    explanation = f"Introduces new character '{tok}' to the narrative."
            else:
                raw = self.config.non_character_score
                explanation = "Non-character lexical token (neutral score)."

            norm = max(0.0, min(1.0, raw))
            scores[c.token] = HeadScore(
                head_name=self.name,
                token=c.token,
                raw_score=raw,
                normalized_score=norm,
                explanation=explanation,
                evidence={
                    "is_character": self.is_character(tok),
                    "active_character": self.active_character,
                    "mentioned_characters": list(self.mentioned_characters),
                },
            )

        return scores

    def update_state(self, selected_token: str) -> None:
        """Update active character and mention history after token is chosen."""
        tok = selected_token.lower()
        if self.is_character(tok):
            self.active_character = tok
            self.mentioned_characters.append(tok)
            self.character_counts[tok] = self.character_counts.get(tok, 0) + 1

    def reset_state(self) -> None:
        """Reset character tracking state."""
        self.active_character = None
        self.mentioned_characters.clear()
        self.character_counts.clear()
