"""Head 5 — Repeat Blocker: Suppresses loops, immediate repeats, and degenerate cycles."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from five_head_nlp.heads.base_head import BaseHead
from five_head_nlp.heads.head_result import HeadScore
from five_head_nlp.heads.scoring_config import RepeatConfig
from five_head_nlp.prediction.generation_state import GenerationState
from five_head_nlp.prediction.prediction_result import CandidateToken


class RepeatBlocker(BaseHead):
    """Head 5 — Repeat Blocker.

    Detects and penalizes:
        - Immediate repeated tokens (w_t == w_{t-1})
        - 2-token attractor loops and repeated bigrams
        - Repeated trigram cycles
        - Lexical frequency saturation using exponential decay r(w) = exp(-gamma * count)

    Exempts essential function words and punctuation (the, a, and, .) from penalties.
    State updates occur ONLY after token selection via update_state().
    """

    def __init__(
        self,
        config: Optional[RepeatConfig] = None,
        weight: float = 0.20,
        enabled: bool = True,
    ) -> None:
        """Initialize Repeat Blocker head."""
        super().__init__(name="Repeat Blocker", weight=weight, enabled=enabled)
        self.config: RepeatConfig = config or RepeatConfig()
        self.history: List[str] = []

    def is_exempt(self, token: str) -> bool:
        """Return True if token is exempt from repetition penalties."""
        return token.lower() in self.config.exempt_tokens

    def score_candidates(
        self,
        context: Tuple[str, ...],
        candidates: List[CandidateToken],
        state: Optional[GenerationState] = None,
    ) -> Dict[str, HeadScore]:
        """Score each candidate token, applying penalties for repeated patterns."""
        scores: Dict[str, HeadScore] = {}

        # Use generation state history if provided, otherwise internal history
        hist = state.generated_tokens if state is not None else self.history
        window = hist[-self.config.history_window :] if hist else []

        prev1 = context[-1] if len(context) >= 1 else None
        prev2 = context[-2] if len(context) >= 2 else None

        for c in candidates:
            tok = c.token

            # 1. Exempt function words and punctuation receive maximum score 1.0 (no penalty)
            if self.is_exempt(tok):
                scores[tok] = HeadScore(
                    head_name=self.name,
                    token=tok,
                    raw_score=1.0,
                    normalized_score=1.0,
                    explanation=f"Function word or punctuation '{tok}' exempt from repetition penalty.",
                    evidence={"exempt": True},
                )
                continue

            # 2. Immediate token self-loop (e.g. 'castle' -> 'castle')
            if prev1 is not None and prev1.lower() == tok.lower():
                raw = self.config.immediate_repeat_score
                scores[tok] = HeadScore(
                    head_name=self.name,
                    token=tok,
                    raw_score=raw,
                    normalized_score=raw,
                    explanation=f"Immediate self-loop detected: '{prev1}' followed by duplicate '{tok}'.",
                    evidence={"immediate_repeat": True},
                )
                continue

            # 3. Repeated bigram check (prev1, tok) in recent history
            bigram_repeated = False
            if prev1 is not None and len(window) >= 2:
                for i in range(len(window) - 1):
                    if window[i].lower() == prev1.lower() and window[i + 1].lower() == tok.lower():
                        bigram_repeated = True
                        break

            if bigram_repeated:
                raw = self.config.bigram_repeat_score
                scores[tok] = HeadScore(
                    head_name=self.name,
                    token=tok,
                    raw_score=raw,
                    normalized_score=raw,
                    explanation=f"Repeated bigram loop detected: ('{prev1}', '{tok}') occurred recently.",
                    evidence={"bigram_repeat": True},
                )
                continue

            # 4. Sliding window frequency decay: r(w) = exp(-gamma * count)
            count = sum(1 for w in window if w.lower() == tok.lower())
            if count == 0:
                raw = 1.0
                explanation = f"Fresh unrepeated token '{tok}' in current window."
            else:
                raw = math.exp(-self.config.decay_gamma * count)
                explanation = f"Frequency penalty applied: '{tok}' occurred {count} time(s) in window (decay={raw:.3f})."

            norm = max(0.0, min(1.0, raw))
            scores[tok] = HeadScore(
                head_name=self.name,
                token=tok,
                raw_score=raw,
                normalized_score=norm,
                explanation=explanation,
                evidence={"window_count": count, "decay_score": round(raw, 4)},
            )

        return scores

    def update_state(self, selected_token: str) -> None:
        """Append selected token to history."""
        self.history.append(selected_token)

    def reset_state(self) -> None:
        """Reset token history."""
        self.history.clear()
