"""Orchestrator coordinating all five transparent scoring heads and score fusion."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

from five_head_nlp.heads.action_booster import ActionBooster
from five_head_nlp.heads.base_head import BaseHead
from five_head_nlp.heads.character_keeper import CharacterKeeper
from five_head_nlp.heads.grammar_scout import GrammarScout
from five_head_nlp.heads.hamilton_apportionment import HamiltonApportionment  # check import path below
from five_head_nlp.heads.head_result import (
    FusedCandidate,
    HeadScore,
    OrchestratorResult,
)
from five_head_nlp.heads.repeat_blocker import RepeatBlocker
from five_head_nlp.heads.score_fusion import ScoreFusion
from five_head_nlp.heads.scoring_config import ScoringConfig
from five_head_nlp.heads.setting_painter import SettingPainter
from five_head_nlp.prediction.generation_state import GenerationState
from five_head_nlp.prediction.prediction_result import CandidateToken


class HeadOrchestrator:
    """Coordinates the five independent scoring heads, fusion, and state management.

    Decoupling Principles:
        1. Pure Evaluation: evaluate_candidates() produces scores without modifying internal states.
        2. Explicit State Updates: state mutations occur ONLY upon update_state(selected_token).
    """

    def __init__(
        self,
        config: Optional[ScoringConfig] = None,
        fusion: Optional[ScoreFusion] = None,
        apportionment: Optional[HamiltonApportionment] = None,
    ) -> None:
        """Initialize HeadOrchestrator with all five scoring heads."""
        self.config: ScoringConfig = config or ScoringConfig()
        self.config.validate()

        self.fusion: ScoreFusion = fusion or ScoreFusion()
        self.apportionment: HamiltonApportionment = apportionment or HamiltonApportionment()

        # Instantiate heads with config parameters
        self.grammar_head: GrammarScout = GrammarScout(
            weight=self.config.grammar_weight, enabled=self.config.enable_grammar
        )
        self.character_head: CharacterKeeper = CharacterKeeper(
            config=self.config.character,
            weight=self.config.character_weight,
            enabled=self.config.enable_character,
        )
        self.setting_head: SettingPainter = SettingPainter(
            config=self.config.setting,
            weight=self.config.setting_weight,
            enabled=self.config.enable_setting,
        )
        self.action_head: ActionBooster = ActionBooster(
            config=self.config.action,
            weight=self.config.action_weight,
            enabled=self.config.enable_action,
        )
        self.repeat_head: RepeatBlocker = RepeatBlocker(
            config=self.config.repeat,
            weight=self.config.repeat_weight,
            enabled=self.config.enable_repeat,
        )

        self.heads: Dict[str, BaseHead] = {
            "Grammar Scout": self.grammar_head,
            "Character Keeper": self.character_head,
            "Setting Painter": self.setting_head,
            "Action Booster": self.action_head,
            "Repeat Blocker": self.repeat_head,
        }

    def sync_weights_from_config(self) -> None:
        """Synchronize head weights and enabled states from current config."""
        self.config.validate()
        self.grammar_head.weight = self.config.grammar_weight
        self.grammar_head.enabled = self.config.enable_grammar

        self.character_head.weight = self.config.character_weight
        self.character_head.enabled = self.config.enable_character

        self.setting_head.weight = self.config.setting_weight
        self.setting_head.enabled = self.config.enable_setting

        self.action_head.weight = self.config.action_weight
        self.action_head.enabled = self.config.enable_action

        self.repeat_head.weight = self.config.repeat_weight
        self.repeat_head.enabled = self.config.enable_repeat

    def get_active_weights(self) -> Dict[str, float]:
        """Return weights for currently enabled heads."""
        return {name: head.weight for name, head in self.heads.items() if head.enabled and head.weight > 0.0}

    def evaluate_candidates(
        self,
        context: Tuple[str, ...],
        candidates: List[CandidateToken],
        state: Optional[GenerationState] = None,
    ) -> OrchestratorResult:
        """Evaluate candidate tokens across all active heads, fuse scores, and rank.

        Does NOT mutate internal head states.
        """
        start_time = time.perf_counter()
        self.sync_weights_from_config()

        if not candidates:
            return OrchestratorResult(
                selected_token=None,
                context=list(context),
                fused_candidates=[],
                head_weights={},
                d10_allocation={},
                mode="five_head",
                metadata={"reason": "empty_candidates_list", "latency_ms": 0.0},
            )

        active_weights = self.get_active_weights()
        if not active_weights:
            # Fallback to grammar head if everything was disabled
            active_weights = {"Grammar Scout": 1.0}

        # 1. Score across all enabled heads
        head_evaluations: Dict[str, Dict[str, HeadScore]] = {}
        for head_name, head in self.heads.items():
            if head.enabled and head.weight > 0.0:
                head_evaluations[head_name] = head.score_candidates(
                    context=context, candidates=candidates, state=state
                )

        # 2. Fuse scores and rank candidates
        fused_candidates = self.fusion.fuse_scores(
            candidate_tokens=candidates,
            head_scores=head_evaluations,
            weights=active_weights,
        )

        # 3. Compute Hamilton d10 allocation
        d10_alloc = self.apportionment.apportion(active_weights)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        top_choice = fused_candidates[0].token if fused_candidates else None

        return OrchestratorResult(
            selected_token=top_choice,
            context=list(context),
            fused_candidates=fused_candidates,
            head_weights=self.fusion.normalize_weights(active_weights),
            d10_allocation=d10_alloc,
            mode="five_head",
            metadata={
                "latency_ms": round(elapsed_ms, 4),
                "num_candidates": len(fused_candidates),
                "active_heads": list(active_weights.keys()),
            },
        )

    def update_state(self, selected_token: str) -> None:
        """Propagate selected token to all stateful heads."""
        self.character_head.update_state(selected_token)
        self.setting_head.update_state(selected_token)
        self.action_head.update_state(selected_token)
        self.repeat_head.update_state(selected_token)

    def reset_state(self) -> None:
        """Reset state across all stateful heads."""
        self.character_head.reset_state()
        self.setting_head.reset_state()
        self.action_head.reset_state()
        self.repeat_head.reset_state()
