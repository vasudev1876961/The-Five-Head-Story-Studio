"""Evaluation module comparing baseline N-gram decoding with multi-head scoring configurations."""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from five_head_nlp.heads.head_orchestrator import HeadOrchestrator
from five_head_nlp.heads.scoring_config import ScoringConfig
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.prediction.predictor import NextTokenPredictor
from five_head_nlp.smoothing.backoff import BackoffSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


class HeadEvaluator:
    """Evaluates and compares baseline N-gram generation against multi-head ablations.

    Ablation Configurations:
        1. Baseline (pure statistical N-gram / backoff)
        2. Grammar Scout only (100% grammar weight)
        3. Grammar + Character (70% grammar, 30% character)
        4. Grammar + Character + Setting (50% grammar, 25% char, 25% setting)
        5. Grammar + Character + Setting + Action (45% grammar, 20% char, 20% setting, 15% action)
        6. All Five Heads (40% grammar, 15% char, 15% setting, 10% action, 20% repeat)
    """

    def __init__(
        self,
        predictor: Optional[NextTokenPredictor] = None,
        test_sentences: Optional[Sequence[Sequence[str]]] = None,
        corpus_loader: Optional[CorpusLoader] = None,
    ) -> None:
        """Initialize HeadEvaluator with leak-free test partition."""
        self.loader: CorpusLoader = corpus_loader or CorpusLoader()
        sentences = self.loader.get_sentences()

        # Deterministic 3-way split (matching previous phases)
        rng = random.Random(42)
        indices = list(range(len(sentences)))
        rng.shuffle(indices)
        train_idx = sorted(indices[:24])
        val_idx = sorted(indices[24:29])
        test_idx = sorted(indices[29:])

        self.train_sentences: List[List[str]] = [sentences[i] for i in train_idx]
        self.val_sentences: List[List[str]] = [sentences[i] for i in val_idx]
        self.test_sentences: List[List[str]] = (
            [list(s) for s in test_sentences] if test_sentences is not None
            else [sentences[i] for i in test_idx]
        )

        if predictor is not None:
            self.predictor: NextTokenPredictor = predictor
        else:
            model = TrigramModel()
            model.fit(self.train_sentences)
            smoother = BackoffSmoother(discount=0.5, model=model)
            self.predictor = NextTokenPredictor(model=model, smoother=smoother)

    def extract_events(
        self, sentences: Sequence[Sequence[str]]
    ) -> List[Tuple[Tuple[str, ...], str]]:
        """Extract (context, gold_token) pairs respecting sentence boundaries."""
        events: List[Tuple[Tuple[str, ...], str]] = []
        for s in sentences:
            for i, gold in enumerate(s):
                if i == 0:
                    ctx: Tuple[str, ...] = ()
                elif i == 1:
                    ctx = (s[0],)
                else:
                    ctx = (s[i - 2], s[i - 1])
                events.append((ctx, gold))
        return events

    def evaluate_configuration(
        self,
        name: str,
        mode: str,
        config: Optional[ScoringConfig] = None,
    ) -> Dict[str, Any]:
        """Evaluate a specific scoring head configuration on test events."""
        events = self.extract_events(self.test_sentences)
        total_events = len(events)

        orchestrator = HeadOrchestrator(config=config) if config else None

        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        reciprocal_ranks: List[float] = []
        latencies_ms: List[float] = []

        for ctx, gold in events:
            t0 = time.perf_counter()
            res = self.predictor.predict_next(
                context=ctx,
                mode=mode,
                orchestrator=orchestrator,
                strategy="greedy",
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000)

            # Evaluate top-1 selection
            if res.selected_token == gold:
                top1_correct += 1

            # Find rank of gold token in candidate list
            cand = res.get_candidate(gold)
            if cand is not None:
                rank = cand.rank
                reciprocal_ranks.append(1.0 / rank)
                if rank <= 3:
                    top3_correct += 1
                if rank <= 5:
                    top5_correct += 1
            else:
                reciprocal_ranks.append(0.0)

        # Measure narrative rollout qualities on 5 generated sequences
        rollout_stats = self.evaluate_rollout_qualities(mode=mode, orchestrator=orchestrator)

        latencies_ms.sort()
        mean_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        p95_lat = latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0

        return {
            "configuration": name,
            "mode": mode,
            "total_events": total_events,
            "top1_accuracy": round(top1_correct / total_events, 4) if total_events > 0 else 0.0,
            "top3_accuracy": round(top3_correct / total_events, 4) if total_events > 0 else 0.0,
            "top5_accuracy": round(top5_correct / total_events, 4) if total_events > 0 else 0.0,
            "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
            "latency_mean_ms": round(mean_lat, 4),
            "latency_p95_ms": round(p95_lat, 4),
            "rollout_repetition_rate": rollout_stats["repetition_rate"],
            "character_continuity_pct": rollout_stats["character_continuity_pct"],
            "setting_consistency_pct": rollout_stats["setting_consistency_pct"],
            "action_token_frequency_pct": rollout_stats["action_token_pct"],
        }

    def evaluate_rollout_qualities(
        self,
        mode: str,
        orchestrator: Optional[HeadOrchestrator] = None,
        num_prompts: int = 5,
    ) -> Dict[str, float]:
        """Generate rollouts from standard test prompts to measure repetition and thematic tokens."""
        prompts = [
            ["the", "clever", "cat"],
            ["pip", "flew"],
            ["in", "the", "secret"],
            ["a", "sleepy", "owl"],
            ["zara", "found"],
        ]

        total_tokens = 0
        repeated_tokens = 0
        character_mentions = 0
        setting_mentions = 0
        action_mentions = 0

        char_set = {"milo", "pip", "zara", "owl", "fox", "cat", "dragon", "wizard"}
        setting_set = {"castle", "cave", "forest", "garden", "tree", "lake", "kingdom"}
        action_set = {"flew", "ran", "walked", "found", "showed", "opened", "made", "help", "helped", "rested", "liked"}

        for p in prompts[:num_prompts]:
            seq, _ = self.predictor.generate(
                prompt=p,
                max_tokens=15,
                strategy="greedy",
                mode=mode,
                orchestrator=orchestrator,
            )
            gen = seq[len(p):]
            total_tokens += len(gen)

            # Check 2-token window repeats
            window = []
            for t in gen:
                if t.lower() not in {".", "the", "a", "and", "in", "to", "near"}:
                    if t.lower() in [w.lower() for w in window[-6:]]:
                        repeated_tokens += 1
                    window.append(t)

                if t.lower() in char_set:
                    character_mentions += 1
                if t.lower() in setting_set:
                    setting_mentions += 1
                if t.lower() in action_set:
                    action_mentions += 1

        rep_rate = round(repeated_tokens / total_tokens, 4) if total_tokens > 0 else 0.0
        char_pct = round((character_mentions / total_tokens) * 100, 2) if total_tokens > 0 else 0.0
        set_pct = round((setting_mentions / total_tokens) * 100, 2) if total_tokens > 0 else 0.0
        act_pct = round((action_mentions / total_tokens) * 100, 2) if total_tokens > 0 else 0.0

        return {
            "repetition_rate": rep_rate,
            "character_continuity_pct": char_pct,
            "setting_consistency_pct": set_pct,
            "action_token_pct": act_pct,
        }

    def run_full_ablation(self) -> List[Dict[str, Any]]:
        """Run complete ablation comparison across all head combinations."""
        # 1. Baseline
        res_baseline = self.evaluate_configuration("1. Baseline (Unfused N-gram)", mode="baseline")

        # 2. Grammar Scout Only
        cfg_g = ScoringConfig(
            grammar_weight=1.0, character_weight=0.0, setting_weight=0.0, action_weight=0.0, repeat_weight=0.0,
            enable_character=False, enable_setting=False, enable_action=False, enable_repeat=False,
        )
        res_g = self.evaluate_configuration("2. Grammar Scout Only", mode="five_head", config=cfg_g)

        # 3. Grammar + Character
        cfg_gc = ScoringConfig(
            grammar_weight=0.70, character_weight=0.30, setting_weight=0.0, action_weight=0.0, repeat_weight=0.0,
            enable_setting=False, enable_action=False, enable_repeat=False,
        )
        res_gc = self.evaluate_configuration("3. Grammar + Character", mode="five_head", config=cfg_gc)

        # 4. Grammar + Character + Setting
        cfg_gcs = ScoringConfig(
            grammar_weight=0.50, character_weight=0.25, setting_weight=0.25, action_weight=0.0, repeat_weight=0.0,
            enable_action=False, enable_repeat=False,
        )
        res_gcs = self.evaluate_configuration("4. Grammar + Character + Setting", mode="five_head", config=cfg_gcs)

        # 5. Grammar + Character + Setting + Action
        cfg_gcsa = ScoringConfig(
            grammar_weight=0.45, character_weight=0.20, setting_weight=0.20, action_weight=0.15, repeat_weight=0.0,
            enable_repeat=False,
        )
        res_gcsa = self.evaluate_configuration("5. Grammar + Char + Setting + Action", mode="five_head", config=cfg_gcsa)

        # 6. All Five Heads
        cfg_all = ScoringConfig(
            grammar_weight=0.40, character_weight=0.15, setting_weight=0.15, action_weight=0.10, repeat_weight=0.20,
        )
        res_all = self.evaluate_configuration("6. All Five Heads (Full System)", mode="five_head", config=cfg_all)

        return [res_baseline, res_g, res_gc, res_gcs, res_gcsa, res_all]
