"""Monte Carlo Generation Harness for batch experimental story execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from five_head_nlp.experiments.experiment_config import ExperimentConfig
from five_head_nlp.experiments.experiment_registry import list_all_experiments
from five_head_nlp.experiments.story_generator import (
    GeneratedStory,
    StoryFailureRecord,
    StoryGenerator,
)
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


@dataclass
class HarnessResult:
    """Aggregated output of a Monte Carlo experimental generation batch."""

    stories: List[GeneratedStory] = field(default_factory=list)
    failures: List[StoryFailureRecord] = field(default_factory=list)
    config_counts: Dict[str, int] = field(default_factory=dict)
    failure_counts: Dict[str, int] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        """Summarize generation run counts and success rates."""
        summary: Dict[str, Any] = {}
        for cid in sorted(self.config_counts.keys()):
            succ = self.config_counts[cid]
            fails = self.failure_counts.get(cid, 0)
            total = succ + fails
            rate = (succ / total * 100.0) if total > 0 else 0.0
            summary[cid] = {
                "success_count": succ,
                "failure_count": fails,
                "total_attempts": total,
                "success_rate_pct": round(rate, 2),
            }
        return summary


class GenerationHarness:
    """Harness executing reproducible Monte Carlo story generation across configurations."""

    def __init__(
        self,
        loader: Optional[CorpusLoader] = None,
        model: Optional[TrigramModel] = None,
        base_seed: int = 42,
    ) -> None:
        """Initialize the generation harness.

        Parameters
        ----------
        loader : Optional[CorpusLoader], default=None
            CorpusLoader instance.
        model : Optional[TrigramModel], default=None
            Pre-fitted TrigramModel.
        base_seed : int, default=42
            Base seed for deterministic Monte Carlo generation.
        """
        self.loader = loader or CorpusLoader().load()
        if model is None:
            self.model = TrigramModel()
            self.model.fit_corpus(self.loader)
        else:
            self.model = model
        self.base_seed = base_seed

    def run_configuration(
        self,
        config: ExperimentConfig,
        num_stories: int = 200,
        config_index: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> HarnessResult:
        """Run generation for a single experimental configuration.

        Parameters
        ----------
        config : ExperimentConfig
            Experimental configuration to evaluate.
        num_stories : int, default=200
            Number of stories to generate.
        config_index : int, default=0
            Ordinal index of config for isolated seed derivation.
        progress_callback : Optional[Callable[[int, int], None]], default=None
            Callback receiving (stories_completed, total_stories).

        Returns
        -------
        HarnessResult
            Batch generation results for this configuration.
        """
        generator = StoryGenerator(config=config, model=self.model, loader=self.loader)
        result = HarnessResult()

        # 1 Warm-up run to prime caches and isolate timing
        _ = generator.generate_story(seed=self.base_seed, story_id="WARMUP")

        for idx in range(num_stories):
            # Deterministic, non-overlapping seed per story:
            # base_seed + 10,000 * config_index + idx
            story_seed = self.base_seed + (config_index * 10_000) + idx
            story_id = f"{config.config_id}_{idx + 1:04d}"

            story, failure = generator.generate_story(seed=story_seed, story_id=story_id)

            if story is not None:
                result.stories.append(story)
                result.config_counts[config.config_id] = (
                    result.config_counts.get(config.config_id, 0) + 1
                )
            elif failure is not None:
                result.failures.append(failure)
                result.failure_counts[config.config_id] = (
                    result.failure_counts.get(config.config_id, 0) + 1
                )

            if progress_callback:
                progress_callback(idx + 1, num_stories)

        return result

    def run_all(
        self,
        configs: Optional[Sequence[ExperimentConfig]] = None,
        stories_per_config: int = 200,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> HarnessResult:
        """Run all experimental configurations in sequence.

        Parameters
        ----------
        configs : Optional[Sequence[ExperimentConfig]], default=None
            List of configurations to run. Defaults to all 8 canonical conditions (E1-E8).
        stories_per_config : int, default=200
            Number of stories per condition.
        progress_callback : Optional[Callable[[str, int, int], None]], default=None
            Callback receiving (config_id, completed_in_config, total_in_config).

        Returns
        -------
        HarnessResult
            Combined results across all conditions.
        """
        all_configs = list(configs) if configs is not None else list_all_experiments()
        combined_result = HarnessResult()

        for c_idx, cfg in enumerate(all_configs):
            def _sub_callback(completed: int, total: int) -> None:
                if progress_callback:
                    progress_callback(cfg.config_id, completed, total)

            cfg_res = self.run_configuration(
                config=cfg,
                num_stories=stories_per_config,
                config_index=c_idx,
                progress_callback=_sub_callback,
            )

            combined_result.stories.extend(cfg_res.stories)
            combined_result.failures.extend(cfg_res.failures)
            for k, v in cfg_res.config_counts.items():
                combined_result.config_counts[k] = combined_result.config_counts.get(k, 0) + v
            for k, v in cfg_res.failure_counts.items():
                combined_result.failure_counts[k] = combined_result.failure_counts.get(k, 0) + v

        return combined_result
