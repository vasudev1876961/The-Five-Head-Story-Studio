"""Story generation service connecting Streamlit UI to the five_head_nlp backend."""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

from five_head_nlp.experiments.metrics import compute_story_metrics
from five_head_nlp.experiments.story_generator import detokenize_story
from five_head_nlp.prediction import (
    BaseDecoder,
    GreedyDecoder,
    TemperatureDecoder,
    TopKDecoder,
)
from five_head_nlp.tokenization.tokenizer import StoryTokenizer
from five_head_nlp.ui.model_service import ModelService, get_corpus_loader, get_model


class GenerationService:
    """Orchestrates story generation runs and collects per-token diagnostic data."""

    def __init__(self) -> None:
        """Initialize generation service."""
        self.tokenizer = StoryTokenizer()

    def _create_decoder(self, name: str, params: Dict[str, Any], seed: int) -> BaseDecoder:
        """Instantiate decoder corresponding to user selection."""
        n = name.lower()
        if "top" in n:
            k = int(params.get("k", 5))
            return TopKDecoder(k=k, seed=seed)
        elif "temp" in n:
            temp = float(params.get("temperature", 0.8))
            return TemperatureDecoder(temperature=temp, seed=seed)
        else:
            return GreedyDecoder()

    def generate_story(
        self,
        config: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Generate a complete multi-sentence story while recording token diagnostics.

        Parameters
        ----------
        config : Dict[str, Any]
            Generation configuration dictionary.

        Returns
        -------
        Tuple[Dict[str, Any], List[Dict[str, Any]]]
            1. story_record containing text, tokens, metrics, latency, and parameters.
            2. step_diagnostics containing per-token candidate distributions and explanations.
        """
        mode = config.get("mode", "five_head")
        smoother_name = config.get("smoother_name", "Trigram Backoff")
        smoother_params = config.get("smoother_params", {})
        decoder_name = config.get("decoder_name", "Top-k")
        decoder_params = config.get("decoder_params", {})
        seed = int(config.get("seed", 42))
        max_tokens = int(config.get("max_tokens", 100))
        max_sentences = int(config.get("max_sentences", 8))
        min_sentences = int(config.get("min_sentences", 3))
        starter_text = str(config.get("starter_prompt", "")).strip()

        # Initialize isolated RNG
        rng = random.Random(seed)

        # 1. Models and Predictor
        model = get_model(order=3)
        smoother = ModelService.create_smoother(smoother_name, model, smoother_params)

        orchestrator = None
        if mode == "five_head":
            weights = config.get("weights")
            enabled = config.get("enabled_heads")
            orchestrator = ModelService.create_orchestrator(weights=weights, enabled_heads=enabled)
            orchestrator.reset_state()

        predictor = ModelService.create_predictor(
            model=model, smoother=smoother, orchestrator=orchestrator
        )

        decoder = self._create_decoder(
            decoder_name, decoder_params, seed=rng.randint(1, 10_000_000)
        )

        # 2. Starting Prompts
        loader = get_corpus_loader()
        corpus_starters = [
            tuple(s[:2]) for s in loader.get_sentences() if len(s) >= 2
        ] or [("milo", "found")]

        story_tokens: List[str] = []
        token_latencies_ms: List[float] = []
        step_diagnostics: List[Dict[str, Any]] = []
        sentences_completed = 0
        termination_reason = "Target length achieved"

        start_time = time.perf_counter()

        while (
            len(story_tokens) < max_tokens
            or sentences_completed < min_sentences
        ) and sentences_completed < max_sentences:

            if len(story_tokens) >= max_tokens + 25:
                termination_reason = "Maximum token safety cap reached"
                break

            # Choose sentence starter
            if not story_tokens and starter_text:
                starter_tokens = self.tokenizer.tokenize(starter_text)
                if not starter_tokens:
                    starter_tokens = list(rng.choice(corpus_starters))
            else:
                starter_tokens = list(rng.choice(corpus_starters))

            curr_sentence = list(starter_tokens)

            # Update orchestrator with prompt tokens
            if orchestrator is not None:
                for t in curr_sentence:
                    orchestrator.update_state(t)

            sent_step = 0
            max_sent_steps = 30
            sent_closed = False
            ngram_history: List[Tuple[str, str]] = []

            while sent_step < max_sent_steps and not sent_closed:
                ctx_window = min(len(curr_sentence), model.n - 1)
                context = tuple(curr_sentence[-ctx_window:])

                t0 = time.perf_counter()
                pred = predictor.predict_next(
                    context=context,
                    decoder=decoder,
                    mode=mode,
                    orchestrator=orchestrator,
                )
                lat_ms = (time.perf_counter() - t0) * 1000.0

                next_tok = pred.selected_token
                token_latencies_ms.append(lat_ms)

                if next_tok is None:
                    # Unsmoothed dead end or vocabulary exhaustion
                    curr_sentence.append(".")
                    if orchestrator is not None:
                        orchestrator.update_state(".")
                    sent_closed = True
                    break

                # Extract diagnostic snapshot
                orch_meta = pred.metadata.get("orchestrator_result")
                step_info = {
                    "step_index": len(story_tokens) + len(curr_sentence),
                    "token": next_tok,
                    "context": list(context),
                    "strategy": pred.strategy,
                    "latency_ms": round(lat_ms, 3),
                    "mode": mode,
                    "candidates": [
                        {
                            "token": c.token,
                            "probability": round(c.probability, 4),
                            "rank": c.rank,
                            "log_probability": round(c.log_probability, 4),
                        }
                        for c in pred.candidates[:10]
                    ],
                    "orchestrator_result": orch_meta,
                }
                step_diagnostics.append(step_info)

                # Attractor loop guard
                if len(curr_sentence) >= 2:
                    current_bg = (curr_sentence[-1].lower(), next_tok.lower())
                    ngram_history.append(current_bg)
                    if len(ngram_history) >= 4 and ngram_history[-1] == ngram_history[-2] == ngram_history[-3]:
                        curr_sentence.append(".")
                        if orchestrator is not None:
                            orchestrator.update_state(".")
                        sent_closed = True
                        break

                curr_sentence.append(next_tok)
                if orchestrator is not None:
                    orchestrator.update_state(next_tok)

                sent_step += 1

                if next_tok == ".":
                    sent_closed = True
                    break

            if not sent_closed:
                curr_sentence.append(".")
                if orchestrator is not None:
                    orchestrator.update_state(".")

            story_tokens.extend(curr_sentence)
            sentences_completed += 1

        if story_tokens and story_tokens[-1] != ".":
            story_tokens.append(".")

        wall_time_sec = time.perf_counter() - start_time

        # Calculate metrics
        story_metrics = compute_story_metrics(
            tokens=story_tokens,
            generation_time_sec=wall_time_sec,
            token_latencies_ms=token_latencies_ms,
        )

        detokenized = detokenize_story(story_tokens)

        story_record: Dict[str, Any] = {
            "story_id": f"UI_{int(time.time()*1000) % 1_000_000:06d}",
            "text": detokenized,
            "tokens": story_tokens,
            "token_count": len(story_tokens),
            "sentence_count": sentences_completed,
            "generation_time_sec": round(wall_time_sec, 3),
            "tokens_per_second": round(len(story_tokens) / max(0.001, wall_time_sec), 1),
            "mean_token_latency_ms": round(
                sum(token_latencies_ms) / max(1, len(token_latencies_ms)), 3
            ),
            "termination_reason": termination_reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": config,
            "metrics": story_metrics.to_dict(),
        }

        return story_record, step_diagnostics
