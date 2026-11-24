"""Story generator module producing multi-sentence narratives."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from five_head_nlp.experiments.experiment_config import ExperimentConfig
from five_head_nlp.experiments.metrics import StoryMetrics, compute_story_metrics
from five_head_nlp.heads.head_orchestrator import HeadOrchestrator
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.prediction import (
    BaseDecoder,
    GreedyDecoder,
    TemperatureDecoder,
    TopKDecoder,
)
from five_head_nlp.prediction.predictor import NextTokenPredictor
from five_head_nlp.smoothing.backoff import BackoffSmoother
from five_head_nlp.smoothing.base_smoother import BaseSmoother
from five_head_nlp.smoothing.laplace import LaplaceSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader


def detokenize_story(tokens: Sequence[str]) -> str:
    """Reconstruct clean readable narrative text from a sequence of tokens.

    Capitalizes sentence beginnings and attaches punctuation directly to preceding words.
    """
    if not tokens:
        return ""

    out_words: List[str] = []
    capitalize_next = True

    for tok in tokens:
        if tok in {".", ",", "!", "?"}:
            if out_words:
                out_words[-1] = out_words[-1] + tok
            else:
                out_words.append(tok)
            if tok in {".", "!", "?"}:
                capitalize_next = True
        else:
            word = tok.capitalize() if capitalize_next else tok
            out_words.append(word)
            capitalize_next = False

    return " ".join(out_words)


@dataclass
class GeneratedStory:
    """Structured representation of a generated multi-sentence story."""

    story_id: str
    config_id: str
    config_name: str
    seed: int
    tokens: List[str]
    text: str
    sentence_count: int
    token_count: int
    generation_time_sec: float
    token_latencies_ms: List[float]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert story to dictionary for JSON/JSONL serialization."""
        return {
            "story_id": self.story_id,
            "config_id": self.config_id,
            "config_name": self.config_name,
            "seed": self.seed,
            "tokens": self.tokens,
            "text": self.text,
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
            "generation_time_sec": round(self.generation_time_sec, 4),
            "metrics": self.metrics,
        }


@dataclass
class StoryFailureRecord:
    """Record of a failed or aborted story generation attempt."""

    story_id: str
    config_id: str
    seed: int
    reason: str
    tokens_generated: int
    partial_tokens: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert failure record to dictionary for JSONL serialization."""
        return asdict(self)


class StoryGenerator:
    """Generates multi-sentence stories under a specific experimental condition."""

    def __init__(
        self,
        config: ExperimentConfig,
        model: Optional[TrigramModel] = None,
        loader: Optional[CorpusLoader] = None,
    ) -> None:
        """Initialize generator with experimental configuration.

        Parameters
        ----------
        config : ExperimentConfig
            Configuration specifying smoother, heads, decoder, and length constraints.
        model : Optional[TrigramModel], default=None
            Pre-fitted TrigramModel. If None, fitted from loader.
        loader : Optional[CorpusLoader], default=None
            CorpusLoader instance. If None, loads canonical corpus.
        """
        self.config = config
        self.loader = loader or CorpusLoader().load()

        if model is None:
            self.model = TrigramModel()
            self.model.fit_corpus(self.loader)
        else:
            self.model = model

        # Setup Smoother
        self.smoother: Optional[BaseSmoother] = None
        if config.smoother_type == "laplace":
            self.smoother = LaplaceSmoother(model=self.model)
        elif config.smoother_type == "backoff":
            discount = config.smoother_params.get("discount", 0.5)
            self.smoother = BackoffSmoother(discount=discount, model=self.model)

        # Setup Orchestrator for five_head mode
        self.orchestrator: Optional[HeadOrchestrator] = None
        if config.mode == "five_head":
            self.orchestrator = HeadOrchestrator(config=config.scoring_config)

        # Setup Predictor
        self.predictor = NextTokenPredictor(
            model=self.model,
            smoother=self.smoother,
            orchestrator=self.orchestrator,
        )

        # Extract 2-token sentence starters from corpus
        self.corpus_starters: List[Tuple[str, ...]] = []
        for sent in self.loader.get_sentences():
            if len(sent) >= 2:
                self.corpus_starters.append((sent[0], sent[1]))
            elif len(sent) == 1:
                self.corpus_starters.append((sent[0],))

        if not self.corpus_starters:
            self.corpus_starters = [("the", "clever"), ("pip", "lived"), ("milo", "found")]

    def _create_decoder(self, rng: random.Random) -> BaseDecoder:
        """Create decoder instance isolated to the provided RNG."""
        dtype = self.config.decoder_type
        dparams = self.config.decoder_params

        if dtype == "greedy":
            return GreedyDecoder()
        elif dtype == "top_k":
            k = dparams.get("k", 5)
            dec_seed = rng.randint(1, 10_000_000)
            return TopKDecoder(k=k, seed=dec_seed)
        elif dtype == "temperature":
            temp = dparams.get("temperature", 0.8)
            dec_seed = rng.randint(1, 10_000_000)
            return TemperatureDecoder(temperature=temp, seed=dec_seed)
        else:
            return GreedyDecoder()

    def generate_story(
        self,
        seed: int,
        story_id: str,
    ) -> Tuple[Optional[GeneratedStory], Optional[StoryFailureRecord]]:
        """Generate a complete multi-sentence story conforming to constraints.

        Parameters
        ----------
        seed : int
            Isolated seed for this story generation.
        story_id : str
            Unique story identifier (e.g. 'E1_001').

        Returns
        -------
        Tuple[Optional[GeneratedStory], Optional[StoryFailureRecord]]
            GeneratedStory if successful, or StoryFailureRecord if failed.
        """
        rng = random.Random(seed)
        decoder = self._create_decoder(rng)

        if self.orchestrator is not None:
            self.orchestrator.reset_state()

        story_tokens: List[str] = []
        token_latencies_ms: List[float] = []
        sentences_completed = 0
        consecutive_dead_ends = 0
        max_sentences = 25

        start_time = time.perf_counter()

        while (
            len(story_tokens) < self.config.target_min_tokens
            or sentences_completed < self.config.min_sentences
        ) and sentences_completed < max_sentences:

            if len(story_tokens) >= self.config.max_total_tokens:
                break

            # Select sentence starter deterministically via isolated rng
            starter = rng.choice(self.corpus_starters)
            curr_sent_tokens = list(starter)

            # Update orchestrator state with initial prompt tokens
            if self.orchestrator is not None:
                for tok in curr_sent_tokens:
                    self.orchestrator.update_state(tok)

            sent_step = 0
            max_sent_steps = 35
            sent_closed = False
            ngram_history: List[Tuple[str, str]] = []

            while sent_step < max_sent_steps and not sent_closed:
                # N-gram context: last 2 tokens of the current sentence
                ctx_window = min(len(curr_sent_tokens), self.model.n - 1)
                context = tuple(curr_sent_tokens[-ctx_window:])

                t_token_start = time.perf_counter()
                pred = self.predictor.predict_next(
                    context=context,
                    decoder=decoder,
                    mode=self.config.mode,
                    orchestrator=self.orchestrator,
                )
                t_token_elapsed = (time.perf_counter() - t_token_start) * 1000.0

                next_tok = pred.selected_token
                token_latencies_ms.append(t_token_elapsed)

                if next_tok is None:
                    # Context dead end (e.g. unsmoothed MLE unseen context)
                    consecutive_dead_ends += 1
                    curr_sent_tokens.append(".")
                    if self.orchestrator is not None:
                        self.orchestrator.update_state(".")
                    sent_closed = True
                    break

                consecutive_dead_ends = 0

                # Detect immediate attractor loops within a single sentence
                if len(curr_sent_tokens) >= 2:
                    current_bigram = (curr_sent_tokens[-1].lower(), next_tok.lower())
                    ngram_history.append(current_bigram)
                    if len(ngram_history) >= 4 and ngram_history[-1] == ngram_history[-2] == ngram_history[-3]:
                        # Loop detected: force sentence termination with '.'
                        curr_sent_tokens.append(".")
                        if self.orchestrator is not None:
                            self.orchestrator.update_state(".")
                        sent_closed = True
                        break

                curr_sent_tokens.append(next_tok)
                if self.orchestrator is not None:
                    self.orchestrator.update_state(next_tok)

                sent_step += 1

                if next_tok == ".":
                    sent_closed = True
                    break

            if not sent_closed:
                curr_sent_tokens.append(".")
                if self.orchestrator is not None:
                    self.orchestrator.update_state(".")

            story_tokens.extend(curr_sent_tokens)
            sentences_completed += 1

            if consecutive_dead_ends >= 5:
                # Unrecoverable dead ends
                fail_rec = StoryFailureRecord(
                    story_id=story_id,
                    config_id=self.config.config_id,
                    seed=seed,
                    reason="Excessive consecutive dead ends in probability engine.",
                    tokens_generated=len(story_tokens),
                    partial_tokens=story_tokens,
                )
                return None, fail_rec

        # Ensure terminal period
        if story_tokens and story_tokens[-1] != ".":
            story_tokens.append(".")

        total_time_sec = time.perf_counter() - start_time

        # Calculate metrics
        story_metrics = compute_story_metrics(
            tokens=story_tokens,
            generation_time_sec=total_time_sec,
            token_latencies_ms=token_latencies_ms,
        )

        narrative_text = detokenize_story(story_tokens)

        story = GeneratedStory(
            story_id=story_id,
            config_id=self.config.config_id,
            config_name=self.config.name,
            seed=seed,
            tokens=story_tokens,
            text=narrative_text,
            sentence_count=sentences_completed,
            token_count=len(story_tokens),
            generation_time_sec=total_time_sec,
            token_latencies_ms=token_latencies_ms,
            metrics=story_metrics.to_dict(),
        )

        return story, None
