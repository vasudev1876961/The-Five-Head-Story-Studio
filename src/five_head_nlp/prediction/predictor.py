"""Next-token prediction engine coordinating language models, smoothing, and decoding."""

from __future__ import annotations

import math
import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from five_head_nlp.models.base_ngram import BaseNGramModel
from five_head_nlp.prediction.candidate_ranker import CandidateRanker
from five_head_nlp.prediction.decoding import BaseDecoder
from five_head_nlp.prediction.generation_state import GenerationState
from five_head_nlp.prediction.greedy import GreedyDecoder
from five_head_nlp.prediction.prediction_result import (
    CandidateToken,
    PredictionResult,
)
from five_head_nlp.prediction.temperature import TemperatureDecoder
from five_head_nlp.prediction.top_k import TopKDecoder
from five_head_nlp.smoothing.base_smoother import BaseSmoother
from five_head_nlp.tokenization.vocabulary import Vocabulary


class NextTokenPredictor:
    """Reusable next-token predictor decoupled from scoring heads and story generation.

    Integrates:
        - Trained BaseNGramModel (Unigram, Bigram, Trigram)
        - Optional BaseSmoother (Laplace, Add-k, Good-Turing, Backoff, Interpolation)
        - CandidateRanker with deterministic tie-breaking
        - Configurable Decoding Strategies (Greedy, Top-k, Temperature)
        - GenerationState for sequence rollout with sentence-boundary enforcement
    """

    def __init__(
        self,
        model: BaseNGramModel,
        smoother: Optional[BaseSmoother] = None,
        ranker: Optional[CandidateRanker] = None,
        orchestrator: Optional[Any] = None,
    ) -> None:
        """Initialize NextTokenPredictor.

        Parameters
        ----------
        model : BaseNGramModel
            Trained language model.
        smoother : Optional[BaseSmoother], default=None
            Optional probability smoother. If None, uses unsmoothed MLE distributions.
        ranker : Optional[CandidateRanker], default=None
            Candidate ranker instance (defaults to CandidateRanker()).
        orchestrator : Optional[Any], default=None
            Optional HeadOrchestrator for multi-head candidate evaluation and scoring.

        Raises
        ------
        ValueError
            If model is not fitted.
        """
        if not model.is_fitted:
            raise ValueError("Cannot initialize NextTokenPredictor with an unfitted model.")

        self.model: BaseNGramModel = model
        self.smoother: Optional[BaseSmoother] = smoother
        self.ranker: CandidateRanker = ranker or CandidateRanker()
        self.orchestrator: Optional[Any] = orchestrator
        assert model.vocabulary is not None
        self.vocabulary: Vocabulary = model.vocabulary

    @property
    def model_order(self) -> int:
        """Return the maximum context order supported by the model."""
        return self.model.n

    @property
    def smoother_name(self) -> Optional[str]:
        """Return the name of the active smoother if attached."""
        return self.smoother.name if self.smoother else None

    def _normalize_context(
        self, context: Optional[Union[str, Sequence[str]]]
    ) -> Tuple[str, ...]:
        """Normalize context to a tuple of strings."""
        if context is None:
            return ()
        if isinstance(context, str):
            return (context,)
        return tuple(context)

    def _truncate_context_for_model(
        self, context: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        """Truncate context to the maximum history supported by model order (n - 1)."""
        max_context_len = max(0, self.model.n - 1)
        if len(context) > max_context_len:
            return context[-max_context_len:] if max_context_len > 0 else ()
        return context

    def get_candidate_distribution(
        self, context: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, float]:
        """Retrieve probability distribution over candidate next tokens.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding tokens.

        Returns
        -------
        Dict[str, float]
            Map of {token: probability}.
            If smoother is present, covers all vocabulary tokens (|V| = 120).
            If unsmoothed MLE, covers observed continuations, or empty dict if context unobserved.

        Raises
        ------
        ValueError
            If computed distribution contains invalid probabilities (< 0 or NaN).
        """
        ctx_tuple = self._normalize_context(context)

        # Sentence-terminal check: if context ends with a period, no valid outgoing transition exists
        if ctx_tuple and ctx_tuple[-1] == ".":
            return {}

        model_ctx = self._truncate_context_for_model(ctx_tuple)

        if self.smoother is not None:
            distribution = self.smoother.get_distribution(context=model_ctx)
        else:
            distribution = self.model.get_candidate_distribution(context=model_ctx)

        # Validate distribution
        for token, prob in distribution.items():
            if math.isnan(prob) or math.isinf(prob) or prob < 0.0:
                raise ValueError(
                    f"Invalid probability for token {token!r}: {prob} in context {ctx_tuple}"
                )

        return distribution

    def rank_candidates(
        self,
        context: Optional[Union[str, Sequence[str]]] = None,
        top_n: Optional[int] = None,
    ) -> List[CandidateToken]:
        """Compute candidate distribution and rank candidates in descending probability order.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context.
        top_n : Optional[int], default=None
            Maximum number of ranked candidates to return.

        Returns
        -------
        List[CandidateToken]
            Ranked candidate tokens.
        """
        dist = self.get_candidate_distribution(context=context)
        return self.rank_candidates_from_distribution(dist, top_n=top_n)

    def rank_candidates_from_distribution(
        self,
        distribution: Dict[str, float],
        top_n: Optional[int] = None,
    ) -> List[CandidateToken]:
        """Rank candidates from an existing probability distribution."""
        return self.rank_ranker_result(distribution, top_n=top_n)

    def rank_ranker_result(
        self,
        distribution: Dict[str, float],
        top_n: Optional[int] = None,
    ) -> List[CandidateToken]:
        """Helper to invoke the ranker on distribution."""
        return self.ranker.rank_candidates(distribution, top_n=top_n)

    def create_decoder(
        self,
        strategy: str = "greedy",
        k: int = 5,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> BaseDecoder:
        """Instantiate decoder corresponding to specified strategy name.

        Parameters
        ----------
        strategy : str, default='greedy'
            One of 'greedy', 'top_k', or 'temperature'.
        k : int, default=5
            Parameter for top_k decoding.
        temperature : float, default=1.0
            Parameter for temperature sampling.
        seed : Optional[int], default=None
            Seed for isolated random number generator.

        Returns
        -------
        BaseDecoder
            Configured decoder instance.

        Raises
        ------
        ValueError
            If strategy is unknown.
        """
        strat = strategy.lower().strip()
        if strat in ("greedy", "argmax"):
            return GreedyDecoder(seed=seed)
        elif strat in ("top_k", "topk"):
            return TopKDecoder(k=k, seed=seed)
        elif strat in ("temperature", "temp"):
            return TemperatureDecoder(temperature=temperature, seed=seed)
        else:
            raise ValueError(
                f"Unknown decoding strategy: {strategy!r}. Expected 'greedy', 'top_k', or 'temperature'."
            )

    def predict_next(
        self,
        context: Optional[Union[str, Sequence[str]]] = None,
        strategy: str = "greedy",
        k: int = 5,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        decoder: Optional[BaseDecoder] = None,
        top_n_candidates: Optional[int] = None,
        mode: str = "baseline",
        orchestrator: Optional[Any] = None,
        state: Optional[GenerationState] = None,
    ) -> PredictionResult:
        """Perform a single next-token prediction step.

        Parameters
        ----------
        context : Optional[Union[str, Sequence[str]]], default=None
            Preceding context tokens.
        strategy : str, default='greedy'
            Decoding strategy ('greedy', 'top_k', or 'temperature').
        k : int, default=5
            k parameter for top_k decoding.
        temperature : float, default=1.0
            Temperature parameter for temperature sampling.
        seed : Optional[int], default=None
            Random seed for reproducibility.
        decoder : Optional[BaseDecoder], default=None
            Pre-configured decoder instance. If provided, overrides strategy/k/temperature/seed.
        top_n_candidates : Optional[int], default=None
            Number of top candidates to store in result.
        mode : str, default='baseline'
            Prediction mode: 'baseline' (raw statistical N-gram) or 'five_head' (multi-head scoring).
        orchestrator : Optional[Any], default=None
            Optional HeadOrchestrator instance for 'five_head' mode.
        state : Optional[GenerationState], default=None
            Current generation state tracking token history.

        Returns
        -------
        PredictionResult
            Standardized prediction output with selected token and candidates.
        """
        start_time = time.perf_counter()
        ctx_tuple = self._normalize_context(context)

        # Check for terminal period in context
        if ctx_tuple and ctx_tuple[-1] == ".":
            return PredictionResult(
                selected_token=None,
                context=ctx_tuple,
                candidates=[],
                strategy=strategy,
                model_order=self.model.n,
                smoother_name=self.smoother_name,
                is_terminal=True,
                distribution_sum=0.0,
                metadata={"reason": "context_ended_with_period", "latency_ms": 0.0, "mode": mode},
            )

        distribution = self.get_candidate_distribution(context=ctx_tuple)
        dist_sum = sum(distribution.values())

        if not distribution:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return PredictionResult(
                selected_token=None,
                context=ctx_tuple,
                candidates=[],
                strategy=strategy,
                model_order=self.model.n,
                smoother_name=self.smoother_name,
                is_terminal=False,
                distribution_sum=0.0,
                metadata={"reason": "empty_candidate_distribution", "latency_ms": elapsed_ms, "mode": mode},
            )

        candidates = self.rank_candidates_from_distribution(distribution, top_n=top_n_candidates)

        active_decoder = decoder or self.create_decoder(
            strategy=strategy, k=k, temperature=temperature, seed=seed
        )

        if mode == "five_head":
            active_orch = orchestrator or self.orchestrator
            if active_orch is None:
                from five_head_nlp.heads.head_orchestrator import HeadOrchestrator
                active_orch = HeadOrchestrator()

            orch_res = active_orch.evaluate_candidates(
                context=ctx_tuple, candidates=candidates, state=state
            )

            # Re-rank candidates according to fused multi-head scores
            fused_tokens: List[CandidateToken] = []
            fused_sum = sum(fc.fused_score for fc in orch_res.fused_candidates)
            for fc in orch_res.fused_candidates:
                prob_val = (fc.fused_score / fused_sum) if fused_sum > 0.0 else (1.0 / len(orch_res.fused_candidates))
                log_p = math.log(max(1e-12, prob_val))
                fused_tokens.append(
                    CandidateToken(
                        token=fc.token,
                        probability=prob_val,
                        rank=fc.rank,
                        log_probability=log_p,
                    )
                )

            effective_candidates = fused_tokens
            extra_meta = {
                "mode": "five_head",
                "orchestrator_result": orch_res.to_dict(),
            }
        else:
            effective_candidates = candidates
            extra_meta = {"mode": "baseline"}

        selected_token = active_decoder.select_token(effective_candidates)
        is_terminal = (selected_token == ".")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        meta = {
            "latency_ms": round(elapsed_ms, 4),
            "seed": getattr(active_decoder, "seed", None),
        }
        meta.update(extra_meta)

        return PredictionResult(
            selected_token=selected_token,
            context=ctx_tuple,
            candidates=effective_candidates,
            strategy=active_decoder.strategy_name,
            model_order=self.model.n,
            smoother_name=self.smoother_name,
            is_terminal=is_terminal,
            distribution_sum=dist_sum,
            metadata=meta,
        )

    def generate(
        self,
        prompt: Optional[Sequence[str]] = None,
        max_tokens: int = 25,
        strategy: str = "greedy",
        k: int = 5,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        decoder: Optional[BaseDecoder] = None,
        mode: str = "baseline",
        orchestrator: Optional[Any] = None,
    ) -> Tuple[List[str], List[PredictionResult]]:
        """Roll out an autoregressive sequence of predictions until terminal '.' or max_tokens.

        Parameters
        ----------
        prompt : Optional[Sequence[str]], default=None
            Initial prompt tokens.
        max_tokens : int, default=25
            Maximum generated tokens.
        strategy : str, default='greedy'
            Decoding strategy.
        k : int, default=5
            Top-k parameter.
        temperature : float, default=1.0
            Temperature parameter.
        seed : Optional[int], default=None
            Initial seed for reproducibility across the sequence rollout.
        decoder : Optional[BaseDecoder], default=None
            Pre-configured decoder instance.
        mode : str, default='baseline'
            Prediction mode ('baseline' or 'five_head').
        orchestrator : Optional[Any], default=None
            Optional HeadOrchestrator for 'five_head' mode.

        Returns
        -------
        Tuple[List[str], List[PredictionResult]]
            1. Full generated token list (prompt + generated).
            2. List of PredictionResult objects for each step.
        """
        state = GenerationState(prompt=prompt, max_tokens=max_tokens)
        results: List[PredictionResult] = []

        active_decoder = decoder or self.create_decoder(
            strategy=strategy, k=k, temperature=temperature, seed=seed
        )

        active_orch = None
        if mode == "five_head":
            active_orch = orchestrator or self.orchestrator
            if active_orch is None:
                from five_head_nlp.heads.head_orchestrator import HeadOrchestrator
                active_orch = HeadOrchestrator()
            active_orch.reset_state()

        while state.can_continue():
            # Get rolling context (last n-1 tokens for N-gram conditioning)
            current_ctx = state.get_context(window_size=self.model.n - 1)

            step_result = self.predict_next(
                context=current_ctx,
                decoder=active_decoder,
                mode=mode,
                orchestrator=active_orch,
                state=state,
            )
            results.append(step_result)

            if step_result.selected_token is None:
                # Dead-end context without candidates
                break

            if active_orch is not None:
                active_orch.update_state(step_result.selected_token)

            state.append(step_result.selected_token)

        return state.all_tokens, results
