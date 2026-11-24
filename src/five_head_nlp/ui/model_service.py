"""Model caching and initialization service for the Streamlit UI."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from five_head_nlp.heads.head_orchestrator import HeadOrchestrator
from five_head_nlp.heads.scoring_config import ScoringConfig
from five_head_nlp.models.base_ngram import BaseNGramModel
from five_head_nlp.models.bigram import BigramModel
from five_head_nlp.models.trigram import TrigramModel
from five_head_nlp.models.unigram import UnigramModel
from five_head_nlp.prediction.predictor import NextTokenPredictor
from five_head_nlp.smoothing.add_k import AddKSmoother
from five_head_nlp.smoothing.backoff import BackoffSmoother
from five_head_nlp.smoothing.base_smoother import BaseSmoother
from five_head_nlp.smoothing.good_turing import GoodTuringSmoother
from five_head_nlp.smoothing.interpolation import InterpolationSmoother
from five_head_nlp.smoothing.laplace import LaplaceSmoother
from five_head_nlp.tokenization.corpus_loader import CorpusLoader

# Global fallback cache for headless / testing runs
_MODEL_CACHE: Dict[str, Any] = {}


def _get_corpus_loader_impl() -> CorpusLoader:
    loader = CorpusLoader()
    loader.load()
    return loader


def _get_model_impl(order: int) -> BaseNGramModel:
    loader = _get_corpus_loader_impl()
    if order == 1:
        model: BaseNGramModel = UnigramModel()
    elif order == 2:
        model = BigramModel()
    elif order == 3:
        model = TrigramModel()
    else:
        raise ValueError(f"Unsupported N-gram order: {order}")
    model.fit_corpus(loader)
    return model


# Wrap with Streamlit caching if running in Streamlit, else standard memoization
def get_corpus_loader() -> CorpusLoader:
    """Retrieve cached CorpusLoader instance."""
    try:
        import streamlit as st
        @st.cache_resource
        def _cached_loader() -> CorpusLoader:
            return _get_corpus_loader_impl()
        return _cached_loader()
    except Exception:
        if "loader" not in _MODEL_CACHE:
            _MODEL_CACHE["loader"] = _get_corpus_loader_impl()
        return _MODEL_CACHE["loader"]


def get_model(order: int = 3) -> BaseNGramModel:
    """Retrieve cached N-gram language model fitted on story corpus."""
    try:
        import streamlit as st
        @st.cache_resource
        def _cached_model(n: int) -> BaseNGramModel:
            return _get_model_impl(n)
        return _cached_model(order)
    except Exception:
        key = f"model_{order}"
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = _get_model_impl(order)
        return _MODEL_CACHE[key]


class ModelService:
    """Service providing language models, smoothers, predictors, and orchestrators."""

    SMOOTHER_CHOICES: Dict[str, str] = {
        "Trigram Backoff (Recommended)": "backoff",
        "Laplace (Add-1)": "laplace",
        "Add-k Smoothing": "add_k",
        "Good-Turing": "good_turing",
        "Linear Interpolation": "interpolation",
        "Unsmoothed MLE": "none",
    }

    DECODER_CHOICES: Tuple[str, ...] = ("Top-k", "Greedy", "Temperature")

    @classmethod
    def get_supported_smoothers(cls) -> Dict[str, str]:
        """Return map of user-friendly label to smoother code."""
        return dict(cls.SMOOTHER_CHOICES)

    @classmethod
    def create_smoother(
        cls,
        smoother_name: str,
        model: BaseNGramModel,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[BaseSmoother]:
        """Create and configure a probability smoother for the given model.

        Parameters
        ----------
        smoother_name : str
            Friendly name or key ('backoff', 'laplace', 'add_k', etc.)
        model : BaseNGramModel
            Trained language model.
        params : Optional[Dict[str, Any]]
            Optional smoother parameters.

        Returns
        -------
        Optional[BaseSmoother]
            Smoother instance or None if 'none' / unsmoothed MLE.
        """
        p = params or {}
        code = cls.SMOOTHER_CHOICES.get(smoother_name, smoother_name.lower())

        if code in {"none", "mle", "unsmoothed mle"}:
            return None
        elif code in {"laplace", "laplace (add-1)"}:
            return LaplaceSmoother(model=model)
        elif code in {"add_k", "add-k smoothing"}:
            k = float(p.get("k", 0.5))
            return AddKSmoother(k=k, model=model)
        elif code in {"good_turing", "good-turing"}:
            return GoodTuringSmoother(model=model)
        elif code in {"backoff", "trigram backoff (recommended)", "trigram backoff"}:
            discount = float(p.get("discount", 0.5))
            return BackoffSmoother(discount=discount, model=model)
        elif code in {"interpolation", "linear interpolation"}:
            lambdas = p.get("lambdas", (0.1, 0.3, 0.6))
            return InterpolationSmoother(lambdas=lambdas, model=model)
        else:
            # Default to Trigram Backoff
            return BackoffSmoother(discount=0.5, model=model)

    @classmethod
    def create_orchestrator(
        cls,
        weights: Optional[Dict[str, float]] = None,
        enabled_heads: Optional[Dict[str, bool]] = None,
    ) -> HeadOrchestrator:
        """Create and configure a HeadOrchestrator instance with custom weights.

        Parameters
        ----------
        weights : Optional[Dict[str, float]]
            Head weights for Grammar, Character, Setting, Action, Repeat.
        enabled_heads : Optional[Dict[str, bool]]
            Toggles for enabling each head.

        Returns
        -------
        HeadOrchestrator
            Configured orchestrator instance.
        """
        w = weights or {
            "Grammar Scout": 0.40,
            "Character Keeper": 0.15,
            "Setting Painter": 0.15,
            "Action Booster": 0.10,
            "Repeat Blocker": 0.20,
        }
        en = enabled_heads or {
            "Grammar Scout": True,
            "Character Keeper": True,
            "Setting Painter": True,
            "Action Booster": True,
            "Repeat Blocker": True,
        }

        # Normalize weights so sum == 1.0
        tot = sum(w.values())
        if tot > 0:
            norm_w = {k: v / tot for k, v in w.items()}
        else:
            norm_w = {k: 0.20 for k in w}

        cfg = ScoringConfig(
            grammar_weight=norm_w.get("Grammar Scout", 0.40),
            character_weight=norm_w.get("Character Keeper", 0.15),
            setting_weight=norm_w.get("Setting Painter", 0.15),
            action_weight=norm_w.get("Action Booster", 0.10),
            repeat_weight=norm_w.get("Repeat Blocker", 0.20),
            enable_grammar=en.get("Grammar Scout", True),
            enable_character=en.get("Character Keeper", True),
            enable_setting=en.get("Setting Painter", True),
            enable_action=en.get("Action Booster", True),
            enable_repeat=en.get("Repeat Blocker", True),
        )
        return HeadOrchestrator(config=cfg)

    @classmethod
    def create_predictor(
        cls,
        model: BaseNGramModel,
        smoother: Optional[BaseSmoother] = None,
        orchestrator: Optional[HeadOrchestrator] = None,
    ) -> NextTokenPredictor:
        """Instantiate NextTokenPredictor with the specified components."""
        return NextTokenPredictor(
            model=model,
            smoother=smoother,
            orchestrator=orchestrator,
        )
