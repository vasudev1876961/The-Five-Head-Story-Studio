"""Streamlit Session State Manager for The Five-Head Story Studio."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


def _has_streamlit_context() -> bool:
    """Return True if running inside an active Streamlit runtime context."""
    try:
        import streamlit as st
        # Verify that session_state is accessible
        _ = st.session_state
        return True
    except Exception:
        return False


class SessionManager:
    """Manages application state, story histories, and user configurations.

    Gracefully falls back to an in-memory dictionary when invoked outside an active
    Streamlit runtime (e.g. during headless unit testing).
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "mode": "five_head",  # 'baseline' or 'five_head'
        "model_order": 3,
        "smoother_name": "Trigram Backoff",
        "smoother_params": {"discount": 0.5, "alpha": 1.0, "k": 0.5},
        "decoder_name": "Top-k",
        "decoder_params": {"k": 5, "temperature": 0.8},
        "seed": 42,
        "max_tokens": 100,
        "max_sentences": 8,
        "min_sentences": 3,
        "starter_prompt": "pip lived near",
        "weights": {
            "Grammar Scout": 0.40,
            "Character Keeper": 0.15,
            "Setting Painter": 0.15,
            "Action Booster": 0.10,
            "Repeat Blocker": 0.20,
        },
        "enabled_heads": {
            "Grammar Scout": True,
            "Character Keeper": True,
            "Setting Painter": True,
            "Action Booster": True,
            "Repeat Blocker": True,
        },
    }

    def __init__(self) -> None:
        """Initialize session state container."""
        self._fallback_store: Dict[str, Any] = {}
        self.initialize_session()

    def _get(self, key: str, default: Any = None) -> Any:
        if _has_streamlit_context():
            import streamlit as st
            return st.session_state.get(key, default)
        return self._fallback_store.get(key, default)

    def _set(self, key: str, value: Any) -> None:
        if _has_streamlit_context():
            import streamlit as st
            st.session_state[key] = value
        else:
            self._fallback_store[key] = value

    def initialize_session(self) -> None:
        """Set up initial session state variables if not already initialized."""
        if self._get("initialized") is None:
            self._set("initialized", True)
            self._set("current_story", None)
            self._set("current_steps", [])
            self._set("story_history", [])
            self._set("selected_token_idx", 0)
            self._set("generation_config", dict(self.DEFAULT_CONFIG))
            self._set("active_page", "Story Studio")

    def get_config(self) -> Dict[str, Any]:
        """Return the active generation configuration."""
        cfg = self._get("generation_config")
        if not cfg:
            cfg = dict(self.DEFAULT_CONFIG)
            self._set("generation_config", cfg)
        return cfg

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update active configuration values."""
        cfg = self.get_config()
        cfg.update(updates)
        self._set("generation_config", cfg)

    def get_current_story(self) -> Optional[Dict[str, Any]]:
        """Return currently generated story record (if any)."""
        return self._get("current_story")

    def set_current_story(
        self,
        story: Dict[str, Any],
        steps: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Store the newly generated story and diagnostic steps in session state."""
        self._set("current_story", story)
        self._set("current_steps", steps or [])
        self._set("selected_token_idx", 0)

        # Add to history (capped at most recent 25 stories)
        history = self._get("story_history") or []
        history.insert(0, story)
        self._set("story_history", history[:25])

    def get_current_steps(self) -> List[Dict[str, Any]]:
        """Return diagnostic token generation steps for the current story."""
        return self._get("current_steps") or []

    def get_history(self) -> List[Dict[str, Any]]:
        """Return list of all stories generated during this session."""
        return self._get("story_history") or []

    def clear_history(self) -> None:
        """Clear all session story history."""
        self._set("story_history", [])

    def get_selected_token_idx(self) -> int:
        """Return the currently inspected token index."""
        return self._get("selected_token_idx", 0)

    def set_selected_token_idx(self, idx: int) -> None:
        """Set the token index to inspect in the Five-Head Inspector."""
        self._set("selected_token_idx", max(0, idx))

    def get_active_page(self) -> str:
        """Return the currently selected page name."""
        return self._get("active_page", "Story Studio")

    def set_active_page(self, page_name: str) -> None:
        """Set the current active page."""
        self._set("active_page", page_name)
