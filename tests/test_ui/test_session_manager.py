"""Unit tests for UI SessionManager."""

import pytest

from five_head_nlp.ui.session_manager import SessionManager


class TestSessionManager:
    """Test session state container, defaults, history tracking, and active page."""

    def test_session_manager_initialization_defaults(self):
        mgr = SessionManager()
        cfg = mgr.get_config()
        assert cfg["mode"] == "five_head"
        assert cfg["smoother_name"] == "Trigram Backoff"
        assert cfg["decoder_name"] == "Top-k"
        assert cfg["seed"] == 42
        assert mgr.get_current_story() is None
        assert mgr.get_current_steps() == []
        assert mgr.get_history() == []
        assert mgr.get_active_page() == "Story Studio"

    def test_update_config(self):
        mgr = SessionManager()
        mgr.update_config({"mode": "baseline", "seed": 999})
        cfg = mgr.get_config()
        assert cfg["mode"] == "baseline"
        assert cfg["seed"] == 999

    def test_set_and_get_current_story_and_history(self):
        mgr = SessionManager()
        story = {
            "story_id": "STORY_001",
            "text": "Milo found a key.",
            "token_count": 5,
        }
        steps = [{"step_index": 1, "token": "milo"}]
        mgr.set_current_story(story, steps)

        assert mgr.get_current_story() == story
        assert mgr.get_current_steps() == steps
        assert len(mgr.get_history()) == 1
        assert mgr.get_history()[0] == story

    def test_history_capping_and_clear(self):
        mgr = SessionManager()
        for i in range(30):
            mgr.set_current_story({"story_id": f"S_{i}"})
        # Capped at 25 most recent
        assert len(mgr.get_history()) == 25
        assert mgr.get_history()[0]["story_id"] == "S_29"

        mgr.clear_history()
        assert mgr.get_history() == []

    def test_selected_token_idx_and_page(self):
        mgr = SessionManager()
        mgr.set_selected_token_idx(5)
        assert mgr.get_selected_token_idx() == 5

        mgr.set_active_page("Five-Head Inspector")
        assert mgr.get_active_page() == "Five-Head Inspector"
