"""Sidebar component for navigation and configuration overview."""

from __future__ import annotations

import streamlit as st

from five_head_nlp.ui.session_manager import SessionManager


def render_sidebar(session_mgr: SessionManager) -> str:
    """Render the application sidebar with page navigation and configuration summary.

    Parameters
    ----------
    session_mgr : SessionManager
        Active session manager instance.

    Returns
    -------
    str
        Selected active page name.
    """
    with st.sidebar:
        st.markdown("### 🎲 Story Studio")
        st.markdown(
            "<span style='color:#8b949e; font-size:0.85rem;'>"
            "The Five-Head NLP Story-Generation System</span>",
            unsafe_allow_html=True,
        )
        st.divider()

        pages = [
            "Story Studio",
            "Five-Head Inspector",
            "Hamilton d10 Visualizer",
            "Narrative Tracker",
            "Experimental Analytics",
            "Story History & Export",
            "Model Architecture Info",
        ]

        current_page = session_mgr.get_active_page()
        selected_page = st.radio(
            "Navigation",
            pages,
            index=pages.index(current_page) if current_page in pages else 0,
            label_visibility="collapsed",
        )
        session_mgr.set_active_page(selected_page)

        st.divider()

        # Active Configuration Summary
        cfg = session_mgr.get_config()
        st.markdown("#### ⚙️ Active Setup")
        mode = "Five-Head System" if cfg.get("mode") == "five_head" else "Baseline MLE"
        st.markdown(f"**Mode**: `{mode}`")
        st.markdown(f"**Smoother**: `{cfg.get('smoother_name')}`")
        st.markdown(f"**Decoder**: `{cfg.get('decoder_name')}`")

        if cfg.get("mode") == "five_head":
            st.markdown("**Head Weights (d10)**:")
            weights = cfg.get("weights", {})
            st.markdown(
                f"<div style='font-size:0.82rem;'>"
                f"• Grammar: {weights.get('Grammar Scout', 0.4)*10:.1f} pts<br>"
                f"• Character: {weights.get('Character Keeper', 0.15)*10:.1f} pts<br>"
                f"• Setting: {weights.get('Setting Painter', 0.15)*10:.1f} pts<br>"
                f"• Action: {weights.get('Action Booster', 0.10)*10:.1f} pts<br>"
                f"• Repeat: {weights.get('Repeat Blocker', 0.20)*10:.1f} pts"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown(
            "<div style='color:#8b949e; font-size:0.75rem; text-align:center;'>"
            "NLP Game Project • Phase 6<br>34-Sentence Canonical Corpus</div>",
            unsafe_allow_html=True,
        )

    return selected_page
