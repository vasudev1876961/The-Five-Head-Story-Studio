"""Side-by-side story comparison panel."""

from __future__ import annotations

import streamlit as st

from five_head_nlp.ui.session_manager import SessionManager


def render_comparison_panel(session_mgr: SessionManager) -> None:
    """Render side-by-side comparison between any two stories."""
    history = session_mgr.get_history()

    st.markdown("### ⚖️ Story Comparison Panel")
    st.caption("Compare narrative quality, length, and repetition metrics side-by-side.")

    if len(history) < 2:
        st.info("ℹ️ At least 2 stories must be generated in this session to compare side-by-side.")
        return

    story_options = {f"{s.get('story_id')} ({s.get('token_count')} tokens)": i for i, s in enumerate(history)}
    keys = list(story_options.keys())

    col_a, col_b = st.columns(2)
    with col_a:
        pick_a = st.selectbox("Story A", keys, index=0)
        idx_a = story_options[pick_a]
        story_a = history[idx_a]
    with col_b:
        pick_b = st.selectbox("Story B", keys, index=1 if len(keys) > 1 else 0)
        idx_b = story_options[pick_b]
        story_b = history[idx_b]

    st.divider()

    # Side-by-side metrics
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f"#### Story A: `{story_a.get('story_id')}`")
        cfg_a = story_a.get("configuration", {})
        st.caption(f"Mode: {cfg_a.get('mode')} • Smoother: {cfg_a.get('smoother_name')} • Decoder: {cfg_a.get('decoder_name')}")
        st.markdown(
            f"""
            <div class="story-card">
                <div class="story-text">{story_a.get('text', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"• **Tokens**: {story_a.get('token_count')} | **Sentences**: {story_a.get('sentence_count')}")
        rep_a = story_a.get("metrics", {}).get("repetition", {})
        st.markdown(f"• **Repeated Bigrams**: {rep_a.get('repeated_bigram_rate', 0.0)*100:.1f}%")

    with m_col2:
        st.markdown(f"#### Story B: `{story_b.get('story_id')}`")
        cfg_b = story_b.get("configuration", {})
        st.caption(f"Mode: {cfg_b.get('mode')} • Smoother: {cfg_b.get('smoother_name')} • Decoder: {cfg_b.get('decoder_name')}")
        st.markdown(
            f"""
            <div class="story-card">
                <div class="story-text">{story_b.get('text', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"• **Tokens**: {story_b.get('token_count')} | **Sentences**: {story_b.get('sentence_count')}")
        rep_b = story_b.get("metrics", {}).get("repetition", {})
        st.markdown(f"• **Repeated Bigrams**: {rep_b.get('repeated_bigram_rate', 0.0)*100:.1f}%")
