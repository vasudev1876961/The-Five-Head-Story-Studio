"""Story history and bulk session export component."""

from __future__ import annotations

import streamlit as st

from five_head_nlp.ui.export_service import ExportService
from five_head_nlp.ui.session_manager import SessionManager


def render_story_history(session_mgr: SessionManager) -> None:
    """Render session story history table and export options."""
    history = session_mgr.get_history()

    st.markdown("### 📜 Session Story History")
    st.caption("View and reload all narratives generated during this session.")

    if not history:
        st.info("ℹ️ No stories generated yet during this session.")
        return

    st.markdown(f"**Total Stories in Session:** `{len(history)}`")

    # Story history table
    for idx, story in enumerate(history):
        sid = story.get("story_id", f"Story #{idx+1}")
        t_cnt = story.get("token_count", 0)
        s_cnt = story.get("sentence_count", 0)
        cfg = story.get("configuration", {})
        mode = cfg.get("mode", "five_head")
        smoother = cfg.get("smoother_name", "Trigram Backoff")
        decoder = cfg.get("decoder_name", "Top-k")

        with st.expander(f"📖 {sid} — {t_cnt} tokens, {s_cnt} sentences ({mode}, {smoother}, {decoder})"):
            st.markdown(f"> {story.get('text', '')}")
            st.caption(f"Generated at {story.get('timestamp')} • Throughput: {story.get('tokens_per_second', 0.0):.1f} tok/s")

            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                txt = ExportService.export_plain_text(story)
                st.download_button("📥 Text (.txt)", txt, file_name=f"{sid}.txt", key=f"dl_txt_{idx}")
            with col2:
                md = ExportService.export_markdown(story)
                st.download_button("📥 Markdown (.md)", md, file_name=f"{sid}.md", key=f"dl_md_{idx}")

    st.divider()
    if st.button("🗑️ Clear Session History", type="secondary"):
        session_mgr.clear_history()
        st.success("Session history cleared.")
        st.rerun()
